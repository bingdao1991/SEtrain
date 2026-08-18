"""
Objective PCL diagnostics for ULUNAS_PCL_Spec (point vs conv2d).

Goal: locate WHY pcl_raw stays above chance, without guessing hyperparameters.

Probes (each has a falsifiable expected outcome):
  P0  Loss-unit test     : synthetic separable q/p/n → pcl_raw should << chance
  P1  Oracle query       : set query:=positive on real feats → pcl_raw should ≈ 0
  P2  Retrieval accuracy : fraction of patches with cos(q,p) > cos(q,n_same)
  P3  Similarity gap     : mean cos(q,p) - mean cos(q,n)  (should be > 0 if learning)
  P4  Sampler A/B        : SAME feat maps through PatchSampleF vs PatchSampleFConv2d
  P5  Pre/post-conv corr : corr(speech_hat, clean) at same TF before vs after conv
  P6  Collapse stats     : embedding std / pairwise cos within query

Usage:
  python debug_pcl_spec.py -C configs/cfg_train_pcl_spec_phase3_conv2d.yaml \\
      --checkpoint <phase2_or_failing_phase3.tar> -D 0 --num_batches 8

  # Experiment A: GT-noise negatives (objective)
  python debug_pcl_spec.py -C configs/cfg_train_pcl_spec_phase3.yaml \\
      --checkpoint <point_phase3_best.tar> --load_pcl_weights \\
      --mask_mode as_is --negative_mode gt_noise -D 0 --num_batches 8

  # Ablation: post-hoc complementary masks (no retrain)
  python debug_pcl_spec.py -C configs/cfg_train_pcl_spec_phase3.yaml \\
      --checkpoint <point_phase3_best.tar> --load_pcl_weights \\
      --mask_mode complement -D 0 --num_batches 8
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataloader import DNS3DualOutputDataset as Dataset
from models.pcl_modules import (
    PCLFeatureExtractor,
    PatchNCELoss,
    PatchSampleF,
    PatchSampleFConv2d,
)
from models.ulunas_pcl_spec import ULUNAS_PCL_Spec


def _chance(num_patches: int) -> float:
    return math.log(1.0 + num_patches)


def _mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: (N, D), already L2-normalized by sampler; still normalize for safety
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(-1).mean().item()


def _retrieval_acc(q: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> float:
    """Per-patch: is positive closer than the same-index negative?"""
    q = F.normalize(q, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    pos = (q * p).sum(-1)
    neg = (q * n).sum(-1)
    return (pos > neg).float().mean().item()


def _collapse_stats(x: torch.Tensor) -> Dict[str, float]:
    x = F.normalize(x, dim=-1)
    std = x.std(dim=0).mean().item()
    # mean off-diagonal cosine among first min(256,N) vectors
    n = min(256, x.shape[0])
    xx = x[:n]
    gram = xx @ xx.T
    eye = torch.eye(n, device=x.device, dtype=torch.bool)
    off = gram.masked_select(~eye)
    return {
        "emb_std": std,
        "mean_offdiag_cos": off.mean().item() if off.numel() else 0.0,
    }


@torch.no_grad()
def _tf_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson corr over flattened TF for batch mean of per-sample corr."""
    # a,b: (B,1,T,F)
    bsz = a.shape[0]
    corrs = []
    for i in range(bsz):
        x = a[i].reshape(-1).float()
        y = b[i].reshape(-1).float()
        x = x - x.mean()
        y = y - y.mean()
        denom = x.norm() * y.norm() + 1e-8
        corrs.append((x * y).sum() / denom)
    return float(torch.stack(corrs).mean().item())


def probe_p0_synthetic(num_patches: int, batch_size: int, device, temperature: float) -> Dict[str, float]:
    """Hand-crafted: q=p, n orthogonal-ish → loss should be near 0."""
    dim = 128
    q = F.normalize(torch.randn(batch_size * num_patches, dim, device=device), dim=-1)
    p = q.clone()
    n = F.normalize(torch.randn(batch_size * num_patches, dim, device=device), dim=-1)
    # Make n nearly orthogonal to q
    n = F.normalize(n - (n * q).sum(-1, keepdim=True) * q, dim=-1)
    loss_fn = PatchNCELoss(temperature=temperature, batch_size=batch_size).to(device)
    loss = loss_fn(q, p, n, batch_size=batch_size).mean().item()
    return {
        "pcl_raw": loss,
        "chance": _chance(num_patches),
        "retrieval_acc": _retrieval_acc(q, p, n),
        "cos_qp": _mean_cos(q, p),
        "cos_qn": _mean_cos(q, n),
    }


def probe_p1_oracle(
    extractor: PCLFeatureExtractor,
    loss_fn: PatchNCELoss,
    feat_gt: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    """Use GT speech as query AND positive; noise still negative."""
    # Build features with speech_hat := gt
    feats = extractor(feat_gt, feat_gt, feat_gt)
    # Force q=p by construction: both from feat_gt with shared ids.
    # For a stronger oracle, set q := p exactly:
    q = feats.f_positive
    p = feats.f_positive
    n = feats.f_negative
    # If speech==noise input, n≈p; use a random negative instead for oracle purity
    n = F.normalize(torch.randn_like(n), dim=-1)
    loss = loss_fn(q, p, n, batch_size=batch_size).mean().item()
    return {
        "pcl_raw": loss,
        "retrieval_acc": _retrieval_acc(q, p, n),
        "cos_qp": _mean_cos(q, p),
        "cos_qn": _mean_cos(q, n),
    }


def probe_real_batch(
    extractor: PCLFeatureExtractor,
    loss_fn: PatchNCELoss,
    feat_s: torch.Tensor,
    feat_n: torch.Tensor,
    feat_gt: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    feats = extractor(feat_s, feat_n, feat_gt)
    q, p, n = feats.f_query, feats.f_positive, feats.f_negative
    loss = loss_fn(q, p, n, batch_size=batch_size).mean().item()
    out = {
        "pcl_raw": loss,
        "chance": _chance(extractor.num_patches),
        "retrieval_acc": _retrieval_acc(q, p, n),
        "cos_qp": _mean_cos(q, p),
        "cos_qn": _mean_cos(q, n),
        "gap_qp_qn": _mean_cos(q, p) - _mean_cos(q, n),
    }
    out.update({f"q_{k}": v for k, v in _collapse_stats(q).items()})
    return out


def build_extractor(sampler_type: str, pcl_cfg, device) -> PCLFeatureExtractor:
    kwargs = dict(
        in_channels=pcl_cfg.get("in_channels", 1),
        embed_dim=pcl_cfg.get("embed_dim", 128),
        use_mlp=pcl_cfg.get("use_mlp", True),
    )
    if sampler_type == "conv2d":
        sampler = PatchSampleFConv2d(
            **kwargs, conv_kernel=pcl_cfg.get("conv_kernel", 3)
        ).to(device)
    else:
        sampler = PatchSampleF(**kwargs).to(device)
    return PCLFeatureExtractor(
        patch_sampler=sampler,
        num_patches=pcl_cfg.get("num_patches", 256),
    ).to(device)


def load_model_and_optional_pcl(cfg, checkpoint: Optional[str], device):
    model = ULUNAS_PCL_Spec(**cfg["network_config"]).to(device)
    pcl_state = None
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded model from {checkpoint}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
        if isinstance(ckpt, dict) and "pcl_extractor" in ckpt:
            pcl_state = ckpt["pcl_extractor"]
            print("  found pcl_extractor weights in checkpoint")
    model.eval()
    return model, pcl_state


def verdict_table(results: Dict[str, Dict[str, float]], chance: float):
    print("\n========== PCL DEBUG VERDICT ==========")
    print(f"chance baseline ln(1+M) = {chance:.3f}")
    print(
        f"{'probe':<28}{'pcl_raw':>10}{'acc':>8}{'cos_qp':>9}{'cos_qn':>9}{'gap':>9}  interpretation"
    )

    def row(name, r, interp):
        print(
            f"{name:<28}{r.get('pcl_raw', float('nan')):10.3f}"
            f"{r.get('retrieval_acc', float('nan')):8.3f}"
            f"{r.get('cos_qp', float('nan')):9.3f}"
            f"{r.get('cos_qn', float('nan')):9.3f}"
            f"{r.get('gap_qp_qn', r.get('cos_qp', 0) - r.get('cos_qn', 0)):9.3f}"
            f"  {interp}"
        )

    p0 = results["P0_synthetic"]
    row(
        "P0 synthetic separable",
        p0,
        "PASS loss OK" if p0["pcl_raw"] < 0.5 else "FAIL PatchNCELoss/batch bug",
    )

    p1 = results["P1_oracle_point"]
    row(
        "P1 oracle (point)",
        p1,
        "PASS loss path OK" if p1["pcl_raw"] < 1.0 else "FAIL extractor/loss wiring",
    )

    for key, label in [
        ("P4_point_real", "P4 point on real feats"),
        ("P4_conv2d_real", "P4 conv2d on real feats"),
    ]:
        r = results[key]
        if r["pcl_raw"] < chance - 0.3 and r["retrieval_acc"] > 0.55:
            interp = "OK separable"
        elif r["retrieval_acc"] < 0.45 or r["gap_qp_qn"] < 0:
            interp = "FAIL inverted / not separable"
        else:
            interp = "WEAK ~chance, not learning"
        row(label, r, interp)

    print("\n-- Feature geometry (same batch) --")
    g = results["P5_geometry"]
    for k, v in g.items():
        print(f"  {k}: {v:.4f}")

    print("\n-- Decision tree --")
    if p0["pcl_raw"] >= 0.5:
        print("=> Fix PatchNCELoss / batch_size reshape first (P0 failed).")
    elif p1["pcl_raw"] >= 1.0:
        print("=> Fix PCLFeatureExtractor / id sharing (P1 failed).")
    elif (
        results["P4_point_real"]["pcl_raw"] < chance - 0.3
        and results["P4_conv2d_real"]["pcl_raw"] >= chance
    ):
        print(
            "=> Root cause localized to PatchSampleFConv2d (or its effect on "
            "separability). Data/model feats are fine under point sampler."
        )
    elif results["P4_point_real"]["gap_qp_qn"] < 0:
        print(
            "=> Even point sampler sees inverted geometry on these feats: "
            "mask/Spec features not speech-noise separable at patch level."
        )
    elif g.get("corr_noise_gt_clean", 1) < g.get("corr_masked_noise_clean", 0) - 0.2:
        print(
            "=> GT-noise negative is much less correlated with clean than "
            "masked-noisy negative: try training with negative_mode=gt_noise "
            "(or equivalent) before adding Conv/λ."
        )
    else:
        print(
            "=> Ambiguous: check collapse stats and whether trained pcl_extractor "
            "weights were loaded; compare phase2 vs failing phase3 checkpoint."
        )
    print("=======================================\n")


def apply_mask_mode(
    erb_feat: torch.Tensor,
    mask_speech: torch.Tensor,
    mask_noise: torch.Tensor,
    mask_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post-hoc mask coupling for ablation (no retrain needed).
      as_is      — keep model masks
      sum        — renormalize s+n=1
      complement — n := 1-s
    Returns feat_s, feat_n_masked, mask_s, mask_n.
    """
    if mask_mode == "as_is":
        ms, mn = mask_speech, mask_noise
    elif mask_mode == "sum":
        total = (mask_speech + mask_noise).clamp(min=1e-8)
        ms = mask_speech / total
        mn = mask_noise / total
    elif mask_mode == "complement":
        ms = mask_speech
        mn = 1.0 - mask_speech
    else:
        raise ValueError(f"Unknown mask_mode={mask_mode!r}")
    return erb_feat * ms, erb_feat * mn, ms, mn


def build_pcl_triplet_feats(
    model: ULUNAS_PCL_Spec,
    out,
    clean: torch.Tensor,
    noise_gt: torch.Tensor,
    noisy: torch.Tensor,
    mask_mode: str,
    negative_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Returns feat_q_speech, feat_negative, feat_positive, aux_corrs.

    negative_mode:
      masked_noisy — ERB(noisy) * mask_n  (current Spec-PCL)
      gt_noise     — ERB(noise_gt)        (oracle noise spectrum)
    """
    feat_gt = model.encode_erb_feat(clean)
    feat_s, feat_n_masked, _, _ = apply_mask_mode(
        out.erb_feat, out.mask_speech, out.mask_noise, mask_mode
    )
    feat_noisy = out.erb_feat
    feat_noise_gt = model.encode_erb_feat(noise_gt)

    if negative_mode == "masked_noisy":
        feat_neg = feat_n_masked
    elif negative_mode == "gt_noise":
        feat_neg = feat_noise_gt
    else:
        raise ValueError(f"Unknown negative_mode={negative_mode!r}")

    aux = {
        "corr_noisy_clean": _tf_corr(feat_noisy, feat_gt),
        "corr_noise_gt_clean": _tf_corr(feat_noise_gt, feat_gt),
        "corr_masked_noise_clean": _tf_corr(feat_n_masked, feat_gt),
    }
    return feat_s, feat_neg, feat_gt, aux


def _map_global_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of flattened (B,C,T,F) maps, averaged over batch."""
    return _mean_cos(
        F.normalize(a.flatten(1), dim=-1),
        F.normalize(b.flatten(1), dim=-1),
    )


def _map_chan_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine over channel axis at each TF location, then spatial/batch mean."""
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1).mean().item()


def _encode_levels(model: ULUNAS_PCL_Spec, erb: torch.Tensor):
    """Return (encoder_out, latent=encoder+dpgrnn)."""
    enc_feat, _ = model.encoder(erb)
    latent = model.dpgrnn(enc_feat)
    return enc_feat, latent


def run_reencode_probe(
    cfg,
    checkpoint: Optional[str],
    device,
    num_batches: int,
    batch_size: int,
    mask_mode: str,
    load_pcl_weights: bool,
):
    """
    Test: masked ERB is shallow; re-encode each branch through Encoder(/+DPGRNN)
    and check whether speech/noise geometry separates in deeper features.
    """
    model, pcl_state = load_model_and_optional_pcl(cfg, checkpoint, device)
    pcl_cfg = cfg.get("pcl", {})
    num_patches = int(pcl_cfg.get("num_patches", 256))
    temperature = float(pcl_cfg.get("temperature", 0.07))
    chance = _chance(num_patches)
    # bottleneck channels from network (default last channel = 16)
    deep_ch = int(cfg.get("network_config", {}).get("channels", [12, 24, 24, 32, 16])[-1])

    val_set = Dataset(**cfg["validation_dataset"])
    loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Dataset.collate_fn,
        drop_last=True,
    )

    # PCL heads: shallow C=1, encoder C=16, latent C=16
    ext_erb = build_extractor("point", {**pcl_cfg, "in_channels": 1}, device)
    ext_enc = PCLFeatureExtractor(
        patch_sampler=PatchSampleF(
            in_channels=deep_ch,
            embed_dim=pcl_cfg.get("embed_dim", 128),
            use_mlp=pcl_cfg.get("use_mlp", True),
        ).to(device),
        num_patches=num_patches,
    ).to(device)
    ext_deep = PCLFeatureExtractor(
        patch_sampler=PatchSampleF(
            in_channels=deep_ch,
            embed_dim=pcl_cfg.get("embed_dim", 128),
            use_mlp=pcl_cfg.get("use_mlp", True),
        ).to(device),
        num_patches=num_patches,
    ).to(device)
    if load_pcl_weights and pcl_state is not None:
        for name, ext in [("erb", ext_erb), ("enc", ext_enc), ("lat", ext_deep)]:
            cur = ext.state_dict()
            filtered = {
                k: v for k, v in pcl_state.items() if k in cur and cur[k].shape == v.shape
            }
            ext.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)} pcl keys into {name} extractor")

    loss_fn = PatchNCELoss(temperature=temperature, batch_size=batch_size).to(device)

    stats = {k: [] for k in [
        "erb_cos(clean,noise_gt)",
        "erb_cos(masked_s,masked_n)",
        "erb_cos(masked_s,clean)",
        "enc_cos(clean,noise_gt)",
        "enc_cos(masked_s,masked_n)",
        "enc_cos(masked_s,clean)",
        "enc_chan_cos(masked_s,masked_n)",
        "lat_cos(clean,noise_gt)",
        "lat_cos(masked_s,masked_n)",
        "lat_cos(masked_s,clean)",
        "lat_chan_cos(masked_s,masked_n)",
        "pcl_erb_raw",
        "pcl_erb_acc",
        "pcl_erb_gap",
        "pcl_enc_raw",
        "pcl_enc_acc",
        "pcl_enc_gap",
        "pcl_lat_raw",
        "pcl_lat_acc",
        "pcl_lat_gap",
    ]}

    print("\n========== RE-ENCODE PROBE (masked ERB → Encoder / Latent) ==========")
    print(
        f"mask_mode={mask_mode} | deep_ch={deep_ch} | "
        f"chance≈{chance:.3f} | checkpoint={'yes' if checkpoint else 'random init'}"
    )
    print(
        "Compare geometry + PCL at 3 levels: ERB → Encoder → Encoder+DPGRNN.\n"
        "Focus: pcl_enc_* (Encoder only, no DPGRNN) is the candidate for training."
    )

    with torch.inference_mode():
        for bi, (noisy, clean, noise_gt) in enumerate(loader):
            if bi >= num_batches:
                break
            noisy = noisy.to(device)
            clean = clean.to(device)
            noise_gt = noise_gt.to(device)

            out = model(noisy, return_dict=True)
            e_clean = model.encode_erb_feat(clean)
            e_noise = model.encode_erb_feat(noise_gt)
            e_ms, e_mn, _, _ = apply_mask_mode(
                out.erb_feat, out.mask_speech, out.mask_noise, mask_mode
            )

            enc_clean, lat_clean = _encode_levels(model, e_clean)
            enc_noise, lat_noise = _encode_levels(model, e_noise)
            enc_ms, lat_ms = _encode_levels(model, e_ms)
            enc_mn, lat_mn = _encode_levels(model, e_mn)

            row = {
                "erb_cos(clean,noise_gt)": _map_global_cos(e_clean, e_noise),
                "erb_cos(masked_s,masked_n)": _map_global_cos(e_ms, e_mn),
                "erb_cos(masked_s,clean)": _map_global_cos(e_ms, e_clean),
                "enc_cos(clean,noise_gt)": _map_global_cos(enc_clean, enc_noise),
                "enc_cos(masked_s,masked_n)": _map_global_cos(enc_ms, enc_mn),
                "enc_cos(masked_s,clean)": _map_global_cos(enc_ms, enc_clean),
                "enc_chan_cos(masked_s,masked_n)": _map_chan_cos(enc_ms, enc_mn),
                "lat_cos(clean,noise_gt)": _map_global_cos(lat_clean, lat_noise),
                "lat_cos(masked_s,masked_n)": _map_global_cos(lat_ms, lat_mn),
                "lat_cos(masked_s,clean)": _map_global_cos(lat_ms, lat_clean),
                "lat_chan_cos(masked_s,masked_n)": _map_chan_cos(lat_ms, lat_mn),
            }

            p_erb = probe_real_batch(
                ext_erb, loss_fn, e_ms, e_mn, e_clean, batch_size
            )
            p_enc = probe_real_batch(
                ext_enc, loss_fn, enc_ms, enc_mn, enc_clean, batch_size
            )
            p_lat = probe_real_batch(
                ext_deep, loss_fn, lat_ms, lat_mn, lat_clean, batch_size
            )
            row.update({
                "pcl_erb_raw": p_erb["pcl_raw"],
                "pcl_erb_acc": p_erb["retrieval_acc"],
                "pcl_erb_gap": p_erb["gap_qp_qn"],
                "pcl_enc_raw": p_enc["pcl_raw"],
                "pcl_enc_acc": p_enc["retrieval_acc"],
                "pcl_enc_gap": p_enc["gap_qp_qn"],
                "pcl_lat_raw": p_lat["pcl_raw"],
                "pcl_lat_acc": p_lat["retrieval_acc"],
                "pcl_lat_gap": p_lat["gap_qp_qn"],
            })

            for k, v in row.items():
                stats[k].append(v)

            print(
                f"batch {bi}: "
                f"cos(s,n) erb={row['erb_cos(masked_s,masked_n)']:.3f} "
                f"enc={row['enc_cos(masked_s,masked_n)']:.3f} "
                f"lat={row['lat_cos(masked_s,masked_n)']:.3f} | "
                f"pcl_erb={row['pcl_erb_raw']:.3f}/{row['pcl_erb_gap']:.3f} "
                f"pcl_enc={row['pcl_enc_raw']:.3f}/{row['pcl_enc_gap']:.3f} "
                f"pcl_lat={row['pcl_lat_raw']:.3f}/{row['pcl_lat_gap']:.3f}"
            )

    print("\n-- averages --")
    avg = {k: float(np.mean(v)) for k, v in stats.items()}
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    print("\n-- verdict --")
    erb_sn = avg["erb_cos(masked_s,masked_n)"]
    enc_sn = avg["enc_cos(masked_s,masked_n)"]
    lat_sn = avg["lat_cos(masked_s,masked_n)"]
    print(f"  masked speech↔noise cosine: ERB {erb_sn:.3f} → Enc {enc_sn:.3f} → Lat {lat_sn:.3f}")
    print(
        f"  PCL gap/raw: erb {avg['pcl_erb_gap']:.3f}/{avg['pcl_erb_raw']:.3f} | "
        f"enc {avg['pcl_enc_gap']:.3f}/{avg['pcl_enc_raw']:.3f} | "
        f"lat {avg['pcl_lat_gap']:.3f}/{avg['pcl_lat_raw']:.3f} (chance {chance:.3f})"
    )
    if enc_sn < erb_sn - 0.1:
        print(
            "=> Encoder geometry is better than ERB. "
            "Prefer training PCL on Encoder features (pcl_feature_mode=encoder)."
        )
    if lat_sn > enc_sn + 0.05:
        print("=> DPGRNN latent re-correlates branches; do NOT attach PCL after DPGRNN.")
    if avg["pcl_enc_gap"] > avg["pcl_erb_gap"]:
        print("=> pcl@Encoder gap >= erb; Encoder-PCL is the right next training target.")
    print("====================================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="model .tar (phase2 or phase3)")
    parser.add_argument("-D", "--device", default="0")
    parser.add_argument("--num_batches", type=int, default=8)
    parser.add_argument("--load_pcl_weights", action="store_true",
                        help="if checkpoint has pcl_extractor, load into BOTH samplers' MLPs when shapes match")
    parser.add_argument(
        "--mask_mode",
        default="as_is",
        choices=["as_is", "sum", "complement"],
        help="post-hoc mask coupling for speech/noise masks",
    )
    parser.add_argument(
        "--negative_mode",
        default="masked_noisy",
        choices=["masked_noisy", "gt_noise"],
        help="PCL negative: masked ERB(noisy) vs ERB(gt noise)",
    )
    parser.add_argument(
        "--erb_baseline",
        action="store_true",
        help="only compute ERB(clean) vs ERB(noise_gt) similarity; no PCL / no checkpoint needed",
    )
    parser.add_argument(
        "--reencode_probe",
        action="store_true",
        help="masked ERB → Encoder/Latent geometry + PCL probe (needs checkpoint)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    batch_size = int(cfg["validation_dataloader"]["batch_size"])

    if args.erb_baseline:
        run_erb_baseline(cfg, device, args.num_batches, batch_size)
        return

    if args.reencode_probe:
        if not args.checkpoint:
            raise SystemExit("--reencode_probe needs --checkpoint (phase2 or point phase3)")
        run_reencode_probe(
            cfg,
            args.checkpoint,
            device,
            args.num_batches,
            batch_size,
            args.mask_mode,
            args.load_pcl_weights,
        )
        return

    pcl_cfg = cfg.get("pcl", {})
    num_patches = int(pcl_cfg.get("num_patches", 256))
    temperature = float(pcl_cfg.get("temperature", 0.07))
    chance = _chance(num_patches)

    model, pcl_state = load_model_and_optional_pcl(cfg, args.checkpoint, device)
    print(
        f"[ablation] mask_mode={args.mask_mode} | negative_mode={args.negative_mode}"
    )

    val_set = Dataset(**cfg["validation_dataset"])
    loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=Dataset.collate_fn,
        drop_last=True,
    )

    ext_point = build_extractor("point", pcl_cfg, device)
    ext_conv = build_extractor("conv2d", pcl_cfg, device)
    if args.load_pcl_weights and pcl_state is not None:
        for name, ext in [("point", ext_point), ("conv2d", ext_conv)]:
            cur = ext.state_dict()
            filtered = {
                k: v for k, v in pcl_state.items() if k in cur and cur[k].shape == v.shape
            }
            missing = ext.load_state_dict(filtered, strict=False)
            print(f"Loaded {len(filtered)} pcl keys into {name} extractor ({missing})")

    loss_fn = PatchNCELoss(temperature=temperature, batch_size=batch_size).to(device)

    agg_point = []
    agg_conv = []
    geo = {
        "corr_pre_speech_clean": [],
        "corr_pre_noise_clean": [],
        "corr_noisy_clean": [],
        "corr_noise_gt_clean": [],
        "corr_masked_noise_clean": [],
        "corr_postconv_speech_clean": [],
        "corr_postconv_noise_clean": [],
    }

    with torch.inference_mode():
        for bi, (noisy, clean, noise_gt) in enumerate(loader):
            if bi >= args.num_batches:
                break
            noisy = noisy.to(device)
            clean = clean.to(device)
            noise_gt = noise_gt.to(device)
            out = model(noisy, return_dict=True)
            feat_s, feat_neg, feat_gt, aux = build_pcl_triplet_feats(
                model,
                out,
                clean,
                noise_gt,
                noisy,
                args.mask_mode,
                args.negative_mode,
            )

            agg_point.append(
                probe_real_batch(
                    ext_point, loss_fn, feat_s, feat_neg, feat_gt, batch_size
                )
            )
            agg_conv.append(
                probe_real_batch(
                    ext_conv, loss_fn, feat_s, feat_neg, feat_gt, batch_size
                )
            )

            geo["corr_pre_speech_clean"].append(_tf_corr(feat_s, feat_gt))
            geo["corr_pre_noise_clean"].append(_tf_corr(feat_neg, feat_gt))
            geo["corr_noisy_clean"].append(aux["corr_noisy_clean"])
            geo["corr_noise_gt_clean"].append(aux["corr_noise_gt_clean"])
            geo["corr_masked_noise_clean"].append(aux["corr_masked_noise_clean"])

            conv = ext_conv.patch_sampler.conv
            geo["corr_postconv_speech_clean"].append(
                _tf_corr(conv(feat_s), conv(feat_gt))
            )
            geo["corr_postconv_noise_clean"].append(
                _tf_corr(conv(feat_neg), conv(feat_gt))
            )

            print(
                f"batch {bi}: point_pcl={agg_point[-1]['pcl_raw']:.3f} "
                f"acc={agg_point[-1]['retrieval_acc']:.3f} gap={agg_point[-1]['gap_qp_qn']:.3f} | "
                f"corr(s,gt)={geo['corr_pre_speech_clean'][-1]:.3f} "
                f"corr(neg,gt)={geo['corr_pre_noise_clean'][-1]:.3f} "
                f"corr(noisy,gt)={aux['corr_noisy_clean']:.3f} "
                f"corr(noise_gt,gt)={aux['corr_noise_gt_clean']:.3f}"
            )

    def mean_dict(dicts):
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

    results = {
        "P0_synthetic": probe_p0_synthetic(num_patches, batch_size, device, temperature),
        "P1_oracle_point": probe_p1_oracle(
            ext_point, loss_fn, feat_gt, batch_size
        ),
        "P4_point_real": mean_dict(agg_point),
        "P4_conv2d_real": mean_dict(agg_conv),
        "P5_geometry": {k: float(np.mean(v)) for k, v in geo.items()},
    }
    verdict_table(results, chance)


if __name__ == "__main__":
    main()
