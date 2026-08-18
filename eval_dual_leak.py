"""
Compare ANO-only vs ANO+PCL on three metrics:
  (1) speech_si_snri
  (2) noise_speech_leak_db   (speech leaking into noise track; lower better)
  (3) speech_noise_leak_db   (noise leaking into speech track; lower better)

Usage A — already inferred wav folders:
  python eval_dual_leak.py --mode wav \\
    --tag_a ano --speech_dir_a <ano_speech_dir> --noise_dir_a <ano_noise_dir> \\
    --tag_b pcl --speech_dir_b <pcl_speech_dir> --noise_dir_b <pcl_noise_dir> \\
    --noisy_dir <val_noisy> --clean_dir <val_clean>

Usage B — from infer yaml configs (re-run model, then score):
  python eval_dual_leak.py --mode ckpt \\
    --cfg_a configs/cfg_infer_pcl_val.yaml \\
    --cfg_b configs/cfg_infer_pcl_spec_val.yaml \\
    --tag_a ano --tag_b pcl --device 0 --max_files 200

Usage C — score one model only:
  python eval_dual_leak.py --mode wav \\
    --tag_a ano --speech_dir_a <speech> --noise_dir_a <noise> \\
    --noisy_dir <val_noisy> --clean_dir <val_clean>
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from metrics import evaluate_dual_leak_batch


def _match_len(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min(a.shape[-1] for a in arrays)
    return tuple(a[..., :n] for a in arrays)


def _resolve_pair(
    speech_dir: str,
    noise_dir: str,
    uid: str,
) -> Optional[Tuple[str, str]]:
    candidates = [
        (os.path.join(speech_dir, f"{uid}_speech.wav"),
         os.path.join(noise_dir, f"{uid}_noise.wav")),
        (os.path.join(speech_dir, f"{uid}.wav"),
         os.path.join(noise_dir, f"{uid}.wav")),
        (os.path.join(speech_dir, uid + ".wav"),
         os.path.join(noise_dir, uid.replace("noisy", "noise") + ".wav")),
    ]
    for sp, np_ in candidates:
        if os.path.isfile(sp) and os.path.isfile(np_):
            return sp, np_
    return None


def score_wav_dirs(
    speech_dir: str,
    noise_dir: str,
    noisy_dir: str,
    clean_dir: str,
    max_files: Optional[int] = None,
) -> Tuple[Dict[str, float], List[dict]]:
    noisy_wavs = sorted([x for x in os.listdir(noisy_dir) if x.endswith(".wav")])
    if max_files is not None:
        noisy_wavs = noisy_wavs[:max_files]

    sums = {
        "speech_si_snri": 0.0,
        "noise_speech_leak_db": 0.0,
        "speech_noise_leak_db": 0.0,
        "speech_si_snr": 0.0,
        "noise_si_snr": 0.0,
        "recon_si_snr": 0.0,
    }
    rows = []
    n = 0

    for wav_name in tqdm(noisy_wavs, desc=f"score:{os.path.basename(speech_dir)}"):
        uid = wav_name.replace(".wav", "")
        pair = _resolve_pair(speech_dir, noise_dir, uid)
        if pair is None:
            continue

        noisy_path = os.path.join(noisy_dir, wav_name)
        clean_path = os.path.join(clean_dir, wav_name)
        if not os.path.isfile(clean_path):
            clean_path = noisy_path.replace("noisy", "clean")
        if not os.path.isfile(clean_path):
            continue

        pred_speech, _ = sf.read(pair[0], dtype="float32")
        pred_noise, _ = sf.read(pair[1], dtype="float32")
        mixture, _ = sf.read(noisy_path, dtype="float32")
        gt_speech, _ = sf.read(clean_path, dtype="float32")

        if pred_speech.ndim > 1:
            pred_speech = pred_speech.mean(axis=-1)
        if pred_noise.ndim > 1:
            pred_noise = pred_noise.mean(axis=-1)
        if mixture.ndim > 1:
            mixture = mixture.mean(axis=-1)
        if gt_speech.ndim > 1:
            gt_speech = gt_speech.mean(axis=-1)

        pred_speech, pred_noise, mixture, gt_speech = _match_len(
            pred_speech, pred_noise, mixture, gt_speech
        )
        gt_noise = mixture - gt_speech

        metrics = evaluate_dual_leak_batch(
            torch.from_numpy(pred_speech).unsqueeze(0),
            torch.from_numpy(pred_noise).unsqueeze(0),
            torch.from_numpy(gt_speech).unsqueeze(0),
            torch.from_numpy(gt_noise).unsqueeze(0),
            torch.from_numpy(mixture).unsqueeze(0),
        )
        for k in sums:
            sums[k] += metrics[k]
        row = {"uid": uid, **metrics}
        rows.append(row)
        n += 1

    if n == 0:
        raise RuntimeError(
            f"No matched files under speech={speech_dir}, noise={noise_dir}, "
            f"noisy={noisy_dir}, clean={clean_dir}"
        )
    avg = {k: v / n for k, v in sums.items()}
    avg["num_files"] = float(n)
    return avg, rows


def _load_model_from_cfg(cfg_infer, device: torch.device):
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    network_config = cfg_network.get("network_config", cfg_network)
    train_cfg_path = str(cfg_infer.network.config)

    if "spec" in train_cfg_path.replace("\\", "/").lower() or "spec" in str(
        cfg_infer.network.get("model_type", "")
    ).lower():
        from models.ulunas_pcl_spec import ULUNAS_PCL_Spec as ModelCls
    else:
        from models.ulunas_pcl import ULUNAS_PCL as ModelCls

    model = ModelCls(**network_config).to(device)
    checkpoint = torch.load(cfg_infer.network.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def infer_and_score_ckpt(
    cfg_path: str,
    device_id: str,
    max_files: Optional[int] = None,
    write_wavs: bool = True,
) -> Tuple[Dict[str, float], List[dict]]:
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device_id
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_cfg(cfg, device)

    noisy_dir = cfg.test_dataset.noisy_dir
    clean_dir = cfg.test_dataset.clean_dir
    speech_out_dir = cfg.network.speech_out_dir
    noise_out_dir = cfg.network.noise_out_dir
    if write_wavs:
        os.makedirs(speech_out_dir, exist_ok=True)
        os.makedirs(noise_out_dir, exist_ok=True)

    noisy_wavs = sorted([x for x in os.listdir(noisy_dir) if x.endswith(".wav")])
    if max_files is not None:
        noisy_wavs = noisy_wavs[:max_files]

    sums = {
        "speech_si_snri": 0.0,
        "noise_speech_leak_db": 0.0,
        "speech_noise_leak_db": 0.0,
        "speech_si_snr": 0.0,
        "noise_si_snr": 0.0,
        "recon_si_snr": 0.0,
    }
    rows = []
    n = 0

    for wav_name in tqdm(noisy_wavs, desc=f"ckpt:{os.path.basename(cfg_path)}"):
        noisy_path = os.path.join(noisy_dir, wav_name)
        clean_path = os.path.join(clean_dir, wav_name)
        if not os.path.isfile(clean_path):
            continue

        noisy, fs = sf.read(noisy_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=-1)
        if clean.ndim > 1:
            clean = clean.mean(axis=-1)

        with torch.inference_mode():
            speech, noise = model(torch.from_numpy(noisy).unsqueeze(0).to(device))
        pred_speech = speech.squeeze(0).cpu().numpy()
        pred_noise = noise.squeeze(0).cpu().numpy()

        pred_speech, pred_noise, mixture, gt_speech = _match_len(
            pred_speech, pred_noise, noisy, clean
        )
        gt_noise = mixture - gt_speech

        if write_wavs:
            uid = wav_name.replace(".wav", "")
            sf.write(os.path.join(speech_out_dir, uid + "_speech.wav"), pred_speech, fs)
            sf.write(os.path.join(noise_out_dir, uid + "_noise.wav"), pred_noise, fs)

        metrics = evaluate_dual_leak_batch(
            torch.from_numpy(pred_speech).unsqueeze(0),
            torch.from_numpy(pred_noise).unsqueeze(0),
            torch.from_numpy(gt_speech).unsqueeze(0),
            torch.from_numpy(gt_noise).unsqueeze(0),
            torch.from_numpy(mixture).unsqueeze(0),
        )
        for k in sums:
            sums[k] += metrics[k]
        rows.append({"uid": wav_name.replace(".wav", ""), **metrics})
        n += 1

    if n == 0:
        raise RuntimeError(f"No files scored for config {cfg_path}")
    avg = {k: v / n for k, v in sums.items()}
    avg["num_files"] = float(n)
    return avg, rows


def _print_table(results: Dict[str, Dict[str, float]]):
    keys = ["speech_si_snri", "noise_speech_leak_db", "speech_noise_leak_db"]
    labels = {
        "speech_si_snri": "(1) speech SI-SNRi ↑",
        "noise_speech_leak_db": "(2) noise-track speech-leak ↓",
        "speech_noise_leak_db": "(3) speech-track noise-leak ↓",
    }
    tags = list(results.keys())
    print("\n========== Dual-output leak comparison ==========")
    header = f"{'metric':<36}" + "".join(f"{t:>14}" for t in tags)
    if len(tags) == 2:
        header += f"{'delta(B-A)':>14}"
    print(header)
    for k in keys:
        line = f"{labels[k]:<36}"
        vals = []
        for t in tags:
            v = results[t][k]
            vals.append(v)
            line += f"{v:14.3f}"
        if len(vals) == 2:
            line += f"{vals[1] - vals[0]:14.3f}"
        print(line)
    print(f"\nnum_files: " + ", ".join(f"{t}={int(results[t]['num_files'])}" for t in tags))
    print("=================================================\n")


def _save_csv(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wav", "ckpt"], required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--out_csv", default="eval_dual_leak_summary.csv")

    parser.add_argument("--tag_a", default="ano")
    parser.add_argument("--tag_b", default=None)

    # wav mode
    parser.add_argument("--speech_dir_a", default=None)
    parser.add_argument("--noise_dir_a", default=None)
    parser.add_argument("--speech_dir_b", default=None)
    parser.add_argument("--noise_dir_b", default=None)
    parser.add_argument("--noisy_dir", default=None)
    parser.add_argument("--clean_dir", default=None)

    # ckpt mode
    parser.add_argument("--cfg_a", default=None)
    parser.add_argument("--cfg_b", default=None)
    parser.add_argument("--no_write_wavs", action="store_true")

    args = parser.parse_args()
    results = {}
    all_rows = []

    if args.mode == "wav":
        if not args.speech_dir_a or not args.noise_dir_a or not args.noisy_dir or not args.clean_dir:
            raise SystemExit(
                "wav mode requires --speech_dir_a --noise_dir_a --noisy_dir --clean_dir"
            )
        avg_a, rows_a = score_wav_dirs(
            args.speech_dir_a,
            args.noise_dir_a,
            args.noisy_dir,
            args.clean_dir,
            args.max_files,
        )
        results[args.tag_a] = avg_a
        for r in rows_a:
            r["tag"] = args.tag_a
            all_rows.append(r)

        if args.tag_b and args.speech_dir_b and args.noise_dir_b:
            avg_b, rows_b = score_wav_dirs(
                args.speech_dir_b,
                args.noise_dir_b,
                args.noisy_dir,
                args.clean_dir,
                args.max_files,
            )
            results[args.tag_b] = avg_b
            for r in rows_b:
                r["tag"] = args.tag_b
                all_rows.append(r)
    else:
        if not args.cfg_a:
            raise SystemExit("ckpt mode requires --cfg_a")
        avg_a, rows_a = infer_and_score_ckpt(
            args.cfg_a,
            args.device,
            args.max_files,
            write_wavs=not args.no_write_wavs,
        )
        results[args.tag_a] = avg_a
        for r in rows_a:
            r["tag"] = args.tag_a
            all_rows.append(r)

        if args.tag_b and args.cfg_b:
            avg_b, rows_b = infer_and_score_ckpt(
                args.cfg_b,
                args.device,
                args.max_files,
                write_wavs=not args.no_write_wavs,
            )
            results[args.tag_b] = avg_b
            for r in rows_b:
                r["tag"] = args.tag_b
                all_rows.append(r)

    _print_table(results)
    _save_csv(args.out_csv, all_rows)
    print(f"per-utt csv saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
