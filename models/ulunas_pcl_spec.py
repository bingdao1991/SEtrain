# Copyright (c) 2026 Xiaobin-Rong
# Licensed under the MIT license.
"""
ULUNAS_PCL_Spec: dual speech/noise masks + PCL on masked ERB spectrograms.

Unlike ULUNAS_PCL (latent * interpolated mask), this variant applies masks on
the ERB log-magnitude map (same 129-bin space as the decoder masks), then runs
PatchNCE on those spectrogram features. No second encoder, no latent interpolate.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ulunas import DPGRNN, ERB, Encoder
from models.ulunas_pcl import DualHeadDecoder


@dataclass
class ULUNASPCLSpecForwardOutput:
    speech: torch.Tensor
    noise: torch.Tensor
    mask_speech: torch.Tensor
    mask_noise: torch.Tensor
    erb_feat: torch.Tensor
    feat_speech: torch.Tensor
    feat_noise: torch.Tensor
    pcl_feature_mode: str = "erb"


class ULUNAS_PCL_Spec(nn.Module):
    """
    Dual-output UL-UNAS with spectrogram-domain PCL features.

    Inference: speech, noise = model(noisy)
    Training:  out = model(noisy, return_dict=True)
               pcl_feature_mode=erb     → feat = erb * mask
               pcl_feature_mode=encoder → feat = Encoder(erb * mask)  # no DPGRNN
               positive = model.encode_pcl_positive(clean, mode)
    """

    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        erb_low=65,
        erb_high=64,
        types=None,
        strides=None,
        groups=None,
        channels=None,
        kernels=None,
        widths=None,
        mask_activation: str = "sigmoid",
        enforce_mask_sum: bool = False,
        mask_couple: str = "none",  # none | sum | complement
        pcl_feature_mode: str = "erb",  # erb | encoder
    ):
        super().__init__()
        if types is None:
            types = [0, 2, 1, 2, 1]
        if strides is None:
            strides = [2, 2, 1, 1, 1]
        if groups is None:
            groups = [1, 2, 2, 2, 2]
        if channels is None:
            channels = [12, 24, 24, 32, 16]
        if kernels is None:
            kernels = [(3, 3), (2, 3), (2, 3), (1, 5), (1, 5)]
        if widths is None:
            widths = [65, 33, 33, 33, 33]

        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.mask_activation = mask_activation
        self.enforce_mask_sum = enforce_mask_sum
        self.mask_couple = mask_couple
        self.pcl_feature_mode = pcl_feature_mode
        self.encoder_out_channels = channels[-1]
        if self.mask_couple not in ("none", "sum", "complement"):
            raise ValueError(
                f"mask_couple must be none|sum|complement, got {mask_couple!r}"
            )
        if self.pcl_feature_mode not in ("erb", "encoder"):
            raise ValueError(
                f"pcl_feature_mode must be erb|encoder, got {pcl_feature_mode!r}"
            )

        self.erb = ERB(erb_low, erb_high, nfft=n_fft, high_lim=8000, fs=16000)
        self.encoder = Encoder(types, channels, widths, kernels, strides, groups)
        self.dpgrnn = nn.Sequential(
            *[DPGRNN(channels[-1], widths[-1], channels[-1]) for _ in range(2)]
        )
        self.decoder = DualHeadDecoder(
            types,
            channels,
            widths,
            kernels,
            strides,
            groups,
            final_width=erb_low + erb_high,
            mask_activation=mask_activation,
        )

    def _get_stft_kwargs(self, device):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": torch.hann_window(self.win_len).to(device),
            "onesided": True,
        }

    def _stft(self, waveform: torch.Tensor, device) -> Tuple[torch.Tensor, dict]:
        stft_kwargs = self._get_stft_kwargs(device)
        spec = torch.stft(waveform, **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = spec.permute(0, 3, 2, 1)
        return spec, stft_kwargs

    def _spec_to_erb_feat(self, spec: torch.Tensor) -> torch.Tensor:
        feat = torch.log10(torch.norm(spec, dim=1, keepdim=True).clamp(1e-12))
        return self.erb.bm(feat)

    def _encode_from_erb(
        self, erb_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        enc_feat, en_outs = self.encoder(erb_feat)
        latent = self.dpgrnn(enc_feat)
        return latent, en_outs

    def _apply_mask_activation_split(
        self, masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        masks: decoder output after sigmoid/softmax, shape (B, 2, T, F).

        mask_couple:
          none       — use two heads as-is
          sum        — renormalize so mask_s + mask_n = 1
          complement — mask_n := 1 - mask_s (ignore noise head)
        """
        mask_speech = masks[:, 0:1]
        mask_noise = masks[:, 1:2]
        couple = getattr(self, "mask_couple", "none")
        if couple == "complement":
            mask_noise = 1.0 - mask_speech
        elif couple == "sum" or (
            self.enforce_mask_sum and self.mask_activation != "softmax"
        ):
            total = (mask_speech + mask_noise).clamp(min=1e-8)
            mask_speech = mask_speech / total
            mask_noise = mask_noise / total
        return mask_speech, mask_noise

    def _mask_to_waveform(
        self,
        spec: torch.Tensor,
        mask_erb: torch.Tensor,
        stft_kwargs: dict,
        n_samples: int,
    ) -> torch.Tensor:
        m = self.erb.bs(mask_erb)
        spec_masked = spec * m
        spec_masked = spec_masked.permute(0, 3, 2, 1)
        spec_complex = torch.complex(spec_masked[..., 0], spec_masked[..., 1])
        output = torch.istft(spec_complex, **stft_kwargs)
        return F.pad(output, (0, n_samples - output.shape[1]))

    def encode_erb_feat(self, waveform: torch.Tensor) -> torch.Tensor:
        """ERB log-magnitude map. Shape (B,1,T,129)."""
        device = waveform.device
        spec, _ = self._stft(waveform, device)
        return self._spec_to_erb_feat(spec)

    def encode_pcl_positive(
        self, waveform: torch.Tensor, mode: Optional[str] = None
    ) -> torch.Tensor:
        """
        Positive branch for PCL.
          erb     → ERB(clean)
          encoder → Encoder(ERB(clean))   # no DPGRNN
        """
        mode = mode or self.pcl_feature_mode
        erb = self.encode_erb_feat(waveform)
        if mode == "erb":
            return erb
        if mode == "encoder":
            enc, _ = self.encoder(erb)
            return enc
        raise ValueError(f"Unknown pcl mode {mode!r}")

    def _pcl_feats_from_masked_erb(
        self, erb_masked_speech: torch.Tensor, erb_masked_noise: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pcl_feature_mode == "erb":
            return erb_masked_speech, erb_masked_noise
        # encoder: re-encode each masked ERB branch (NO dpgrnn)
        feat_s, _ = self.encoder(erb_masked_speech)
        feat_n, _ = self.encoder(erb_masked_noise)
        return feat_s, feat_n

    def forward(
        self,
        input: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], ULUNASPCLSpecForwardOutput]:
        device = input.device
        assert input.ndim == 2
        n_samples = input.shape[1]

        spec, stft_kwargs = self._stft(input, device)
        erb_feat = self._spec_to_erb_feat(spec)
        latent, en_outs = self._encode_from_erb(erb_feat)

        masks = self.decoder(latent, en_outs)
        mask_speech, mask_noise = self._apply_mask_activation_split(masks)

        speech = self._mask_to_waveform(spec, mask_speech, stft_kwargs, n_samples)
        noise = self._mask_to_waveform(spec, mask_noise, stft_kwargs, n_samples)

        if not return_dict:
            return speech, noise

        erb_speech = erb_feat * mask_speech
        erb_noise = erb_feat * mask_noise
        feat_speech, feat_noise = self._pcl_feats_from_masked_erb(erb_speech, erb_noise)

        return ULUNASPCLSpecForwardOutput(
            speech=speech,
            noise=noise,
            mask_speech=mask_speech,
            mask_noise=mask_noise,
            erb_feat=erb_feat,
            feat_speech=feat_speech,
            feat_noise=feat_noise,
            pcl_feature_mode=self.pcl_feature_mode,
        )

    @classmethod
    def from_ulunas_checkpoint(
        cls,
        ulunas_state_dict: dict,
        init_noise_mask: str = "complement",
        **model_kwargs,
    ) -> "ULUNAS_PCL_Spec":
        model = cls(**model_kwargs)
        new_state = model.state_dict()
        loaded = {}

        for key, value in ulunas_state_dict.items():
            if key not in new_state:
                continue
            if new_state[key].shape == value.shape:
                loaded[key] = value
            elif (
                "decoder.de_convs" in key
                and key.endswith(".ops.1.weight")
                and value.shape[0] == 1
                and new_state[key].shape[0] == 2
            ):
                loaded[key] = new_state[key].clone()
                loaded[key][0:1] = value
                if init_noise_mask == "complement":
                    loaded[key][1:2] = value
                elif init_noise_mask == "zero":
                    loaded[key][1:2] = 0.0
            elif (
                "decoder.de_convs" in key
                and key.endswith(".ops.1.bias")
                and value is not None
                and new_state[key].shape[0] == 2
                and value.shape[0] == 1
            ):
                loaded[key] = new_state[key].clone()
                loaded[key][0:1] = value
                if init_noise_mask == "complement":
                    loaded[key][1:2] = -value
                elif init_noise_mask == "zero":
                    loaded[key][1:2] = 0.0

        model.load_state_dict(loaded, strict=False)
        return model


if __name__ == "__main__":
    model = ULUNAS_PCL_Spec().eval()
    x = torch.randn(2, 16000)
    speech, noise = model(x)
    print(f"speech: {speech.shape}, noise: {noise.shape}")

    out = model(x, return_dict=True)
    print(
        f"erb: {out.erb_feat.shape}, mask: {out.mask_speech.shape}, "
        f"feat_s: {out.feat_speech.shape}, mode={out.pcl_feature_mode}"
    )
    clean_feat = model.encode_pcl_positive(x)
    print(f"positive: {clean_feat.shape}")

    model_enc = ULUNAS_PCL_Spec(pcl_feature_mode="encoder").eval()
    out_e = model_enc(x, return_dict=True)
    pos_e = model_enc.encode_pcl_positive(x)
    print(f"encoder feat_s: {out_e.feat_speech.shape}, positive: {pos_e.shape}")
    assert out_e.feat_speech.shape[1] == model_enc.encoder_out_channels
    assert out_e.feat_speech.shape == pos_e.shape
