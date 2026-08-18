# Copyright (c) 2026 Xiaobin-Rong
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ulunas import DPGRNN, ERB, Encoder, XConvBlock, XDWSBlock, XMBBlocks


@dataclass
class ULUNASPCLForwardOutput:
    speech: torch.Tensor
    noise: torch.Tensor
    mask_speech: torch.Tensor
    mask_noise: torch.Tensor
    latent: torch.Tensor
    latent_speech: torch.Tensor
    latent_noise: torch.Tensor


class DualHeadDecoder(nn.Module):
    """U-Net decoder with 2-channel output: speech mask and noise mask."""

    def __init__(
        self,
        types,
        channels,
        widths,
        kernels,
        strides,
        groups,
        final_width,
        num_outputs: int = 2,
        mask_activation: str = "sigmoid",
    ):
        super().__init__()
        self.mask_activation = mask_activation
        block_types = [XConvBlock, XDWSBlock, XMBBlocks]
        n_blocks = len(types)
        de_convs = []
        in_channels = channels[-1]

        for i in range(n_blocks - 1, 0, -1):
            module = block_types[types[i]]
            out_channels = channels[i - 1]
            de_convs.append(
                module(
                    in_channels,
                    out_channels,
                    widths[i - 1],
                    kernels[i],
                    strides[i],
                    groups[i],
                    use_deconv=True,
                )
            )
            in_channels = out_channels

        module = block_types[types[0]]
        de_convs.append(
            module(
                in_channels,
                num_outputs,
                final_width,
                kernels[0],
                strides[0],
                groups[0],
                use_deconv=True,
                is_last=True,
            )
        )
        self.de_convs = nn.ModuleList(de_convs)

    def forward(self, x, en_outs):
        n_blocks = len(self.de_convs)
        for i in range(n_blocks):
            x = self.de_convs[i](x + en_outs[n_blocks - i - 1])

        if self.mask_activation == "softmax":
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


class ULUNAS_PCL(nn.Module):
    """
    UL-UNAS with dual output (speech + noise) and PCL-compatible latent features.
    Inference: speech, noise = model(noisy)
    Training:   out = model(noisy, return_dict=True)
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

    def _encode_from_spec(self, spec: torch.Tensor) -> Tuple[torch.Tensor, list, torch.Tensor]:
        feat = self._spec_to_erb_feat(spec)
        feat, en_outs = self.encoder(feat)
        latent = self.dpgrnn(feat)
        return latent, en_outs, feat

    def _apply_mask_activation_split(self, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_speech = masks[:, 0:1]
        mask_noise = masks[:, 1:2]
        if self.enforce_mask_sum and self.mask_activation != "softmax":
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

    def encode_latent(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        spec, _ = self._stft(waveform, device)
        latent, _, _ = self._encode_from_spec(spec)
        return latent

    def forward(
        self,
        input: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], ULUNASPCLForwardOutput]:
        device = input.device
        assert input.ndim == 2
        n_samples = input.shape[1]

        spec, stft_kwargs = self._stft(input, device)
        latent, en_outs, _ = self._encode_from_spec(spec)

        masks = self.decoder(latent, en_outs)
        mask_speech, mask_noise = self._apply_mask_activation_split(masks)

        speech = self._mask_to_waveform(spec, mask_speech, stft_kwargs, n_samples)
        noise = self._mask_to_waveform(spec, mask_noise, stft_kwargs, n_samples)

        if not return_dict:
            return speech, noise

        # Decoder masks are at ERB width (129); DPGRNN latent is at bottleneck
        # width (33). Resize masks to latent spatial size for PCL (NASS-style
        # masked latent), matching NASS: sep_h = mix_w * mask in same space.
        mask_s_lat = F.interpolate(
            mask_speech, size=latent.shape[-2:], mode="nearest"
        )
        mask_n_lat = F.interpolate(
            mask_noise, size=latent.shape[-2:], mode="nearest"
        )
        latent_speech = latent * mask_s_lat
        latent_noise = latent * mask_n_lat

        return ULUNASPCLForwardOutput(
            speech=speech,
            noise=noise,
            mask_speech=mask_speech,
            mask_noise=mask_noise,
            latent=latent,
            latent_speech=latent_speech,
            latent_noise=latent_noise,
        )

    @classmethod
    def from_ulunas_checkpoint(
        cls,
        ulunas_state_dict: dict,
        init_noise_mask: str = "complement",
        **model_kwargs,
    ) -> "ULUNAS_PCL":
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
    model = ULUNAS_PCL().eval()
    x = torch.randn(2, 16000)
    speech, noise = model(x)
    print(f"speech: {speech.shape}, noise: {noise.shape}")

    out = model(x, return_dict=True)
    print(f"latent: {out.latent.shape}, mask_s: {out.mask_speech.shape}")

    recon = speech + noise
    print(f"recon err: {(recon - x).abs().mean():.4f} (untrained)")
