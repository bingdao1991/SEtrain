import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Union


class HybridLoss(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.window = torch.hann_window(win_len)
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        device = y_true.device
        
        pred_stft = torch.stft(y_pred, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)
        true_stft = torch.stft(y_true, self.n_fft, self.hop_len, self.win_len, self.window.to(device), return_complex=True)

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)
        
        pred_stft_c = pred_stft / pred_mag**(1 - self.c)
        true_stft_c = true_stft / true_mag**(1 - self.c)

        real_loss = torch.mean((pred_stft_c.real - true_stft_c.real)**2)
        imag_loss = torch.mean((pred_stft_c.imag - true_stft_c.imag)**2)
        mag_loss = torch.mean((pred_mag**self.c - true_mag**self.c)**2)

        # SISNR loss
        y_norm = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr = - 2*torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True) / 
            torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(self.eps) + 
            self.eps
        ).mean()
        
        return self.lamda_ri*(real_loss + imag_loss) + self.lamda_mag*mag_loss + sisnr


class STFTLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_len=120, win_len=600, window="hann_window"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        self.register_buffer("window", getattr(torch, window)(win_len))

    def loss_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

    def loss_log_magnitude(self, x_mag, y_mag):
        return torch.nn.functional.l1_loss(torch.log(y_mag), torch.log(x_mag))

    def forward(self, x, y):
        """x, y: (B, T), in time domain"""
        x = torch.stft(x, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        y = torch.stft(y, self.n_fft, self.hop_len, self.win_len, self.window.to(x.device), return_complex=True)
        x_mag = torch.abs(x).clamp(1e-8)
        y_mag = torch.abs(y).clamp(1e-8)
        
        sc_loss = self.loss_spectral_convergence(x_mag, y_mag)
        mag_loss = self.loss_log_magnitude(x_mag, y_mag)
        loss = sc_loss + mag_loss

        return loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x, y)
        loss /= len(self.stft_losses)
        return loss


class ReconstructionLoss(nn.Module):
    """Encourage mixture ≈ speech + noise."""

    def __init__(self, loss_type: str = "sisnr", eps: float = 1e-8):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        speech: torch.Tensor,
        noise: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        recon = speech + noise
        if self.loss_type == "l1":
            return torch.mean(torch.abs(recon - mixture))
        if self.loss_type == "mse":
            return torch.mean((recon - mixture) ** 2)

        y_true = mixture
        y_pred = recon
        y_norm = (
            torch.sum(y_true * y_pred, dim=-1, keepdim=True)
            * y_true
            / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + self.eps)
        )
        sisnr = -2 * torch.log10(
            torch.norm(y_norm, dim=-1, keepdim=True)
            / torch.norm(y_pred - y_norm, dim=-1, keepdim=True).clamp(self.eps)
            + self.eps
        ).mean()
        return sisnr


class MaskSumLoss(nn.Module):
    def forward(self, mask_speech: torch.Tensor, mask_noise: torch.Tensor) -> torch.Tensor:
        target = torch.ones_like(mask_speech)
        return torch.mean((mask_speech + mask_noise - target) ** 2)


@dataclass
class DualOutputLossOutput:
    total: torch.Tensor
    speech: torch.Tensor
    noise: torch.Tensor
    recon: torch.Tensor
    mask_sum: torch.Tensor
    pcl: Optional[torch.Tensor] = None
    details: Optional[Dict[str, float]] = None


class DualOutputLoss(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        compress_factor=0.3,
        eps=1e-12,
        lamda_ri=30,
        lamda_mag=70,
        lambda_speech: float = 1.0,
        lambda_noise: float = 0.5,
        lambda_recon: float = 0.3,
        lambda_mask_sum: float = 0.0,
        lambda_pcl: float = 0.0,
        recon_loss_type: str = "sisnr",
        pcl_loss=None,
    ):
        super().__init__()
        self.hybrid_loss = HybridLoss(
            n_fft=n_fft,
            hop_len=hop_len,
            win_len=win_len,
            compress_factor=compress_factor,
            eps=eps,
            lamda_ri=lamda_ri,
            lamda_mag=lamda_mag,
        )
        self.recon_loss = ReconstructionLoss(loss_type=recon_loss_type, eps=eps)
        self.mask_sum_loss = MaskSumLoss()
        self.pcl_loss = pcl_loss
        self.lambda_speech = lambda_speech
        self.lambda_noise = lambda_noise
        self.lambda_recon = lambda_recon
        self.lambda_mask_sum = lambda_mask_sum
        self.lambda_pcl = lambda_pcl

    def set_lambda_pcl(self, value: float):
        self.lambda_pcl = value

    def forward(
        self,
        pred_speech: torch.Tensor,
        pred_noise: torch.Tensor,
        gt_speech: torch.Tensor,
        gt_noise: torch.Tensor,
        mixture: torch.Tensor,
        mask_speech: Optional[torch.Tensor] = None,
        mask_noise: Optional[torch.Tensor] = None,
        pcl_features=None,
        batch_size: Optional[int] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, DualOutputLossOutput]:
        loss_speech = self.hybrid_loss(pred_speech, gt_speech)
        loss_noise = self.hybrid_loss(pred_noise, gt_noise)
        loss_recon = self.recon_loss(pred_speech, pred_noise, mixture)

        loss_mask_sum = torch.tensor(0.0, device=pred_speech.device)
        if mask_speech is not None and mask_noise is not None and self.lambda_mask_sum > 0:
            loss_mask_sum = self.mask_sum_loss(mask_speech, mask_noise)

        loss_pcl = None
        if (
            self.lambda_pcl > 0
            and pcl_features is not None
            and self.pcl_loss is not None
        ):
            loss_pcl = self.pcl_loss(
                pcl_features.f_query,
                pcl_features.f_positive,
                pcl_features.f_negative,
                batch_size=batch_size,
            ).mean()

        total = (
            self.lambda_speech * loss_speech
            + self.lambda_noise * loss_noise
            + self.lambda_recon * loss_recon
            + self.lambda_mask_sum * loss_mask_sum
        )
        if loss_pcl is not None:
            total = total + self.lambda_pcl * loss_pcl

        if not return_details:
            return total

        details = {
            "speech": loss_speech.item(),
            "noise": loss_noise.item(),
            "recon": loss_recon.item(),
            "mask_sum": loss_mask_sum.item(),
            "pcl": loss_pcl.item() if loss_pcl is not None else 0.0,
            "lambda_pcl": float(self.lambda_pcl),
            "pcl_weighted": (
                float(self.lambda_pcl) * loss_pcl.item()
                if loss_pcl is not None
                else 0.0
            ),
            "total": total.item(),
        }
        return DualOutputLossOutput(
            total=total,
            speech=loss_speech,
            noise=loss_noise,
            recon=loss_recon,
            mask_sum=loss_mask_sum,
            pcl=loss_pcl,
            details=details,
        )


class LossWeightScheduler:
    """Warmup scheduler for PCL loss weight."""

    def __init__(
        self,
        lambda_speech: float = 1.0,
        lambda_noise: float = 0.5,
        lambda_recon: float = 0.3,
        lambda_mask_sum: float = 0.0,
        lambda_pcl_target: float = 2.0,
        pcl_warmup_steps: int = 10000,
        training_phase: str = "phase2",
    ):
        self.lambda_speech = lambda_speech
        self.lambda_noise = lambda_noise
        self.lambda_recon = lambda_recon
        self.lambda_mask_sum = lambda_mask_sum
        self.lambda_pcl_target = lambda_pcl_target
        self.pcl_warmup_steps = pcl_warmup_steps
        self.training_phase = training_phase

    def get_weights(self, global_step: int) -> Dict[str, float]:
        if self.training_phase == "phase2":
            lambda_pcl = 0.0
        elif self.pcl_warmup_steps <= 0:
            lambda_pcl = self.lambda_pcl_target
        else:
            ratio = min(1.0, global_step / self.pcl_warmup_steps)
            lambda_pcl = self.lambda_pcl_target * ratio

        return {
            "lambda_speech": self.lambda_speech,
            "lambda_noise": self.lambda_noise,
            "lambda_recon": self.lambda_recon,
            "lambda_mask_sum": self.lambda_mask_sum,
            "lambda_pcl": lambda_pcl,
        }


if __name__=='__main__':
    a = torch.randn(2, 10000)
    b = torch.randn(2, 10000)

    loss_func = HybridLoss()
    loss = loss_func(a, b)
    print(loss)