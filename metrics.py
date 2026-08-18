import torch


def compute_si_snr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Returns (B,) SI-SNR in dB."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(estimate * target, dim=-1, keepdim=True) * target / target_energy
    noise = estimate - proj
    ratio = (torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def _proj_energy_ratio_db(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Energy ratio (dB) of the component of `estimate` that lies along `reference`.
    Lower means less leakage of `reference` into `estimate`.
    Returns shape (B,).
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    proj = torch.sum(estimate * reference, dim=-1, keepdim=True) * reference / ref_energy
    est_energy = torch.sum(estimate ** 2, dim=-1) + eps
    proj_energy = torch.sum(proj ** 2, dim=-1) + eps
    return 10 * torch.log10(proj_energy / est_energy)


def evaluate_dual_output_batch(
    pred_speech: torch.Tensor,
    pred_noise: torch.Tensor,
    gt_speech: torch.Tensor,
    gt_noise: torch.Tensor,
    mixture: torch.Tensor,
) -> dict:
    speech_si_snr = compute_si_snr(pred_speech, gt_speech).mean().item()
    noise_si_snr = compute_si_snr(pred_noise, gt_noise).mean().item()
    recon_si_snr = compute_si_snr(pred_speech + pred_noise, mixture).mean().item()
    return {
        "speech_si_snr": speech_si_snr,
        "noise_si_snr": noise_si_snr,
        "recon_si_snr": recon_si_snr,
    }


def evaluate_dual_leak_batch(
    pred_speech: torch.Tensor,
    pred_noise: torch.Tensor,
    gt_speech: torch.Tensor,
    gt_noise: torch.Tensor,
    mixture: torch.Tensor,
) -> dict:
    """
    Three headline metrics for ANO vs ANO+PCL comparison.

    1) speech_si_snri: SI-SNR improvement of speech track over mixture (higher better)
    2) noise_speech_leak_db: speech energy ratio inside pred_noise (lower better)
    3) speech_noise_leak_db: noise energy ratio inside pred_speech (lower better)
    """
    speech_si_snr = compute_si_snr(pred_speech, gt_speech)
    mixture_si_snr = compute_si_snr(mixture, gt_speech)
    speech_si_snri = (speech_si_snr - mixture_si_snr).mean().item()

    noise_speech_leak_db = _proj_energy_ratio_db(pred_noise, gt_speech).mean().item()
    speech_noise_leak_db = _proj_energy_ratio_db(pred_speech, gt_noise).mean().item()

    return {
        "speech_si_snri": speech_si_snri,
        "noise_speech_leak_db": noise_speech_leak_db,
        "speech_noise_leak_db": speech_noise_leak_db,
        # extras for debugging
        "speech_si_snr": speech_si_snr.mean().item(),
        "noise_si_snr": compute_si_snr(pred_noise, gt_noise).mean().item(),
        "recon_si_snr": compute_si_snr(pred_speech + pred_noise, mixture).mean().item(),
    }
