# Copyright (c) 2026 Xiaobin-Rong
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PCLFeatures:
    f_query: torch.Tensor
    f_positive: torch.Tensor
    f_negative: torch.Tensor


class Normalize(nn.Module):
    def __init__(self, power: float = 2.0):
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm + 1e-7)


class PatchSampleF(nn.Module):
    """
    Point sampler (no spatial conv): pick random (t, f) locations from (B, C, T, F),
    then project with MLP. Used by the original Spec-PCL baseline.
    """

    def __init__(
        self,
        in_channels: int = 16,
        embed_dim: int = 128,
        use_mlp: bool = True,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_mlp = use_mlp
        self.l2norm = Normalize(2)

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, embed_dim),
            )
        else:
            self.mlp = nn.Identity()
            embed_dim = in_channels
        self.out_dim = embed_dim if use_mlp else in_channels

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return_feats = []
        return_ids = []

        for feat_id, feat in enumerate(feats):
            b, _, t, f = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(b, t * f, -1)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    n_spatial = feat_reshape.shape[1]
                    patch_id = np.random.permutation(n_spatial)
                    patch_id = patch_id[: int(min(num_patches, n_spatial))]
                    patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].reshape(-1, feat_reshape.shape[-1])
            else:
                x_sample = feat_reshape.reshape(-1, feat_reshape.shape[-1])
                patch_id = torch.tensor([], dtype=torch.long, device=feat.device)

            x_sample = self.mlp(x_sample)
            x_sample = self.l2norm(x_sample)
            return_feats.append(x_sample)
            return_ids.append(patch_id)

        return return_feats, return_ids


class PatchSampleFConv2d(nn.Module):
    """
    NASS-style patch sampler for Spec-PCL:
      two Conv2d(1→1, k=3, stride=1, padding=0) + ReLU, then random location
      sampling on the shrunk map, then MLP + L2 norm.

    Distinct from PatchSampleF (point sample, no conv).
    Intended for single-channel ERB maps (B, 1, T, F).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        use_mlp: bool = True,
        mlp_hidden: int = 256,
        conv_kernel: int = 3,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "PatchSampleFConv2d mirrors NASS Conv2d(1,1,*) and expects "
                f"in_channels=1, got {in_channels}"
            )
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_mlp = use_mlp
        self.conv_kernel = conv_kernel
        self.l2norm = Normalize(2)

        # Same structure as NASS PatchSampleF: two 3x3 convs, no padding.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=conv_kernel, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=conv_kernel, stride=1, padding=0),
        )

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(1, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, embed_dim),
            )
            self.out_dim = embed_dim
        else:
            self.mlp = nn.Identity()
            self.out_dim = 1

    def forward(
        self,
        feats: List[torch.Tensor],
        num_patches: int = 256,
        patch_ids: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return_feats = []
        return_ids = []

        for feat_id, feat in enumerate(feats):
            if feat.dim() != 4:
                raise ValueError(f"Expected (B,C,T,F), got shape {tuple(feat.shape)}")
            if feat.shape[1] != 1:
                raise ValueError(
                    f"PatchSampleFConv2d expects C=1, got C={feat.shape[1]}"
                )

            # Local aggregation on (T, F), then sample points on the shrunk grid.
            feat = self.conv(feat)
            b, _, t, f = feat.shape
            if t < 1 or f < 1:
                raise ValueError(
                    f"Feature map too small after conv: {(t, f)}; "
                    f"need larger T/F or smaller conv_kernel={self.conv_kernel}"
                )

            feat_reshape = feat.permute(0, 2, 3, 1).reshape(b, t * f, -1)

            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    n_spatial = feat_reshape.shape[1]
                    patch_id = np.random.permutation(n_spatial)
                    patch_id = patch_id[: int(min(num_patches, n_spatial))]
                    patch_id = torch.tensor(
                        patch_id, dtype=torch.long, device=feat.device
                    )
                x_sample = feat_reshape[:, patch_id, :].reshape(
                    -1, feat_reshape.shape[-1]
                )
            else:
                x_sample = feat_reshape.reshape(-1, feat_reshape.shape[-1])
                patch_id = torch.tensor([], dtype=torch.long, device=feat.device)

            x_sample = self.mlp(x_sample)
            x_sample = self.l2norm(x_sample)
            return_feats.append(x_sample)
            return_ids.append(patch_id)

        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    """InfoNCE loss: query close to positive, far from negative."""

    def __init__(self, temperature: float = 0.07, batch_size: int = 16):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_p: torch.Tensor,
        feat_n: torch.Tensor,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.batch_size
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_p = feat_p.detach()
        feat_n = feat_n.detach()

        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1),
            feat_p.view(num_patches, -1, 1),
        ).view(num_patches, 1)

        feat_q = feat_q.view(self.batch_size, -1, dim)
        feat_n = feat_n.view(self.batch_size, -1, dim)
        npatches = feat_q.size(1)

        l_neg = torch.bmm(feat_q, feat_n.transpose(2, 1))
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(diagonal, -10.0)
        l_neg = l_neg.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature
        labels = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        return self.cross_entropy_loss(out, labels)


class PCLFeatureExtractor(nn.Module):
    """Build (f_q, f_p, f_n) triplets for patch-wise contrastive learning."""

    def __init__(self, patch_sampler: PatchSampleF, num_patches: int = 256):
        super().__init__()
        self.patch_sampler = patch_sampler
        self.num_patches = num_patches

    def forward(
        self,
        latent_speech_hat: torch.Tensor,
        latent_noise_hat: torch.Tensor,
        latent_speech_gt: torch.Tensor,
    ) -> PCLFeatures:
        _, ids = self.patch_sampler([latent_speech_hat], self.num_patches)
        shared_ids = ids * 3

        spk_ef_pool, _ = self.patch_sampler(
            [latent_speech_hat, latent_noise_hat, latent_speech_gt],
            self.num_patches,
            shared_ids,
        )
        spk_tf_pool, _ = self.patch_sampler(
            [latent_speech_gt],
            self.num_patches,
            ids,
        )

        return PCLFeatures(
            f_query=spk_ef_pool[0],
            f_positive=spk_tf_pool[0],
            f_negative=spk_ef_pool[1],
        )
