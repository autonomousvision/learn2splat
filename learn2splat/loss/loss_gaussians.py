from dataclasses import dataclass
from math import isqrt

import torch
from jaxtyping import Float
from torch import Tensor

from learn2splat.loss import Loss
from learn2splat.model.decoder.decoder import DecoderOutput
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossGaussiansCfg:
    weight: float | int
    weight_scales: float
    weight_opacities: float
    weight_sh: float
    sh_alpha: float


@dataclass
class LossGaussiansCfgWrapper:
    gaussians: LossGaussiansCfg


class LossGaussians(Loss[LossGaussiansCfg, LossGaussiansCfgWrapper]):
    """L2 regularization on Gaussian scales, opacities, and SH coefficients.

    Each component has an independent weight so they can be tuned separately.
    SH degree 0 (DC / base color) is always excluded.

    sh_alpha controls per-degree weighting for the SH term:
        alpha=1.0 (default): uniform across all degrees >= 1
        alpha>1.0: exponentially increasing penalty on higher degrees
                   (degree d gets alpha^d weighting)
    """

    def forward(
            self,
            prediction: DecoderOutput,
            gaussians: Gaussians | GaussiansModule | None,
            global_step: int,
            gt_rgb: Tensor,
            pred_rgb: Tensor,
            valid_depth_mask: Tensor | None,
            **kwargs,
    ) -> Float[Tensor, ""]:
        if gaussians is None:
            raise ValueError("Gaussians must be provided for LossGaussians.")

        loss = 0
        nr_valid = gaussians.nr_valid

        if self.cfg.weight_scales > 0:
            loss = loss + self.cfg.weight_scales * (gaussians.scales[:, :nr_valid] ** 2).mean()  # [B, G, 3]

        if self.cfg.weight_opacities > 0:
            loss = loss + self.cfg.weight_opacities * (gaussians.opacities[:, :nr_valid] ** 2).mean()  # [B, G]

        if self.cfg.weight_sh > 0:
            shs = gaussians.harmonics  # [B, G, 3, d_sh]
            d_sh = shs.shape[-1]
            if d_sh > 1:
                alpha = self.cfg.sh_alpha
                if alpha == 1.0:
                    shN = shs[:, :nr_valid, :, 1:]
                    loss = loss + self.cfg.weight_sh * (shN ** 2).mean()
                else:
                    max_degree = isqrt(d_sh) - 1
                    sh_loss = torch.tensor(0.0, device=shs.device, dtype=shs.dtype)
                    for degree in range(1, max_degree + 1):
                        start = degree ** 2
                        end = (degree + 1) ** 2
                        sh_band = shs[:, :nr_valid, :, start:end]
                        sh_loss = sh_loss + (alpha ** degree) * (sh_band ** 2).mean()
                    loss = loss + self.cfg.weight_sh * sh_loss

        return loss * self.cfg.weight
