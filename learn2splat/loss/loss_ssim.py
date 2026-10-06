#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from .loss import Loss
from ..model.decoder.decoder import DecoderOutput
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from ..model.types import Gaussians

from fused_ssim import fused_ssim
from einops import rearrange

@dataclass
class LossSsimCfg:
    weight: float


@dataclass
class LossSsimCfgWrapper:
    ssim: LossSsimCfg


class LossSsim(Loss[LossSsimCfg, LossSsimCfgWrapper]):
    """Structural-similarity reconstruction loss (1 - SSIM) between the rendered and ground-truth views."""

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
        # same calculation as gsplat
        # https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py#L684
        # predicted_image, gt_image: [BS, CH, H, W]
        # predicted_image is differentiable
        pred = rearrange(pred_rgb, "b v c h w -> (b v) c h w")
        gt = rearrange(gt_rgb, "b v c h w -> (b v) c h w")
        ssim_value = 1 - fused_ssim(pred, gt, padding="valid")

        return self.cfg.weight * ssim_value
