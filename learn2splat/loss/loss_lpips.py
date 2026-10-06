from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from .loss import Loss
from .perceptual_loss import PerceptualLoss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    perceptual_loss: bool
    half_res: bool  # downsample to half resolution before LPIPS to save memory
    bf16: bool = True  # run the custom PerceptualLoss VGG in bf16 (faster; only when perceptual_loss)


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    """LPIPS perceptual distance between the rendered and ground-truth views.

    Returns 0 until global_step >= cfg.apply_after_step. Uses VGG-LPIPS, or the custom
    PerceptualLoss when cfg.perceptual_loss.
    """

    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        if self.cfg.perceptual_loss:
            self.lpips = PerceptualLoss(bf16=self.cfg.bf16)
        else:
            self.lpips = LPIPS(net="vgg")

        convert_to_buffer(self.lpips, persistent=False)

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

        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=pred_rgb.device)

        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5:
            pred_rgb = pred_rgb.clone()
            gt_rgb = gt_rgb.clone()
            pred_rgb[valid_depth_mask] = 0
            gt_rgb[valid_depth_mask] = 0

        pred = rearrange(pred_rgb, "b v c h w -> (b v) c h w")
        gt = rearrange(gt_rgb, "b v c h w -> (b v) c h w")

        if self.cfg.half_res:
            pred = F.interpolate(pred, scale_factor=0.5, mode="bilinear", align_corners=True)
            gt = F.interpolate(gt, scale_factor=0.5, mode="bilinear", align_corners=True)

        if self.cfg.perceptual_loss:
            loss = self.lpips(pred, gt)
        else:
            loss = self.lpips(pred, gt, normalize=True)

        return self.cfg.weight * loss.mean()
