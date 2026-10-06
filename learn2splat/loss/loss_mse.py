from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    l1_loss: bool  # use L1 instead of L2 on the RGB error
    clamp_large_error: float  # if > 0, drop per-pixel errors above this threshold


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    """RGB reconstruction loss between the rendered and ground-truth views (L2, or L1 if cfg.l1_loss)."""

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

        error = pred_rgb - gt_rgb  # [B, V, C, H, W]

        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5 and valid_depth_mask.min() < 0.5:
            error = error[~valid_depth_mask]

        if self.cfg.l1_loss:
            # l1 loss
            error = error.abs()
        else:
            # l2 loss
            error = error ** 2

        if self.cfg.clamp_large_error > 0:
            valid_mask = error < self.cfg.clamp_large_error
            error = error[valid_mask]

        error = error.mean()

        return self.cfg.weight * error
