from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from .loss import Loss


@dataclass
class LossRenderDepthCfg:
    weight: float


@dataclass
class LossRenderDepthCfgWrapper:
    render_depth: LossRenderDepthCfg


class LossRenderDepth(Loss[LossRenderDepthCfg, LossRenderDepthCfgWrapper]):
    """Log-depth L1 between rendered depth and GT depth. Applied per optimizer step to each
    rendered view set (target and context), like the photometric losses (use_in_inner_steps).

    Assumes RAW (linear) rendered depth and takes the log itself (once per side). gsplat supplies this
    as `RGB+ED` expected depth and ignores the render mode, so it's always safe; the inria/fastgs
    backends DO apply `_depth_mode`, so it must stay `'depth'` there -- a `'log'` render would pre-log
    and this loss would then log twice."""

    def forward(
            self,
            prediction: DecoderOutput,
            gaussians: Gaussians | GaussiansModule | None,
            global_step: int,
            gt_depth: Tensor | None = None,
            depth_mask: Tensor | None = None,
            near: Tensor | None = None,
            far: Tensor | None = None,
            **kwargs,
    ) -> Float[Tensor, ""]:
        if gt_depth is None or prediction.depth is None:
            return prediction.color.new_zeros(())
        # GT validity comes from the dataset's depth mask (constructed + cached next to the depth, with
        # a per-dataset strategy). Fall back to the near/far range when depth is supplied without a mask.
        valid = depth_mask if depth_mask is not None else ((gt_depth >= near) & (gt_depth <= far))
        if not valid.any():
            return prediction.color.new_zeros(())
        # Clamp the rendered depth into [near, far] before log(): empty / zero-coverage pixels render
        # depth ~0 and log(0) = -inf. Clamping (rather than masking them out) keeps such pixels
        # supervised with a bounded penalty toward the GT depth.
        render_depth = torch.minimum(torch.maximum(prediction.depth, near), far)
        return self.cfg.weight * (torch.log(gt_depth[valid]) - torch.log(render_depth[valid])).abs().mean()
