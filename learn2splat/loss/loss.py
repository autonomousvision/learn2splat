from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..misc.batchify import batched_select
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str
    use_in_inner_steps: bool = True  # False → skip in compute_losses; subclass handles its own call site
    # False → loss is independent of the rendered views (e.g. on the Gaussians), so it is added
    # once per step rather than once per target/context view set.
    view_dependent: bool = True

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @property
    def is_disabled(self) -> bool:
        """True when the loss has an explicit weight of 0, so the aggregation can skip it entirely
        instead of computing a term that contributes nothing. Losses that name their weight field
        differently (the depth losses) have no `weight`, so they are never treated as disabled here."""
        return getattr(self.cfg, "weight", None) == 0

    @abstractmethod
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
        pass

    @staticmethod
    def extract_pred_gt(curr_gt_rgb, prediction, error_idx, valid_depth_mask):
        # curr_gt_rgb is already subsampled to the rendered views (opt_batch_size subset);
        # error_idx further subsamples both gt and pred to the views used for the loss.
        pred_rgb = prediction.color  # [B, V_rendered, C, H, W]
        gt_rgb = curr_gt_rgb
        if error_idx is not None:
            gt_rgb = batched_select(gt_rgb, error_idx)
            pred_rgb = batched_select(pred_rgb, error_idx)
            if valid_depth_mask is not None:
                valid_depth_mask = batched_select(valid_depth_mask, error_idx)

        return gt_rgb, pred_rgb, valid_depth_mask
