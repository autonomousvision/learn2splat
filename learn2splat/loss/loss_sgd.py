from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from learn2splat.loss import Loss
from learn2splat.model.decoder.decoder import DecoderOutput
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossSGDCfg:
    l1_loss: bool  # use L1 instead of L2 on the delta-vs-gradient error
    clamp_large_error: float  # if > 0, clamp the loss to this maximum

@dataclass
class LossSGDCfgWrapper:
    sgd: LossSGDCfg


class LossSGD(Loss[LossSGDCfg, LossSGDCfgWrapper]):
    """Debugging / sanity-check loss: supervises the optimizer's predicted per-Gaussian deltas to match
    the true gradients, i.e. checks the learned optimizer can reproduce a plain gradient-descent step
    (L2, or L1 if cfg.l1_loss)."""

    def forward(
        self,
        prediction: DecoderOutput,
        gaussians: Gaussians | GaussiansModule | None,
        global_step: int,
        valid_depth_mask: Tensor | None,
        **kwargs,
    ) -> Float[Tensor, ""]:

        if gaussians is None:
            raise ValueError("Gaussians must be provided for LossSGD.")

        predicted_deltas = gaussians.deltas
        gt_gradients = gaussians.gradients

        # cast to float16 if necessary
        if predicted_deltas.dtype != gt_gradients.dtype:
            gt_gradients = gt_gradients.to(predicted_deltas.dtype)

        if self.cfg.l1_loss:
            loss = (predicted_deltas - gt_gradients).abs().mean()
        else:
            loss = ((predicted_deltas - gt_gradients) ** 2).mean()
        if self.cfg.clamp_large_error > 0:
            loss = loss.clamp(max=self.cfg.clamp_large_error)
        return loss
