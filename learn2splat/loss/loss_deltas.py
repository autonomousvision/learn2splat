from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from learn2splat.loss import Loss
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class LossDeltasCfg:
    weight: float | int
    exclude_by_norm_grad: bool
    exclude_by_norm_grad_opposite: bool
    eps: float
    apply_after_step: int

@dataclass
class LossDeltasCfgWrapper:
    deltas: LossDeltasCfg


class LossDeltas(Loss[LossDeltasCfg, LossDeltasCfgWrapper]):
    """L1 regularization on the optimizer's predicted per-Gaussian deltas, keeping the updates small.

    With cfg.exclude_by_norm_grad the penalty is restricted to Gaussians with small gradients (and,
    with exclude_by_norm_grad_opposite, those whose delta already agrees in sign with the gradient).
    Returns 0 until global_step >= cfg.apply_after_step.

    View-independent (only reads the predicted deltas), so it is added once per optimizer step
    rather than once per target/context view set.
    """
    view_dependent = False

    def forward(
        self,
        prediction,
        gaussians: Gaussians | GaussiansModule | None,
        global_step: int,
        **kwargs,
    ) -> Float[Tensor, ""]:

        cfg = self.cfg
        if gaussians is None:
            raise ValueError("Gaussians must be provided for LossDeltas.")

        predicted_deltas = gaussians.deltas

        # Before the specified step, don't apply the loss.
        if global_step < cfg.apply_after_step:
            return predicted_deltas.new_zeros(())

        if not cfg.exclude_by_norm_grad:
            return predicted_deltas.abs().mean() * cfg.weight

        norm_g = gaussians.norm_gradients
        if norm_g is None:
            return predicted_deltas.abs().mean() * cfg.weight

        g = gaussians.gradients
        eps = cfg.eps
        g_abs = g.abs()

        # Condition 1: small gradients
        cond_small = g_abs < eps
        mask = cond_small

        # Condition 2: large gradients but opposite sign
        # deltas are added (sgd substract), so in practice we want to exclude when they have the same sign
        if cfg.exclude_by_norm_grad_opposite:
            cond_opposite = (g_abs > self.cfg.eps) & (norm_g.sign() == predicted_deltas.sign())
            # Combine both
            mask = cond_small | cond_opposite

        if not mask.any():
            return predicted_deltas.new_zeros(())

        # predicted_deltas[mask] creates a new tensor
        # return predicted_deltas[mask].abs().mean() * cfg.weight
        # alternative without indexing
        mask_f = mask.to(predicted_deltas.dtype)

        loss = (predicted_deltas.abs() * mask_f).sum() / mask_f.sum()
        return loss * cfg.weight
