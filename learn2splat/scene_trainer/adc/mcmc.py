import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation

from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.adc.base import BaseStrategyCfg, GenericStrategyState
from learn2splat.scene_trainer.adc.base import _replace_objects, _add_to_objects
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class McmcStrategyCfg(BaseStrategyCfg):
    """MCMC densification (3DGS-MCMC, arXiv:2404.09591). Only ``apply_mcmc_strategy`` /
    ``inject_noise_to_position`` read these; the narrowed ``name`` discriminates the union variant."""
    name: Literal["mcmc"]
    noise_lr: float  # MCMC sampling noise learning rate, 0.0 for no MCMC sampling

    # Whether relocated Gaussians copy the sampled-alive *latent* state is controlled by the shared
    # BaseStrategyCfg.relocate_gaussian_state ("copy" == the old relocate_copy_state=True). The Adam
    # moments of relocated Gaussians follow FastGS regardless (see AdamInputSmoothing).

    # Cap on the scales used for the noise covariance (does NOT affect rendered scales). Needed
    # because knn_based saturates clamp_max_scale, producing covariances orders of magnitude larger
    # than vanilla's Adam-evolved scales; the resulting noise overflows the renderer's tile-binning
    # math and causes silent CUDA OOB. Rule of thumb: ~scene_scale / 5.
    noise_scale_cap: float = 1.0


@dataclass
class McmcStrategyState(GenericStrategyState):
    # Add MCMC specific state variables here
    binoms: Tensor  # [n_max, n_max]

    def external_pruning(self, valid_points_mask: Tensor) -> None:
        # MCMC has no 2D-gradient accumulators, nothing to prune
        pass

    @classmethod
    def initialize(cls, device: torch.device) -> "McmcStrategyState":

        # from gsplat
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)

        return cls(
            binoms=binoms.to(device)
        )


def update_mcmc_strategy_state(
        adc_state: McmcStrategyState
) -> None:
    """Updates adc_state in place."""
    pass


@torch.no_grad()
def inject_noise_to_position(
        gaussians: Gaussians | GaussiansModule,
        scaler: float,
        scale_cap: float = 1.0,
):
    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("noise injection not implemented for GaussiansModule")

    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    # rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    rotations = F.normalize(rotations_unnorm, dim=-1)

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        opacities = torch.sigmoid(opacities)  # [G]
        scales = torch.exp(scales)  # [G, 3]

    def _quat_scale_to_covar_preci(
            quats: Tensor,  # [..., 4],
            scales: Tensor,  # [..., 3],
            compute_covar: bool = True,
            compute_preci: bool = True,
            triu: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Converts quaternions and scales to covariance and precision matrices.

        Args:
            quats: Quaternions (No need to be normalized). [..., 4]
            scales: Scales. [..., 3]
            compute_covar: Whether to compute covariance matrices. Default: True. If False,
                the returned covariance matrices will be None.
            compute_preci: Whether to compute precision matrices. Default: True. If False,
                the returned precision matrices will be None.
            triu: If True, the return matrices will be upper triangular. Default: False.

        Returns:
            A tuple:

            - **Covariance matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
            - **Precision matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
        """
        batch_dims = quats.shape[:-1]
        assert quats.shape == batch_dims + (4,), quats.shape
        assert scales.shape == batch_dims + (3,), scales.shape
        quats = quats.contiguous()
        scales = scales.contiguous()
        covars, precis = quat_scale_to_covar_preci(
            quats, scales, compute_covar, compute_preci, triu
        )
        return covars if compute_covar else None, precis if compute_preci else None

    # Cap the scales used for the noise covariance only — does NOT change the rendered Gaussian
    # scales. knn_based's network saturates clamp_max_scale, producing covariances orders
    # of magnitude larger than vanilla's Adam-evolved scales; the resulting noise overflows the
    # renderer's tile-binning math and causes a silent CUDA OOB downstream. See BaseStrategyCfg.
    scales_for_noise = scales.clamp(max=scale_cap)
    covars, _ = _quat_scale_to_covar_preci(
        rotations,
        scales_for_noise,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    noise = (
            torch.randn_like(means)
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * scaler
    )
    noise = torch.einsum("bij,bj->bi", covars, noise)

    means.add_(noise)
    # means is a view of gaussians.means[0], so the add_ above already updated
    # the underlying storage. Do NOT reassign gaussians.means here — that would
    # replace the original leaf tensor with a requires_grad=False view (created
    # inside @torch.no_grad), breaking gradient flow on the next iteration.


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2 ** 24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


@torch.no_grad()
def _compute_relocation(
        opacities: Tensor,  # [N]
        scales: Tensor,  # [N, 3]
        ratios: Tensor,  # [N]
        binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Precomputed lookup table for binomial coefficients used in
          Equation 9 in the paper. [n_max, n_max]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    return compute_relocation(opacities, scales, ratios, binoms)


def relocate(
        gaussians: Gaussians | GaussiansModule,
        smoothers: dict[str, Any],
        adc_state: McmcStrategyState,
        min_opacity: float,
) -> int:
    """Relocates Gaussians based on MCMC strategy.

    Args:
        gaussians (Gaussians | GaussiansModule): Gaussian distributions to relocate.
        smoothers (dict[str, Any]): Optimizer smoothers.
        adc_state (McmcStrategyState): State of the ADC.

    Returns:
        int: Number of relocated Gaussians.
    """

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("noise injection not implemented for GaussiansModule")

    n_relocated = 0

    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    rotations = F.normalize(rotations_unnorm, dim=-1)
    harmonics = gaussians.harmonics.squeeze(0)  # [G, H]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 6] or None

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        opacities = torch.sigmoid(opacities)  # [G]
        scales = torch.exp(scales)  # [G, 3]

    dead_mask = opacities <= min_opacity
    n_gs = dead_mask.sum().item()
    if n_gs > 0:
        # Inplace relocate some dead Gaussians to the lives ones.
        n_relocated = int(n_gs)

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        n = len(dead_indices)

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = opacities[alive_indices].flatten()  # ensure its shape is [N,]
        sampled_idxs = _multinomial_sample(probs, n, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = _compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=scales[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binoms=adc_state.binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

        if gaussians.stores_activated:
            # already activated
            pass
        else:
            # deactivate
            new_opacities = torch.logit(new_opacities)  # [n]
            new_scales = torch.log(new_scales)  # [n, 3]

        # replace values (batch dim = 0, Gaussian dim = 1)
        gaussians.means[0, dead_indices] = means[sampled_idxs]
        gaussians.scales[0, dead_indices] = new_scales  # relocated scale (deactivated if needed)
        gaussians.opacities[0, dead_indices] = new_opacities  # relocated opacity (deactivated if needed)
        gaussians.rotations_unnorm[0, dead_indices] = rotations_unnorm[sampled_idxs]
        gaussians.harmonics[0, dead_indices] = harmonics[sampled_idxs]
        if covariances is not None:
            gaussians.covariances[0, dead_indices] = covariances[sampled_idxs]

        # replace smoothers
        _replace_objects(dead_indices, sampled_idxs, smoothers)

    return n_relocated


def add_new(
        gaussians: Gaussians | GaussiansModule,
        smoothers: dict[str, Any],
        adc_state: McmcStrategyState,
        cap_max: int,
        min_opacity: float
) -> int:
    """Adds new Gaussians based on MCMC strategy.

    Args:
        gaussians (Gaussians | GaussiansModule): Gaussian distributions to add new ones to.
        smoothers (dict[str, Any]): Optimizer smoothers.
        adc_state (McmcStrategyState): State of the ADC.

    Returns:
        int: Number of new Gaussians added.
    """

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("noise injection not implemented for GaussiansModule")

    n_new = 0

    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    # rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    rotations = F.normalize(rotations_unnorm, dim=-1)
    harmonics = gaussians.harmonics.squeeze(0)  # [G, H]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 6] or None

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        opacities = torch.sigmoid(opacities)  # [G]
        scales = torch.exp(scales)  # [G, 3]

    current_n_points = means.shape[0]
    n_target = min(cap_max, int(1.05 * current_n_points))
    n_gs = max(0, n_target - current_n_points)
    if n_gs > 0:
        # add new
        n_new = int(n_gs)

        eps = torch.finfo(torch.float32).eps
        probs = opacities.flatten()
        sampled_idxs = _multinomial_sample(probs, n_gs, replacement=True)
        new_opacities, new_scales = _compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=scales[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
            binoms=adc_state.binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

        # deactivate new opacities/scales for storage if needed
        if not gaussians.stores_activated:
            new_opacities = torch.logit(new_opacities.clamp(min=eps, max=1.0 - eps))
            new_scales = torch.log(new_scales)

        new_means = means[sampled_idxs]
        new_rotations_unnorm = rotations_unnorm[sampled_idxs]
        new_harmonics = harmonics[sampled_idxs]

        # append to existing Gaussians (batch dim = 0, Gaussian dim = 1)
        gaussians.means = torch.cat([gaussians.means, new_means.unsqueeze(0)], dim=1)
        gaussians.scales = torch.cat([gaussians.scales, new_scales.unsqueeze(0)], dim=1)
        gaussians.opacities = torch.cat([gaussians.opacities, new_opacities.unsqueeze(0)], dim=1)
        gaussians.rotations_unnorm = torch.cat([gaussians.rotations_unnorm, new_rotations_unnorm.unsqueeze(0)], dim=1)
        gaussians.harmonics = torch.cat([gaussians.harmonics, new_harmonics.unsqueeze(0)], dim=1)
        if gaussians.rotations is not None:
            gaussians.rotations = torch.cat(
                [gaussians.rotations, F.normalize(new_rotations_unnorm, dim=-1).unsqueeze(0)], dim=1
            )
        if covariances is not None:
            gaussians.covariances = torch.cat(
                [gaussians.covariances, covariances[sampled_idxs].unsqueeze(0)], dim=1
            )

        # add new entries to smoothers state
        _add_to_objects(n_new, smoothers)

    return n_new


@torch.no_grad()
def apply_mcmc_strategy(
        cfg: McmcStrategyCfg,
        step: int,
        gaussians: Gaussians | GaussiansModule,
        adc_state: McmcStrategyState,
        smoothers: dict[str, Any],
        lr: float,
        # zero_t: bool = False
) -> tuple[int, int, int, float | None, float | None]:
    """Applies MCMC strategy to the given Gaussian distributions.

    Args:
        cfg (BaseStrategyCfg): Configuration for the strategy.
        step (int): Current training step.
        gaussians (Gaussians | GaussiansModule): Gaussian distributions to apply the strategy to.
        adc_state (McmcStrategyState): State of the ADC.
        smoothers (dict[str, Any]): Optimizer smoothers.
        lr (float): Learning rate for "means" attribute of the GS.
    """

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("cloning not implemented for GaussiansModule")

    # Densification and Pruning
    nr_cloned, nr_splitted, nr_pruned = 0, 0, 0

    # check if should densify/prune
    if (
            step > cfg.refine_start_iter
            and step % cfg.refine_every == 0
            and step % cfg.reset_every >= cfg.pause_refine_after_reset
    ):
        # teleport dead GSs onto alive ones; their latent state copies/resets per
        # relocate_gaussian_state (applied inside KnnBasedOptimizerState.replace).
        n_relocated_gs = relocate(gaussians, smoothers, adc_state, cfg.min_opacity)

        # grow population up to cap_max; stop before refine_stop_iter so new Gaussians can converge
        n_new_gs = 0
        if step < cfg.refine_stop_iter:
            n_new_gs = add_new(gaussians, smoothers, adc_state, cfg.cap_max, cfg.min_opacity)

        torch.cuda.empty_cache()

        print(
            f"MCMC @ iter {step}: n_relocated {n_relocated_gs}, n_new_gs {n_new_gs}, total now {gaussians.means.shape[1]}"
        )

    # add noise to GSs
    inject_noise_to_position(
        gaussians, scaler=lr * cfg.noise_lr, scale_cap=cfg.noise_scale_cap
    )
    # no need to update smoothers

    return nr_cloned, nr_splitted, nr_pruned, None, None
