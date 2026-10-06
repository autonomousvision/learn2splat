from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float, Bool
from torch import Tensor

from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.adc.base import BaseStrategyCfg, GenericStrategyState
from learn2splat.scene_trainer.adc.base import (
    _1d_indices_from_mask,
    _densification_postfix,
    _prune_objects,
    _split_objects,
    _clone_objects,
)
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


@dataclass
class VanillaStrategyCfg(BaseStrategyCfg):
    """Vanilla 3DGS / EDGS densification (clone/split/prune). Adds no fields beyond the common
    base; the narrowed ``name`` lets the StrategyCfg union discriminate this variant by name."""
    name: Literal["default", "edgs", "none"]


@dataclass
class VanillaStrategyState(GenericStrategyState):
    # for densification and pruning
    grad2d_norm_accum: Float[Tensor, "gaussian"]  # running accum of the norm of the image plane gradients for each GS
    denom: Float[Tensor, "gaussian"]
    radii2d: Float[Tensor, "gaussian"]  # max radius in 2D screen space observed for each GS
    scene_extent: Float[float, ""]

    def external_pruning(self, valid_points_mask: Tensor) -> None:
        if self.grad2d_norm_accum is not None:
            self.grad2d_norm_accum = self.grad2d_norm_accum[valid_points_mask]
        if self.radii2d is not None:
            self.radii2d = self.radii2d[valid_points_mask]
        if self.denom is not None:
            self.denom = self.denom[valid_points_mask]

    @classmethod
    def initialize(cls, nr_points: int, device: torch.device, scene_extent: int | float) -> "VanillaStrategyState":
        return cls(
            grad2d_norm_accum=torch.zeros(nr_points, device=device),
            denom=torch.zeros(nr_points, device=device),
            radii2d=torch.zeros(nr_points, device=device),
            scene_extent=scene_extent,
        )


def update_vanilla_strategy_state(
    adc_state: VanillaStrategyState,
    radii_2d: Tensor,  # [B, V, G, 2]
    means2d_grads: Tensor | None,  # [B, V, G, 2]
    visibility_mask: Bool[Tensor, "b v gaussian"],  # [B, V, G]
    v: int,  # number of views rendered
    w: int,  # image width
    h: int,  # image height
) -> None:
    """Updates adc_state in place."""
    # get gs ids from visibility mask

    visibility_mask = visibility_mask.squeeze(0)  # [V, G], assume batch size 1
    gs_ids = torch.where(visibility_mask)[1]  # [G_valid]
    assert visibility_mask.ndim == 2, "visibility_mask should be of shape [V, G]"

    if means2d_grads is not None:
        assert means2d_grads.ndim == 4 and means2d_grads.shape[-1] == 2, "means2d_grads should be of shape [B, V, G, 2]"
        means2d_grads = means2d_grads.squeeze(0)  # [V, G, 2], assume batch size 1
        grads = means2d_grads[visibility_mask]  # [G_valid, 2]

        # The per-renderer [-1, 1] NDC screen normalization now lives in the decoder
        # (Decoder.means2d_grad_to_ndc), so means2d_grads already arrives in NDC and this strategy
        # is renderer-agnostic. Only the view-count factor remains here.
        grads = grads * v

        # accumulate 2D grads norm
        adc_state.grad2d_norm_accum.index_add_(0, gs_ids, grads.norm(dim=-1))
        
        # accumulate denominator
        adc_state.denom.index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

    radii_2d = radii_2d.squeeze(0)  # [V, N, 2], assume batch size 1
    assert radii_2d.ndim == 3 and radii_2d.shape[2] == 2, "radii_2d should be of shape [V, G, 2]"

    radii_2d = radii_2d[visibility_mask]
    radii_max = radii_2d.max(dim=-1).values  # [V, N]
    # normalize radii to [0, 1] screen space
    radii_max /= float(max(w, h))

    # update radii2d
    adc_state.radii2d[gs_ids] = torch.maximum(adc_state.radii2d[gs_ids], radii_max)


def reset_adc_state(
        adc_state: VanillaStrategyState,
) -> None:
    """Resets adc_state in place."""
    adc_state.grad2d_norm_accum.zero_()
    adc_state.denom.zero_()
    adc_state.radii2d.zero_()
    torch.cuda.empty_cache()

def prune(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    prune_mask: Tensor,
) -> None:
    """Gaussians are updated in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("pruning not implemented for GaussiansModule")

    # NOTE: ADC runs per-Gaussian at batch size 1, so the batch dim is squeezed off here and added back after.
    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G,

    keep_idx = _1d_indices_from_mask(~prune_mask)

    # prune gaussians
    gaussians.means = means[keep_idx].unsqueeze(0)
    gaussians.scales = scales[keep_idx].unsqueeze(0)
    gaussians.opacities = opacities[keep_idx].unsqueeze(0)
    gaussians.rotations = rotations[keep_idx].unsqueeze(0)
    gaussians.rotations_unnorm = rotations_unnorm[keep_idx].unsqueeze(0)
    gaussians.harmonics = harmonics[keep_idx].unsqueeze(0)
    if covariances is not None:
        gaussians.covariances = covariances[keep_idx].unsqueeze(0)

    # prune adc state
    adc_state.grad2d_norm_accum = adc_state.grad2d_norm_accum[keep_idx]
    adc_state.radii2d = adc_state.radii2d[keep_idx]
    adc_state.denom = adc_state.denom[keep_idx]

def splitting(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    split_mask: Tensor,
    N=2,
    revised_opacity: bool = False,
) -> None:
    """Gaussians are updated in place.
    
    revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("splitting not implemented for GaussiansModule")
    
    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    # rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 3, 3]
    rotations = F.normalize(rotations_unnorm, dim=-1)

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        opacities = torch.sigmoid(opacities)  # [G]
        scales = torch.exp(scales)  # [G, 3]
    
    sel = _1d_indices_from_mask(split_mask)
    rest = _1d_indices_from_mask(~split_mask)
    
    # get params to split
    scales = scales[sel]  # [S, 3]
    rotations = rotations[sel]  # [S, 4]
    rotations_unnorm = rotations_unnorm[sel]  # [S, 4]
    opacities = opacities[sel]  # [S]
    means = means[sel]  # [S, 3]
    harmonics = harmonics[sel]  # [S, 3, d_sh]
    if covariances is not None:
        covariances = covariances[sel]  # [S, 3, 3]
    
    def _normalized_quat_to_rotmat(quat: Tensor) -> Tensor:
        """Convert normalized quaternion to rotation matrix.

        Args:
            quat: Normalized quaternion in wxyz convension. (..., 4)

        Returns:
            Rotation matrix (..., 3, 3)
        """
        assert quat.shape[-1] == 4, quat.shape
        w, x, y, z = torch.unbind(quat, dim=-1)
        mat = torch.stack(
            [
                1 - 2 * (y ** 2 + z ** 2),
                2 * (x * y - w * z),
                2 * (x * z + w * y),
                2 * (x * y + w * z),
                1 - 2 * (x ** 2 + z ** 2),
                2 * (y * z - w * x),
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x ** 2 + y ** 2),
            ],
            dim=-1,
        )
        return mat.reshape(quat.shape[:-1] + (3, 3))
    
    # new means
    rotmats = _normalized_quat_to_rotmat(rotations[:, [3, 0, 1, 2]])  # [N, 3, 3] xyzw to wxyz
    device = means.device
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(N, len(scales), 3, device=device),
    )  # [split, N, 3]
    new_means = (means + samples).reshape(-1, 3)  # [2N, 3]

    # new scales
    new_scales = (scales / 1.6).repeat(N, 1)  # [2N, 3]

    # new opacities
    if revised_opacity:
        new_opacities = (1.0 - torch.sqrt(1.0 - opacities)).repeat(N)  # [2N]
    else:
        new_opacities = opacities.repeat(N)  # [2N]

    if gaussians.stores_activated:
        # already activated
        pass
    else:
        # activate
        new_opacities = torch.logit(new_opacities)
        new_scales = torch.log(new_scales)

    # new rotations
    new_rotations = rotations.repeat(N, 1)  # [2N, 4]

    # new rotations unnorm
    new_rotations_unnorm = rotations_unnorm.repeat(N, 1)  # [2N, 4]

    # new harmonics
    new_harmonics = harmonics.repeat(N, 1, 1)  # [2N, 3, d_sh]

    # new covariances
    if covariances is not None:
        new_covariances = covariances.repeat(N, 1, 1)  # [2N, 3, 3]
    else:
        new_covariances = None

    def params_fn(v: Tensor, v_new: Tensor, dim: int) -> Tensor:
        v = v.squeeze(0)[rest].unsqueeze(0)
        return torch.cat([v, v_new.unsqueeze(0)], dim=dim)

    def state_fn(v: Tensor) -> Tensor:
        repeats = [2] + [1] * (v.dim() - 1)
        v_new = v[sel].repeat(repeats)
        return torch.cat([v[rest], v_new], dim=0)

    _densification_postfix(
        gaussians,
        adc_state,
        new_means,
        new_scales,
        new_opacities,
        new_rotations,
        new_rotations_unnorm,
        new_harmonics,
        new_covariances,
        params_fn,
        state_fn,
    )

def cloning(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,  # Vanilla or FastGS state (FastGS reuses this helper)
    clone_mask: Tensor,
) -> None:
    """Gaussians are updated in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("cloning not implemented for GaussiansModule")

    # get all params and remove batch dim
    means = gaussians.means.squeeze(0)  # [G, 3]
    scales = gaussians.scales.squeeze(0)  # [G, 3]
    opacities = gaussians.opacities.squeeze(0)  # [G]
    rotations = gaussians.rotations.squeeze(0)  # [G, 4]
    rotations_unnorm = gaussians.rotations_unnorm.squeeze(0)  # [G, 4]
    harmonics = gaussians.harmonics.squeeze(0)  # [G, 3, d_sh]
    covariances = gaussians.covariances.squeeze(0) if gaussians.covariances is not None else None  # [G, 3, 3]

    sel = _1d_indices_from_mask(clone_mask)

    # Clone
    new_means = means[sel]
    new_opacities = opacities[sel]
    new_scales = scales[sel]
    new_rotations_unnorm = rotations_unnorm[sel]
    new_rotations = rotations[sel]
    new_harmonics = harmonics[sel]
    new_covariances = covariances[sel] if covariances is not None else None

    def params_fn(v: Tensor, v_new: Tensor, dim: int) -> Tensor:
        return torch.cat([v, v_new.unsqueeze(0)], dim=dim)

    def state_fn(v: Tensor) -> Tensor:
        return torch.cat([v, v[sel]], dim=0)

    _densification_postfix(
        gaussians,
        adc_state,
        new_means,
        new_scales,
        new_opacities,
        new_rotations,
        new_rotations_unnorm,
        new_harmonics,
        new_covariances,
        params_fn,
        state_fn,
    )

@torch.no_grad()
def apply_vanilla_strategy(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState,
    smoothers: dict[str, Any],
) -> tuple[int, int, int, float | None, float | None]:
    
    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("cloning not implemented for GaussiansModule")
    
    # Densification and Pruning
    nr_cloned, nr_splitted, nr_pruned = 0, 0, 0

    if step >= cfg.refine_stop_iter:
        return nr_cloned, nr_splitted, nr_pruned, None, None

    # Calculate average 2D grads magnitude and scales
    grads: Tensor = adc_state.grad2d_norm_accum / adc_state.denom.clamp_min(1.0)  # [G]
    max_grad2d = grads.max().item()
    max_radii = adc_state.radii2d.max().item()

    # check if should densify/prune
    if (
        step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    ):
        device = gaussians.means.device
        grow_grad2d: float = cfg.grow_grad2d
        grow_scale3d: float = cfg.grow_scale3d
        prune_scale3d: float = cfg.prune_scale3d
        prune_scale2d: float | None = cfg.prune_scale2d
        grow_scale2d: float = cfg.grow_scale2d
        min_opacity: float = cfg.min_opacity
        prune_zero_radii: bool = cfg.prune_zero_radii

        if cfg.do_densify:
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Densification not implemented for GaussiansModule")
                scales: Tensor = gaussians.scales  # [G, 3]
            elif isinstance(gaussians, Gaussians):
                scales: Tensor = gaussians.scales.squeeze(0)  # [G, 3]
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    scales = torch.exp(scales)  # [G, 3]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")
            # Extract points that satisfy the gradient condition
            is_grad_high: Tensor = grads > grow_grad2d  # [G]

            is_small: Tensor = scales.max(dim=-1).values <= grow_scale3d * adc_state.scene_extent

            is_large: Tensor = ~is_small

            clone_mask: Tensor = is_grad_high & is_small

            split_mask: Tensor = is_grad_high & is_large

            if step < cfg.refine_scale2d_stop_iter:
                split_mask |= adc_state.radii2d > grow_scale2d

            # clone ---------------------------------------------------------------------
            
            # clone points
            cloning(
                gaussians=gaussians,
                adc_state=adc_state,
                clone_mask=clone_mask,
            )
            _clone_objects(clone_mask, smoothers)

            # update states
            nr_cloned = int(clone_mask.sum().item())

            # new GSs added by cloning will not be split
            split_mask = torch.cat(
                [
                    split_mask,
                    torch.zeros(nr_cloned, dtype=torch.bool, device=device),
                ]
            )

            # split ---------------------------------------------------------------------
                        
            # split points
            # No need to prune after splitting since we already removed the original points in _densification_postfix
            N = 2
            splitting(
                gaussians=gaussians,
                adc_state=adc_state,
                split_mask=split_mask,
                N=N,  # split each point into N points
            )
            _split_objects(split_mask, smoothers, N=N)
            nr_splitted = int(split_mask.sum().item())
            
        if cfg.do_prune:

            # prune ---------------------------------------------------------------------
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Densification not implemented for GaussiansModule")
                scales: Tensor = gaussians.scales  # [G, 3]
                opacities = gaussians.opacities  # [G]
            elif isinstance(gaussians, Gaussians):
                opacities = gaussians.opacities.squeeze(0)  # [G]
                scales = gaussians.scales.squeeze(0)  # [G, 3]
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    scales = torch.exp(scales)  # [G, 3]
                    opacities = torch.sigmoid(opacities)  # [G]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")

            # find points to prune and prune gaussians
            prune_mask = opacities < min_opacity

            if step > cfg.reset_every:
                is_too_big = scales.max(dim=-1).values > prune_scale3d * adc_state.scene_extent
                if step < cfg.refine_scale2d_stop_iter and prune_scale2d is not None:
                    is_too_big |= adc_state.radii2d > prune_scale2d
                prune_mask = prune_mask | is_too_big

            # invisible from training views
            if prune_zero_radii:
                raise NotImplementedError("prune_zero_radii not implemented yet")

            prune(gaussians, adc_state, prune_mask)
            _prune_objects(prune_mask, objects=smoothers)

            # update states
            nr_pruned = int(prune_mask.sum().item())

        # --------------------------------------------------------------------------

        # reset adc state
        reset_adc_state(adc_state)

        print(
            f"Densification/Pruning @ iter {step}: cloned {nr_cloned}, splitted {nr_splitted}, pruned {nr_pruned}, total now {gaussians.means.shape[1]}"
        )

    if cfg.do_opacity_reset:

        # Opacity reset
        if step % cfg.reset_every == 0 and step > 0:
            
            if isinstance(gaussians, GaussiansModule):
                raise NotImplementedError("Opacity reset not implemented for GaussiansModule")
            elif isinstance(gaussians, Gaussians):
                opacities = gaussians.opacities
                if gaussians.stores_activated:
                    # already activated
                    pass
                else:
                    # activate
                    opacities = torch.sigmoid(opacities)  # [G]
            else:
                raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")
            
            value = cfg.min_opacity * 2.0
            new_opacities = torch.min(opacities, torch.ones_like(opacities) * value)

            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # deactivated
                new_opacities = torch.logit(new_opacities)

            gaussians.opacities = new_opacities

            # reset momentums of opacities
            smoothers["opacities"].zero_out()  # opacity reset zeros Adam moments, keeps t (FastGS)
            print("Opacity reset @ iter", step)

    if cfg.reduce_opacity:
        # Slightly reduce opacity every few steps (from EDGS)
        if step % cfg.reduce_every == 0:
            
            opacities = gaussians.opacities
            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # activate
                opacities = torch.sigmoid(opacities)  # [G]
                
            opacities_new = opacities * cfg.reduce_factor
            
            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # deactivate
                opacities_new = torch.logit(opacities_new)
                
            gaussians.opacities = opacities_new
    
    return nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d
