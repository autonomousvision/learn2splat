from dataclasses import dataclass
from typing import Literal, Callable
from torch import Tensor
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from learn2splat.model.types import Gaussians

@dataclass
class GenericStrategyState:
    pass

@dataclass
class BaseStrategyCfg:
    """Common adaptive-density-control config: the densification settings every strategy shares, plus
    the two read on cross-strategy paths (``cap_max`` by the config bridge, ``fallback_means_lr`` by
    the optimizer). Strategy-specific settings live in the ``name``-narrowed subclasses —
    ``VanillaStrategyCfg`` (vanilla.py), ``McmcStrategyCfg`` (mcmc.py), ``FastGSStrategyCfg``
    (fastgs.py) — which the ``StrategyCfg`` union in ``__init__.py`` discriminates by ``name``."""
    name: Literal["default", "edgs", "mcmc", "none", "fastgs"]

    do_densify: bool
    do_prune: bool
    do_opacity_reset: bool

    cap_max: int  # Maximum number of GSs, -1 for no cap (general densification cap; read by config_bridge)

    pause_refine_after_reset: int
    refine_every: int
    reset_every: int
    refine_start_iter: int
    refine_stop_iter: int
    refine_scale2d_stop_iter: int  # Until which iteration 2D scale based refinement / pruning is applied

    grow_grad2d: float # GSs with image plane gradient above this value will be split/duplicated
    grow_scale3d: float # GSs with scale below this value will be duplicated. Above will be split
    prune_scale3d: float # GSs with scale above this value will be pruned
    prune_scale2d: float | None  # GSs with 2d screen radius (normalized by max(w,h)) above this are pruned; None disables screen-space big-point pruning
    grow_scale2d: float  # GSs with 2d scale (normalized by image resolution) above this value will be split
    min_opacity: float  # GSs with opacity below this value will be pruned
    prune_zero_radii: bool  # GSs with zero radii in screen space will be pruned

    reduce_opacity: bool # Slightly reduce opacity every few steps
    reduce_factor: float # Factor to reduce opacity by
    reduce_every: int # Reduce opacity every N iterations

    # Fallback means lr used when the optimizer has no means_lr_scheduler (read on the shared apply
    # path in optimizer.py; also sets the MCMC noise scale via McmcStrategyCfg.noise_lr).
    fallback_means_lr: float

    # How each densification op initializes a new Gaussian's *learned latent state* (the per-Gaussian
    # vector the knn optimizer carries). This governs the latent state ONLY — the Adam optimizer
    # moments are not affected and follow FastGS (see AdamInputSmoothing). Per op since they differ in
    # meaning (clone duplicates in place, split repositions, relocate respawns, add has no parent):
    #   "copy"   - copy the parent/source state (clone/split parent, MCMC-relocate source)
    #   "zero"   - fresh zeroed state
    #   "random" - fresh random state, randn * init_state_scale (the scene-start init)
    # Each fires only for the strategy that uses it: clone/split under vanilla/fastgs, relocate/add
    # under MCMC (the others are inert there). Applied by KnnBasedOptimizerState, which stores these.
    clone_gaussian_state: Literal["copy", "zero", "random"]
    split_gaussian_state: Literal["copy", "zero", "random"]
    relocate_gaussian_state: Literal["copy", "zero", "random"]
    add_gaussian_state: Literal["copy", "zero", "random"]


def is_densify_step(cfg: BaseStrategyCfg, step: int) -> bool:
    """Densification gate shared by the strategies' callers — the in-loop optimizer (optimizer.py)
    and test-time postprocessing: a refine step inside the active window, clear of the post-reset
    pause. Used to time the FastGS multi-view score render and to detect module-modifying steps."""
    return (
        step < cfg.refine_stop_iter
        and step > cfg.refine_start_iter
        and step % cfg.refine_every == 0
        and step % cfg.reset_every >= cfg.pause_refine_after_reset
    )


def is_final_prune_step(cfg: BaseStrategyCfg, step: int) -> bool:
    """FastGS phase-2 gate (train.py:153 ``iter % 3000 == 0 and iter > 15000``): once densification
    has stopped, an aggressive multi-view prune fires every ``reset_every``. Shared by the strategy's
    callers to time the score render and the final prune. Disjoint from ``is_densify_step``
    (that needs step < refine_stop_iter; this needs step > refine_stop_iter)."""
    return step > cfg.refine_stop_iter and step % cfg.reset_every == 0


def _prune_objects(prune_mask, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].prune(prune_mask)

# NOTE: the densification helpers do NOT thread a state policy to the smoothers. Each smoother
# decides for itself: KnnBasedOptimizerState applies its per-op latent-state policy; the
# AdamInputSmoothing moments use their FastGS-reference default (zero_t=False).
def _clone_objects(clone_mask, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].clone(clone_mask)

def _split_objects(split_mask, objects, N):
    for key in objects:
        if objects[key] is not None:
            objects[key].split(split_mask, N)

def _add_to_objects(nr_new, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].add(nr_new)

def _replace_objects(dest_indices, from_indices, objects):
    for key in objects:
        if objects[key] is not None:
            objects[key].replace(from_indices, dest_indices)

def _1d_indices_from_mask(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None
    return mask.nonzero(as_tuple=True)[0]
            
            
def _densification_postfix(
    gaussians: Gaussians | GaussiansModule,
    adc_state: GenericStrategyState,
    new_means: Tensor,
    new_scales: Tensor,
    new_opacities: Tensor,
    new_rotations: Tensor,
    new_rotations_unnorm: Tensor,
    new_harmonics: Tensor,
    new_covariances: Tensor | None,
    params_fn: Callable[[Tensor], Tensor],
    state_fn: Callable[[Tensor], Tensor],
) -> None:
    """Updates gaussians and adc_state in place."""

    if isinstance(gaussians, GaussiansModule):
        raise NotImplementedError("_densification_postfix not implemented for GaussiansModule")

    # update gaussians
    gaussians.means = params_fn(gaussians.means, new_means, dim=1)
    gaussians.scales = params_fn(gaussians.scales, new_scales, dim=1)
    gaussians.opacities = params_fn(gaussians.opacities, new_opacities, dim=1)
    gaussians.rotations = params_fn(gaussians.rotations, new_rotations, dim=1)
    gaussians.rotations_unnorm = params_fn(gaussians.rotations_unnorm, new_rotations_unnorm, dim=1)
    gaussians.harmonics = params_fn(gaussians.harmonics, new_harmonics, dim=1)
    if gaussians.covariances is not None and new_covariances is not None:
        gaussians.covariances = params_fn(gaussians.covariances, new_covariances, dim=1)

    # update adc state
    adc_state.grad2d_norm_accum = state_fn(adc_state.grad2d_norm_accum)
    adc_state.denom = state_fn(adc_state.denom)
    adc_state.radii2d = state_fn(adc_state.radii2d)
