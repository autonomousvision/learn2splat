from typing import Any
from learn2splat.model.types import Gaussians
import torch
from torch import Tensor
from jaxtyping import Bool
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from learn2splat.scene_trainer.adc.base import BaseStrategyCfg
from learn2splat.scene_trainer.adc.mcmc import McmcStrategyState, McmcStrategyCfg, update_mcmc_strategy_state
from learn2splat.scene_trainer.adc.vanilla import VanillaStrategyState, VanillaStrategyCfg, update_vanilla_strategy_state
from learn2splat.scene_trainer.adc.fastgs import FastGSStrategyState, FastGSStrategyCfg, update_fastgs_strategy_state

# Discriminated union of the per-strategy configs, resolved by the `name` Literal (mirrors the
# DecoderCfg / SceneOptimizerCfg unions). OptimizerCfg.refiner and the postprocessing adc config
# are typed with this so dacite builds the right variant — vanilla/edgs/none -> VanillaStrategyCfg,
# mcmc -> McmcStrategyCfg, fastgs -> FastGSStrategyCfg.
StrategyCfg = VanillaStrategyCfg | McmcStrategyCfg | FastGSStrategyCfg

def init_strategy_state(
    cfg: BaseStrategyCfg,
    **kwargs
) -> VanillaStrategyState | McmcStrategyState | FastGSStrategyState:

    if cfg.name == "mcmc":
        return McmcStrategyState.initialize(
            device=kwargs["device"]
        )
    elif cfg.name == "fastgs":
        return FastGSStrategyState.initialize(
            nr_points=kwargs["nr_points"],
            device=kwargs["device"],
            scene_extent=kwargs["scene_extent"],
        )
    elif cfg.name in ["default", "edgs", "none"]:
        return VanillaStrategyState.initialize(
            nr_points=kwargs["nr_points"],
            device=kwargs["device"],
            scene_extent=kwargs["scene_extent"],
        )
    else:
        raise NotImplementedError(f"ADC strategy state initialization not implemented for {cfg.name}")

def update_strategy_state(
    adc_state: VanillaStrategyState | McmcStrategyState | FastGSStrategyState,
    **kwargs
) -> None:
    """Updates adc_state in place."""

    if isinstance(adc_state, FastGSStrategyState):
        # means2d_abs_grads is the Abs-GS split signal; the in-loop callers do not produce it
        # (only the postprocessing loop does), so it defaults to None -> split on the normal grad.
        return update_fastgs_strategy_state(
            adc_state,
            radii_2d=kwargs["radii_2d"],
            means2d_grads=kwargs["means2d_grads"],
            means2d_abs_grads=kwargs.get("means2d_abs_grads"),
            visibility_mask=kwargs["visibility_mask"],
            v=kwargs["v"],
            w=kwargs["w"],
            h=kwargs["h"],
        )
    elif isinstance(adc_state, VanillaStrategyState):
        return update_vanilla_strategy_state(
            adc_state,
            radii_2d=kwargs["radii_2d"],
            means2d_grads=kwargs["means2d_grads"],
            visibility_mask=kwargs["visibility_mask"],
            v=kwargs["v"],
            w=kwargs["w"],
            h=kwargs["h"],
        )
    elif isinstance(adc_state, McmcStrategyState):
        return update_mcmc_strategy_state(adc_state)
    else:
        raise NotImplementedError(f"ADC strategy state update not implemented for {type(adc_state)}")


def apply_adc_strategy(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState | McmcStrategyState | FastGSStrategyState,
    smoothers: dict[str, Any],
    **kwargs
) -> tuple[int, int, int, float | None, float | None]:
    """Applies ADC strategy and returns number of cloned, split, pruned GSs."""

    # The per-op latent-state policy (clone/split/relocate/add_gaussian_state) is applied by
    # KnnBasedOptimizerState itself; the dispatch threads nothing extra to the strategies.
    if cfg.name == "fastgs":
        from learn2splat.scene_trainer.adc.fastgs import apply_fastgs_strategy
        assert isinstance(adc_state, FastGSStrategyState), "adc_state type mismatch."
        return apply_fastgs_strategy(
            cfg,
            step=step,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=smoothers,
            importance_score=kwargs.get("importance_score"),
            pruning_score=kwargs.get("pruning_score"),
        )
    elif cfg.name in ["default", "edgs", "none"]:
        from learn2splat.scene_trainer.adc.vanilla import apply_vanilla_strategy
        assert isinstance(adc_state, VanillaStrategyState), "adc_state type mismatch."
        return apply_vanilla_strategy(
            cfg,
            step=step,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=smoothers,
        )
    elif cfg.name == "mcmc":
        from learn2splat.scene_trainer.adc.mcmc import apply_mcmc_strategy
        assert isinstance(adc_state, McmcStrategyState), "adc_state type mismatch."
        return apply_mcmc_strategy(
            cfg,
            step=step,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=smoothers,
            lr=kwargs["lr"],
            # zero_t=zero_t
        )
    else:
        raise NotImplementedError(f"ADC strategy not implemented for {cfg.name}")

    
@torch.no_grad()
def post_backward(
    cfg: BaseStrategyCfg,
    step: int,
    gaussians: Gaussians | GaussiansModule,
    adc_state: VanillaStrategyState | McmcStrategyState | FastGSStrategyState,
    smoothers: dict[str, Any],
    radii_2d: Tensor,  # [B, V, G, 2]
    means2d_grads: Tensor | None,  # [B, V, G, 2]
    visibility_mask: Bool[Tensor, "b v gaussian"],  # [B, V, G]
    iter_batch_size: int,
    w: int,
    h: int,
    means2d_abs_grads: Tensor | None = None,  # [B, V, G, 2] FastGS Abs-GS split signal (None elsewhere)
    **kwargs
) -> tuple[int, int, int, float | None, float | None]:

    # update adc state
    update_strategy_state(
        adc_state,
        radii_2d=radii_2d,
        means2d_grads=means2d_grads,
        means2d_abs_grads=means2d_abs_grads,
        visibility_mask=visibility_mask,
        v=iter_batch_size,
        w=w,
        h=h,
    )

    return apply_adc_strategy(
        cfg,
        step=step,
        gaussians=gaussians,
        adc_state=adc_state,
        smoothers=smoothers,
        **kwargs
    )
