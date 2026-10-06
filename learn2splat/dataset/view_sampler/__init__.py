from typing import Any

from ...misc.step_tracker import StepTracker
from ..data_types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_ids import ViewSamplerIDs, ViewSamplerIDsCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_bounded_v2 import ViewSamplerBoundedV2, ViewSamplerBoundedV2Cfg
from learn2splat.dataset.view_sampler.view_sampler_dense import ViewSamplerDense, ViewSamplerDenseCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "ids": ViewSamplerIDs,
    "dense": ViewSamplerDense,  # colmap datasets
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,  # during evaluation
    "boundedv2": ViewSamplerBoundedV2,  # during training
}

ViewSamplerCfg = (
        ViewSamplerArbitraryCfg
        | ViewSamplerBoundedCfg
        | ViewSamplerEvaluationCfg
        | ViewSamplerAllCfg
        | ViewSamplerBoundedV2Cfg
        | ViewSamplerDenseCfg
        | ViewSamplerIDsCfg
)


def get_view_sampler(
        cfg: ViewSamplerCfg,
        stage: Stage,
        overfit: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    print("Using view sampler:", cfg.name)
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
