from .optimizer import Optimizer, OptimizerCfg
from .optimizer_adam import AdamOptimizerCfg, AdamOptimizer
from .optimizer_knn_based import KnnBasedOptimizer, KnnBasedOptimizerCfg
from .optimizer_learn2splat import Learn2SplatOptimizer
from .optimizer_resplat import ResplatOptimizerV1, ResplatOptimizerV2

SceneOptimizerCfg = KnnBasedOptimizerCfg | AdamOptimizerCfg


SCENE_OPTIMIZERS = {
    "none": None,
    "knn_based": KnnBasedOptimizer,
    "resplat_v1": ResplatOptimizerV1,
    "resplat_v2": ResplatOptimizerV2,
    "l2s": Learn2SplatOptimizer,
    "adam": AdamOptimizer,
}


def get_scene_optimizer(cfg: SceneOptimizerCfg | None) -> Optimizer | None:
    if cfg is None:
        print("Using scene optimizer: None")
        return None
    print(f"Using scene optimizer: {cfg.name}")
    scene_optimizer = SCENE_OPTIMIZERS[cfg.name]
    if scene_optimizer is None:
        return None
    scene_optimizer = scene_optimizer(cfg)
    return scene_optimizer
