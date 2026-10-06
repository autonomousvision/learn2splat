from learn2splat.scene_trainer.optimizer import KnnBasedOptimizer


class ResplatOptimizerV1(KnnBasedOptimizer):
    OPTIMIZER_NAME = "resplat_v1"

class ResplatOptimizerV2(KnnBasedOptimizer):
    OPTIMIZER_NAME = "resplat_v2"
