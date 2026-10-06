from dataclasses import dataclass
from typing import Literal

from .initializer import InitializerCfg
from .optimizer import SceneOptimizerCfg
from ..model.decoder import DecoderCfg


@dataclass
class SceneTrainerCfg:
    scene_initializer: InitializerCfg  # default + structural representative; also the eval-path init
    scene_optimizer: SceneOptimizerCfg | None
    decoder: DecoderCfg
    use_fsdp: bool
    train_scene_opt: bool
    num_update_steps: int
    # Views per render call for the init and saved-step renders; -1 renders the full view set at
    # once. The videos chunk by meta_trainer.test.render_chunk_size instead (TODO: unify the two).
    iter_batch_size: int

    train_min_refine: int
    train_max_refine: int

    opt_batch_size: int  # if -1, use full batch
    opt_batch_size_min: int  # if > 0, use random sub-batch
    opt_batch_size_max: int  # if > 0, use random sub-batch
    opt_batch_strategy: Literal["random", "sequential", "neighbors", "fps"]  # strategy for sub-batch
    sh_degree_interval: int  # 0 = disabled; N = steps between SH degree increments (like gsplat simple_trainer)

    # Optional per-source initializer overrides for mixed-dataset training, keyed by the mixed
    # dataset's source key. A scene from source K inits with scene_initializers[K] if present, else
    # scene_initializer. None -> every scene uses scene_initializer (single datasets / uniform mixes).
    scene_initializers: dict[str, InitializerCfg] | None = None

    def __post_init__(self):
        if self.scene_optimizer is not None:
            # Pass the per-source overrides too: the optimizer sizes itself around ALL initializers it
            # will refine (point-cloud constraints + a feature branch for a feature init like resplat).
            self.scene_optimizer.update(self.scene_initializer, self.scene_initializers)
        # Per-source inits must share sh_degree and init_gaussian_multiple with the default: the
        # optimizer sizes its channels around both (position encoding differences are fine — inits
        # always unproject to 3D means before the optimizer sees the Gaussians).
        if self.scene_initializers is not None:
            base_sh_d = self.scene_initializer.get_sh_d()
            base_igm = self.scene_initializer.get_init_gaussian_multiple()
            for key, init in self.scene_initializers.items():
                got = (init.get_sh_d(), init.get_init_gaussian_multiple())
                assert got == (base_sh_d, base_igm), (
                    f"per-source initializer '{key}' has sh_degree/init_gaussian_multiple "
                    f"{got[0]}/{got[1]} but the shared optimizer expects "
                    f"{base_sh_d}/{base_igm} (from scene_initializer)")
