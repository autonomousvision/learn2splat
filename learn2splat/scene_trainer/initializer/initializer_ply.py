from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from learn2splat.dataset.data_types import BatchedViews


# from learn2splat.dataset.colmap.utils import Parser
# from learn2splat.experimental.initializers_utils import knn, points_to_gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.model.ply_export import load_gaussians_ply
from learn2splat.scene_trainer.initializer.initializer import NonlearnedInitializer, InitializerOutput, NonlearnedInitializerCfg


@dataclass
class InitializerPlyCfg(NonlearnedInitializerCfg):
    name: Literal["ply"]
    path: Path
    # normalize_world_space: bool
    # scaling_factor: float
    # init_opacity: float
    sh_degree: int
    # dl3dv_settings: bool
    ply_filename: str = "gaussians.ply"  # relative path under the scene dir; can include subdirs, e.g. "iteration_20000/point_cloud.ply"

    def get_sh_d(self):
        sh_d = (self.sh_degree + 1) ** 2
        return sh_d


class InitializerPly(NonlearnedInitializer[InitializerPlyCfg]):
    def __init__(self, cfg: InitializerPlyCfg) -> None:
        super().__init__(cfg)

    def forward(
            self,
            context: BatchedViews,
            visualization_dump: Optional[dict] = None,
            **kwargs
    ) -> InitializerOutput:
        device = context["extrinsics"].device
        verbose = False

        # assert COLMAP dir exists
        if not self.cfg.path.exists():
            raise ValueError(f"COLMAP dir {self.cfg.path} does not exist.")
        
        if "scene" in kwargs:
            scene_name = kwargs["scene"]
            assert len(scene_name) == 1, f"Only single scene initialization supported. {scene_name}"
            scene_name = scene_name[0]
            if verbose:
                print(f"Initializing scene '{scene_name}' from COLMAP at {self.cfg.path}.")
            datadir = self.cfg.path / scene_name
            if not datadir.exists():
                raise ValueError(f"COLMAP scene dir {datadir} does not exist.")
        else:
            scene_name = None
            datadir = self.cfg.path

        # ply_filename supports {scene} substitution and glob patterns. The glob
        # is matched relative to datadir; exactly one match is required.
        rel = self.cfg.ply_filename
        if scene_name is not None and "{scene}" in rel:
            rel = rel.replace("{scene}", scene_name)
        if any(c in rel for c in "*?["):
            matches = sorted(datadir.glob(rel))
            if not matches:
                raise FileNotFoundError(f"No PLY matched pattern {rel!r} under {datadir}.")
            if len(matches) > 1:
                raise ValueError(f"PLY pattern {rel!r} matched {len(matches)} files under {datadir}; expected one. Matches: {matches}")
            ply_path = matches[0]
        else:
            ply_path = datadir / rel

        # pre-activation values on device
        gaussians = load_gaussians_ply(ply_path, max_sh_degree=self.cfg.sh_degree)
        
        # move to device
        gaussians = gaussians.to(device)
        
        # build covariances
        covariances = build_covariance(scale=gaussians.scales, rotation_xyzw=gaussians.rotations)
        gaussians.covariances = covariances
        
        return InitializerOutput(
            gaussians=gaussians,
            features=None,
            depths=None
        )
