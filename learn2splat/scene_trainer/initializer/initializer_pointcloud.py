from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from learn2splat.experimental.initializers_utils import nn_dist2_for_scales, points_to_gaussians
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.scene_trainer.initializer.initializer import NonlearnedInitializer, InitializerOutput, NonlearnedInitializerCfg


@dataclass
class InitializerPointcloudCfg(NonlearnedInitializerCfg):
    name: Literal["pointcloud"]
    path: Path  # Directory containing <scene_id>.ply files
    scaling_factor: float
    init_opacity: float
    sh_degree: int
    filter_zero_rgb: bool
    # 4x4 world transform applied to point cloud positions.
    # Needed when the PLY is in a different coordinate system than the camera poses.
    # For ScanNet++/NeRFstudio: the PLY is in COLMAP space while cameras are in
    # NeRFstudio space. The transform is (x,y,z) -> (y,x,-z), i.e.:
    #   [[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]
    # Set to null to skip.
    world_transform: Optional[list]

    def get_gaussian_param_num(self):
        sh_d = (self.sh_degree + 1) ** 2
        return 3 + 4 + 3 * sh_d + 2 + 1

    def get_sh_d(self):
        return (self.sh_degree + 1) ** 2


class InitializerPointcloud(NonlearnedInitializer[InitializerPointcloudCfg]):
    def __init__(self, cfg: InitializerPointcloudCfg) -> None:
        super().__init__(cfg)

    @staticmethod
    def _load_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load Open3D binary PLY file.

        Returns:
            xyz: [N, 3] float32 array of 3D positions
            rgb: [N, 3] uint8 array of colors
        """
        with open(ply_path, "rb") as f:
            num_vertices = 0
            while True:
                line = f.readline().decode("ascii").strip()
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                if line == "end_header":
                    break

            dtype = np.dtype([
                ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ])
            data = np.frombuffer(f.read(num_vertices * dtype.itemsize), dtype=dtype)

        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
        return xyz, rgb

    def forward(
            self,
            context,
            visualization_dump: Optional[dict] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ) -> InitializerOutput:
        # Resolve PLY path
        if "scene" in kwargs:
            scene_name = kwargs["scene"]
            assert len(scene_name) == 1, f"Only single scene initialization supported. {scene_name}"
            scene_name = scene_name[0]
            ply_path = self.cfg.path / f"{scene_name}.ply"
        else:
            raise ValueError("Scene name is required for pointcloud initializer.")

        if not ply_path.exists():
            raise ValueError(f"PLY file {ply_path} does not exist.")

        print(f"Loading point cloud from {ply_path}")

        # Load PLY
        points_xyz, points_rgb = self._load_ply(ply_path)
        print(f"Loaded {points_xyz.shape[0]} points.")

        xyz = torch.from_numpy(points_xyz).float().to(device)
        rgbs = torch.from_numpy(points_rgb / 255.0).float().to(device)

        # Apply world transform to align point cloud with camera coordinate system
        if self.cfg.world_transform is not None:
            T = torch.tensor(self.cfg.world_transform, dtype=torch.float32, device=device)
            # Transform: new_xyz = (T @ [xyz, 1])[:3]
            xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device)], dim=-1)  # [N, 4]
            xyz = (T @ xyz_h.T)[:3].T  # [N, 3]

        # Filter zero-RGB points
        if self.cfg.filter_zero_rgb:
            valid_mask = rgbs.sum(dim=-1) > 0
            xyz = xyz[valid_mask]
            rgbs = rgbs[valid_mask]

        # ── Step 1+2: subsample the point cloud ("sample Gaussians") ─────────────
        # Optional subsampling augmentation, then down to the fixed count for DDP consistency.
        xyz, rgbs = self.subsample_points(xyz, rgbs)

        # KNN → scales
        dist2_avg = nn_dist2_for_scales(xyz)  # [N]
        dist_avg = torch.sqrt(dist2_avg)
        scales = dist_avg.unsqueeze(-1).repeat(1, 3)  # [N, 3]
        opacities = torch.full((xyz.shape[0],), self.cfg.init_opacity)

        # Pad to fixed size for distributed training
        fixed_num = self.fixed_gaussians_num
        if self.training and fixed_num is not None:
            current_num = xyz.shape[0]
            if current_num < fixed_num:
                pad_size = fixed_num - current_num
                xyz = F.pad(xyz, (0, 0, 0, pad_size), mode='constant', value=0.0)
                rgbs = F.pad(rgbs, (0, 0, 0, pad_size), mode='constant', value=0.0)
                scales = F.pad(scales, (0, 0, 0, pad_size), mode='constant', value=1e-10)
                opacities = F.pad(opacities, (0, pad_size), mode='constant', value=1e-10)

        points_dict = {
            "xyz": xyz,
            "rgb": rgbs,
            "scales": scales * self.cfg.scaling_factor,
            "opacities": opacities,
        }

        # Convert to Gaussian representation
        gaussians_dict = points_to_gaussians(points_dict, sh_degree=self.cfg.sh_degree, device=device)

        means = gaussians_dict["xyz"]
        sh0 = gaussians_dict["sh0"]
        shN = gaussians_dict["shN"]
        harmonics = torch.cat([sh0, shN], dim=1)  # [N, sh_d, 3]
        harmonics = harmonics.permute(0, 2, 1)  # [N, 3, sh_d]
        rotations_unnorm = gaussians_dict["rotations_unnorm"]

        opacities = torch.sigmoid(gaussians_dict["opacities_raw"])
        scales = torch.exp(gaussians_dict["scales_raw"])
        rotations = F.normalize(gaussians_dict["rotations_unnorm"], dim=-1)
        covariances = build_covariance(scale=scales, rotation_xyzw=rotations)

        gaussians = Gaussians(
            means=means.unsqueeze(0),
            covariances=covariances.unsqueeze(0),
            harmonics=harmonics.unsqueeze(0),
            opacities=opacities.unsqueeze(0),
            scales=scales.unsqueeze(0),
            rotations=rotations.unsqueeze(0),
            rotations_unnorm=rotations_unnorm.unsqueeze(0),
        )

        return InitializerOutput(
            gaussians=gaussians,
            features=None,
            depths=None,
        )
