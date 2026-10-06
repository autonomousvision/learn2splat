from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData

from learn2splat.dataset.colmap.utils import Parser
from learn2splat.dataset.data_types import BatchedViews
from learn2splat.experimental.initializers_utils import nn_dist2_for_scales, points_to_gaussians
from learn2splat.misc.general_utils import SkipBatchException
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.scene_trainer.initializer.initializer import NonlearnedInitializer, InitializerOutput, NonlearnedInitializerCfg


@dataclass
class InitializerColmapCfg(NonlearnedInitializerCfg):
    name: Literal["colmap"]
    path: Path
    normalize_world_space: bool
    scaling_factor: float
    init_opacity: float
    sh_degree: int
    dl3dv_settings: bool
    filter_zero_rgb: bool
    randomize_opacity: bool
    randomize_opacity_distribution: Literal["uniform", "gaussian"]
    randomize_opacity_std: float  # Standard deviation for gaussian distribution
    randomize_opacity_min: float  # Minimum value for uniform distribution
    points3d_subdir: Optional[str]  # if set, overrides dl3dv_settings/default subdir logic
    points3d_ply_filename: Optional[str]  # if set, loads points from this PLY file (relative to scene dir) instead of COLMAP binary
    override_dataset_poses: bool  # if true, overrides the dataset poses with the COLMAP poses (after applying T_world transform)

    def get_sh_d(self):
        sh_d = (self.sh_degree + 1) ** 2
        return sh_d


class InitializerColmap(NonlearnedInitializer[InitializerColmapCfg]):
    def __init__(self, cfg: InitializerColmapCfg) -> None:
        super().__init__(cfg)

    def _npz_path(self, datadir: Path) -> Path:
        suffix = "_norm" if self.cfg.normalize_world_space else ""
        if self.cfg.points3d_ply_filename is not None:
            ply_stem = Path(self.cfg.points3d_ply_filename).stem
            return datadir / f"colmap_points_cache_ply_{ply_stem}{suffix}.npz"
        return datadir / f"colmap_points_cache{suffix}.npz"

    def _load_colmap(self, datadir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load COLMAP points/colors/poses.

        On first access, parses the raw COLMAP binary files (or a PLY file when
        ``points3d_ply_filename`` is set) and saves a compact .npz next to the
        scene folder.  On subsequent calls only the tiny .npz is loaded.
        """
        npz_path = self._npz_path(datadir)
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                return data["points"], data["points_rgb"], data["camtoworlds"]
            except PermissionError:
                print(f"Warning: No read permission for cache {npz_path}. Attempting to delete and regenerate.")
                try:
                    os.unlink(npz_path)
                except Exception as del_e:
                    print(f"Warning: Could not delete {npz_path} ({del_e}). Will re-parse but cannot cache.")
            except Exception as e:
                print(f"Warning: Failed to load cache {npz_path} ({e}). Re-parsing COLMAP data.")

        # Always parse COLMAP cameras/images for the poses.
        parser = Parser(
            data_dir=str(datadir),
            factor=1,
            normalize=self.cfg.normalize_world_space,
            load_images=False,
            dl3dv_settings=False,
            points3d_subdir=self.cfg.points3d_subdir,
            verbose=False,
        )
        camtoworlds = parser.camtoworlds    # (M, 4, 4) float64

        if self.cfg.points3d_ply_filename is not None:
            # Load 3-D points from a PLY file located directly in the scene dir.
            ply_path = datadir / self.cfg.points3d_ply_filename
            if not ply_path.exists():
                raise IOError(f"PLY file not found: {ply_path}")
            plydata = PlyData.read(str(ply_path))
            vertex = plydata["vertex"]
            points = np.stack([
                np.asarray(vertex["x"]),
                np.asarray(vertex["y"]),
                np.asarray(vertex["z"]),
            ], axis=1).astype(np.float32)
            points_rgb = np.stack([
                np.asarray(vertex["red"]),
                np.asarray(vertex["green"]),
                np.asarray(vertex["blue"]),
            ], axis=1).astype(np.uint8)
        else:
            points = parser.points          # (N, 3) float32
            points_rgb = parser.points_rgb  # (N, 3) uint8

        # Write atomically with a temp file that already ends in .npz.
        try:
            tmp_path = ''
            tmp_fd, tmp_path = tempfile.mkstemp(dir=datadir, suffix=".npz")
            os.close(tmp_fd)
            np.savez_compressed(tmp_path, points=points, points_rgb=points_rgb, camtoworlds=camtoworlds)
            os.chmod(tmp_path, 0o664)  # group-readable so other users can use this cache
            os.replace(tmp_path, npz_path)  # atomic on POSIX
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            print(f"Warning: Failed to save COLMAP cache to {npz_path}. This may cause slow loading in the future.")
        return points, points_rgb, camtoworlds

    def forward(
            self,
            context: BatchedViews,
            visualization_dump: Optional[dict] = None,
            device: Optional[torch.device] = None,
            **kwargs
    ) -> InitializerOutput:
        verbose = False

        # context not used

        # assert COLMAP dir exists
        if not self.cfg.path.exists():
            raise ValueError(f"COLMAP dir {self.cfg.path} does not exist.")

        if "scene" in kwargs:
            scene_name = kwargs["scene"]
            assert len(scene_name) == 1, f"Only single scene initialization supported. {scene_name}"
            scene_name = scene_name[0]
            if self.cfg.dl3dv_settings:
                scene_name = scene_name.replace("dl3dv_", "")
            if verbose:
                print(f"Initializing scene '{scene_name}' from COLMAP at {self.cfg.path}.")
            datadir = self.cfg.path / scene_name
            if not datadir.exists():
                raise ValueError(f"COLMAP scene dir {datadir} does not exist.")
        else:
            datadir = self.cfg.path

        # run COLMAP parser (cached after first load)
        points_xyz, points_rgb, camtoworlds = self._load_colmap(datadir)

        if verbose:
            print(f"Loaded {points_xyz.shape[0]} points from COLMAP.")

        xyz = torch.from_numpy(points_xyz).float().to(device)
        rgbs = torch.from_numpy(points_rgb / 255.0).float().to(device)

        if self.cfg.filter_zero_rgb:
            # Filter out points with 0,0,0 RGB values (these are often outliers in COLMAP reconstructions)
            valid_mask = (rgbs.sum(dim=-1) > 0)
            xyz = xyz[valid_mask]
            rgbs = rgbs[valid_mask]

        if self.cfg.dl3dv_settings:
            assert "target" in kwargs, "Target key is required in kwargs for COLMAP initializer with dl3dv format."
            target = kwargs["target"]

            # In some configration we might move the batch to device later, so we want to keep the device consistent
            batch_device = target['extrinsics'].device

            context_c2w_dataset = context['extrinsics']  # (b, V, 4, 4)
            c2w_colmap = torch.from_numpy(camtoworlds).to(device=batch_device,
                                                                 dtype=context_c2w_dataset.dtype)  # (N, 4, 4)
            # T_world = c2w_dataset[0] @ c2w_colmap[0].inverse()
            # eps = 1e-3
            # T_world[T_world.abs() < eps] = 0
            # T_world[(T_world - 1.0).abs() < eps] = 1.0
            # T_world[(T_world + 1.0).abs() < eps] = -1.0
            T_world = torch.tensor([[0., 1., 0., 0.],
                                    [1., 0., 0., 0.],
                                    [0., 0., -1., 0.],
                                    [0., 0., 0., 1.]], device=batch_device,
                                   dtype=context_c2w_dataset.dtype)  # hard coded for dl3dv colmap reconstructions
            c2w_dataset_predicted = T_world @ c2w_colmap

            # Assume only one scene in the batch
            context_x_flipped = context['x_flipped'][0]
            target_x_flipped = target['x_flipped'][0]
            assert context_x_flipped == target_x_flipped, "Context and target x_flipped values must match."
            x_flipped = context_x_flipped
            flip_transform = torch.eye(4, device=batch_device, dtype=context_c2w_dataset.dtype)
            flip_transform[0, 0] = -1.0

            if x_flipped:
                c2w_dataset_predicted = flip_transform @ c2w_dataset_predicted @ flip_transform

            # Overriding the dataset poses with the COLMAP to ensure consistency
            if self.cfg.override_dataset_poses:
                context_indices = context['index'][0]
                new_context_c2w = c2w_dataset_predicted[context_indices]
                new_context_c2w = new_context_c2w[None, ...]  # (1, V, 4, 4)
                context['extrinsics'] = new_context_c2w

                target_indices = target['index'][0]
                new_target_c2w = c2w_dataset_predicted[target_indices]
                new_target_c2w = new_target_c2w[None, ...]
                target['extrinsics'] = new_target_c2w

            xyz = xyz.to(device)
            xyz = T_world.to(device) @ torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1).T
            if x_flipped:
                xyz[0] *= -1.0
            xyz = xyz[:3, :].T

        # ── Step 1+2: subsample the point cloud ("sample Gaussians") ─────────────
        # Optional subsampling augmentation, then down to the fixed count for DDP consistency.
        # Done before knn so the neighbour distances are computed on the kept points.
        xyz, rgbs = self.subsample_points(xyz, rgbs)

        if xyz.shape[0] == 0:
            black_gaussians_num = (points_rgb == 0).all(axis=-1).sum()
            raise SkipBatchException(f"No valid points found in COLMAP data for scene {datadir}. Skipping batch. "
                                     f"Originally {points_xyz.shape[0]} points. Black gaussian num {black_gaussians_num}.")

        # ── Step 3: knn-based scale initialisation ───────────────────────────────
        dist2_avg = nn_dist2_for_scales(xyz)  # [N]
        dist_avg = torch.sqrt(dist2_avg)
        scales = dist_avg.unsqueeze(-1).repeat(1, 3)  # [N, 3]

        # Initialize opacities with optional randomization
        if self.cfg.randomize_opacity:
            if self.cfg.randomize_opacity_distribution == "uniform":
                # Randomize opacities uniformly between min and max
                opacities = (torch.rand(xyz.shape[0], device=xyz.device) * (self.cfg.init_opacity - self.cfg.randomize_opacity_min)) + self.cfg.randomize_opacity_min
            elif self.cfg.randomize_opacity_distribution == "gaussian":
                # Randomize opacities with a Gaussian distribution
                mean = self.cfg.init_opacity
                stddev = self.cfg.randomize_opacity_std
                opacities = torch.normal(mean, stddev, size=(xyz.shape[0],), device=xyz.device)
                opacities = opacities.clamp(0, 1)  # Clamp to ensure valid values
            else:
                raise ValueError(f"Unknown randomize_opacity_distribution: {self.cfg.randomize_opacity_distribution}")
        else:
            opacities = torch.full((xyz.shape[0],), self.cfg.init_opacity)

        nr_valid = xyz.shape[0]
        # ── Step 4: pad to fixed count for DDP consistency ───────────────────────
        fixed_num = self.fixed_gaussians_num
        if fixed_num is not None and xyz.shape[0] < fixed_num:
            pad = fixed_num - xyz.shape[0]
            xyz = F.pad(xyz, (0, 0, 0, pad), value=0.0)
            rgbs = F.pad(rgbs, (0, 0, 0, pad), value=0.0)
            scales = F.pad(scales, (0, 0, 0, pad), value=1e-10)
            opacities = F.pad(opacities, (0, pad), value=1e-10)
            # NOTE: These are invisible (zero-opacity) dummy Gaussians, added only so that every GPU optimizes
            # the same number of Gaussians; nr_valid (set below) is the true count. They must stay
            # invisible: if the optimizer updates them they gain opacity and become real Gaussians. The
            # knn optimizer keeps them frozen by updating only Gaussians with nonzero gradient (the dummies
            # are invisible, so their gradient is zero) — see KnnBasedOptimizer._forward_impl.

        points_dict = {
            "xyz": xyz,
            "rgb": rgbs,
            "scales": scales,
            "opacities": opacities,
        }

        points_dict["scales"] *= self.cfg.scaling_factor

        # pre-activation values on device
        gaussians_dict = points_to_gaussians(points_dict, sh_degree=self.cfg.sh_degree, device=device)

        means = gaussians_dict["xyz"]
        sh0 = gaussians_dict["sh0"]
        shN = gaussians_dict["shN"]
        if shN is not None:
            harmonics = torch.cat([sh0, shN], dim=1)  # [N, sh_d, 3]
        else:
            harmonics = sh0
        harmonics = harmonics.permute(0, 2, 1)  # [N, 3, sh_d]
        rotations_unnorm = gaussians_dict["rotations_unnorm"]

        # post-activation values
        opacities = torch.sigmoid(gaussians_dict["opacities_raw"])
        scales = torch.exp(gaussians_dict["scales_raw"])
        rotations = F.normalize(gaussians_dict["rotations_unnorm"], dim=-1)
        covariances = build_covariance(scale=scales, rotation_xyzw=rotations)

        gaussians = Gaussians(
            means=means.unsqueeze(0),
            covariances=covariances.unsqueeze(0),
            harmonics=harmonics.unsqueeze(0),  # [1, N, C, sh_d]
            opacities=opacities.unsqueeze(0),
            scales=scales.unsqueeze(0),
            rotations=rotations.unsqueeze(0),
            rotations_unnorm=rotations_unnorm.unsqueeze(0),
            nr_valid=nr_valid
        )

        return InitializerOutput(
            gaussians=gaussians,
            features=None,
            depths=None
        )
