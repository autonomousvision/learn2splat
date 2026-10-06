# Adapted from https://github.com/nerfstudio-project/gsplat/blob/b5392febf6097655c18db17693636cd21bbe58c0/examples/datasets/colmap.py

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import imageio
import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import IterableDataset

from .colmap.utils import Parser
from .data_types import Stage
from .dataset import DatasetCfgCommon
from .shims.patch_shim import apply_patch_shim
from .view_sampler import ViewSampler
from .view_sampler.view_sampler_all import ViewSamplerAll
from .view_sampler.view_sampler_dense import ViewSamplerDense
from .view_sampler.view_sampler_evaluation import ViewSamplerEvaluation
from .view_sampler.view_sampler_ids import ViewSamplerIDs


@dataclass
class DatasetColmapCfg(DatasetCfgCommon):
    name: Literal["colmap"]
    roots: Path
    scene_name: Optional[str]  # If None, iterate over all scenes in roots
    normalize_world_space: bool
    subsample_factor: int
    # Per-scene image-downscale overrides, e.g. {bicycle: 4, room: 2}. Scenes not listed use
    # subsample_factor. Lets one test pass evaluate scenes at different resolutions, so the
    # per-scene runtimes stay comparable within a single process.
    subsample_factor_per_scene: dict[str, int] | None
    crop_size: None | int | list[int]
    symmetric_principal_point: bool = False  # override cx, cy to image center (matches 3DGS getProjectionMatrix)


class DatasetColmap(IterableDataset):
    cfg: DatasetColmapCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 100.0

    def __init__(
            self,
            cfg: DatasetColmapCfg,
            stage: Stage,
            view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        # COLMAP datasets should only be used for testing/validation, not training
        if stage == "train":
            raise ValueError(
                "COLMAP dataset does not support training stage. "
                "Use 'test' or 'val' stage instead. "
                "COLMAP scenes are typically small and meant for evaluation."
            )

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler

        # check if view_sampler is supported
        assert isinstance(self.view_sampler, (ViewSamplerDense, ViewSamplerIDs, ViewSamplerAll, ViewSamplerEvaluation)), \
            "COLMAP dataset requires ViewSamplerDense, ViewSamplerIDs, ViewSamplerAll, or ViewSamplerEvaluation."
        self.to_tensor = tf.ToTensor()

        # Discover available scenes
        if cfg.scene_name is not None:
            # Single scene mode (backward compatible)
            self.scene_names = [cfg.scene_name]
        else:
            # Multi-scene mode: list all subdirectories that contain COLMAP data
            self.scene_names = self._discover_scenes(cfg.roots)

        print(f"Found {len(self.scene_names)} scene(s) in {cfg.roots}: {self.scene_names}")

        # Image shape will be set when loading the first scene
        self.image_shape = None

    @staticmethod
    def _discover_scenes(roots: Path) -> List[str]:
        """Discover all valid COLMAP scenes in the roots directory."""
        scenes = []
        for subdir in sorted(roots.iterdir()):
            if subdir.is_dir():
                # Check if this looks like a COLMAP scene (has sparse folder or images folder)
                if (subdir / "sparse").exists() or (subdir / "images").exists():
                    scenes.append(subdir.name)
        return scenes

    def _load_scene(self, scene_name: str) -> dict:
        """Load a single scene and return it in chunk format."""
        colmap_root = self.cfg.roots / scene_name
        assert colmap_root.exists(), f"COLMAP root {colmap_root} does not exist."

        factor = (self.cfg.subsample_factor_per_scene or {}).get(scene_name, self.cfg.subsample_factor)
        print(f"Loading COLMAP scene '{scene_name}' from {colmap_root} with subsample factor {factor}")

        # Create parser for this scene
        print(f"in dataset NORMALIZE {self.cfg.normalize_world_space}")
        parser = Parser(
            data_dir=str(colmap_root),
            factor=factor,
            normalize=self.cfg.normalize_world_space,
        )
        print(f"parser scene scale {parser.scene_scale * 1.1}")

        # Update image shape from first loaded scene
        if self.image_shape is None:
            self.image_shape = [parser.height, parser.width]

        # Convert to chunk format
        return self._create_chunk_from_parser(parser, scene_name)

    def _create_chunk_from_parser(self, parser: Parser, scene_name: str) -> dict:
        """Convert COLMAP parser data to DL3DV-style chunk format."""

        # Collect all camera data (both context and target)
        all_indices = list(range(len(parser.image_names)))

        # Build cameras tensor (fx, fy, cx, cy, 4x4 w2c matrix)
        extrinsics_list = []
        intrinsics_list = []
        images_list = []

        for idx in all_indices:
            camera_id = parser.camera_ids[idx]

            # Get image dimensions
            w, h = parser.imsize_dict[camera_id]

            # Get camera intrinsics
            K = parser.Ks_dict[camera_id].copy()

            if self.cfg.symmetric_principal_point:
                K[0, 2] = w / 2.0
                K[1, 2] = h / 2.0

            # Normalize camera intrinsics
            K[0, :] /= w
            K[1, :] /= h

            # check if K is invertible
            if np.linalg.matrix_rank(K) < 3:
                print(K)
                raise ValueError(f"Camera intrinsic matrix for image {parser.image_names[idx]} is not invertible.")

            # Get camera-to-world matrix
            c2w = parser.camtoworlds[idx]

            # Pack
            extrinsics = torch.from_numpy(c2w).float()
            intrinsics = torch.from_numpy(K).float()

            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

            # Load image
            image = imageio.imread(parser.image_paths[idx])[..., :3]
            image = torch.from_numpy(image).permute(2, 0, 1)  # C, H, W

            images_list.append(image)  # list of C, H, W tensors

        extrinsics = torch.stack(extrinsics_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)

        chunk = {
            "key": scene_name,
            "cameras": (extrinsics, intrinsics),
            "images": images_list,
            "scene_scale": parser.scene_scale * 1.1
        }

        return chunk

    def _process_scene(self, chunk: dict):
        """Process a single scene chunk and yield examples."""
        extrinsics, intrinsics = chunk["cameras"]
        scene = chunk["key"]

        out_data = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
        )

        context_indices, target_indices = out_data[:2]

        c_list = [context_indices]
        t_list = [target_indices]
        for context_indices, target_indices in zip(c_list, t_list):
            # Load the images
            context_images = [
                chunk["images"][index.item()] for index in context_indices
            ]
            context_images = torch.stack(context_images).float() / 255.0

            target_images = [
                chunk["images"][index.item()] for index in target_indices
            ]
            target_images = torch.stack(target_images).float() / 255.0

            example_out = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)),
                    "far": self.get_bound("far", len(context_indices)),
                    "index": context_indices,
                    "scene_scale": chunk["scene_scale"],
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)),
                    "far": self.get_bound("far", len(target_indices)),
                    "index": target_indices,
                    "scene_scale": chunk["scene_scale"],
                },
                "scene": scene,
            }

            if self.cfg.crop_size is not None:
                example_out = apply_patch_shim(example_out, self.cfg.crop_size)

            yield example_out

    def __iter__(self):
        # Handle multiple workers - each worker should only process a subset of scenes
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            # Split scenes among workers
            scene_names = [
                scene_name
                for scene_index, scene_name in enumerate(self.scene_names)
                if scene_index % worker_info.num_workers == worker_info.id
            ]
        else:
            scene_names = self.scene_names

        # Iterate over assigned scenes
        test_scene_counter = 0
        for i, scene_name in enumerate(scene_names):
            # Skip scenes before test_start_idx (for scene-chunked SLURM jobs)
            if self.stage == "test" and test_scene_counter < self.cfg.test_start_idx:
                test_scene_counter += 1
                continue
            test_scene_counter += 1

            # Load the scene data
            chunk = self._load_scene(scene_name)
            # Process and yield examples from this scene
            yield from self._process_scene(chunk)

    def get_bound(
            self,
            bound: Literal["near", "far"],
            num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __len__(self) -> int:
        return len(self.scene_names)
