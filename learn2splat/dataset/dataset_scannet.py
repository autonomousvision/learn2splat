import json
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

from .data_types import Stage
from .dataset import DatasetCfgCommon
from .shims.patch_shim import apply_patch_shim
from .view_sampler import ViewSampler
from .view_sampler.view_sampler_all import ViewSamplerAll
from .view_sampler.view_sampler_dense import ViewSamplerDense
from .view_sampler.view_sampler_ids import ViewSamplerIDs


# OpenGL to OpenCV conversion: flip Y and Z axes
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


@dataclass
class DatasetScannetCfg(DatasetCfgCommon):
    name: Literal["scannet"]
    roots: Path
    scene_name: Optional[str]  # If None, iterate over all scenes from split
    split: str  # "test", "val", "train", "test_debug" -> splits/{split}_scene_ids.txt
    subsample_factor: int
    crop_size: None | int | list[int]
    num_context_views: int  # Max context views to select via FPS
    filter_bad_frames: bool


class DatasetScannet(IterableDataset):
    cfg: DatasetScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 100.0

    def __init__(
            self,
            cfg: DatasetScannetCfg,
            stage: Stage,
            view_sampler: ViewSampler,
    ) -> None:
        super().__init__()

        if stage == "train":
            raise ValueError(
                "ScanNet dataset does not support training stage. "
                "Use 'test' or 'val' stage instead."
            )

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler

        assert isinstance(self.view_sampler, (ViewSamplerDense, ViewSamplerIDs, ViewSamplerAll)), \
            "ScanNet dataset requires ViewSamplerDense, ViewSamplerIDs, or ViewSamplerAll."
        self.to_tensor = tf.ToTensor()

        # Discover available scenes
        if cfg.scene_name is not None:
            self.scene_names = [cfg.scene_name]
        else:
            self.scene_names = self._discover_scenes()

        print(f"Found {len(self.scene_names)} scene(s) for split '{cfg.split}': {self.scene_names}")

        self.image_shape = None

    @staticmethod
    def _read_split_file(roots: Path, split: str) -> List[str]:
        """Read scene IDs from a split file."""
        split_path = roots / "splits" / f"{split}_scene_ids.txt"
        with open(split_path) as f:
            return [line.strip() for line in f if line.strip()]

    def _discover_scenes(self) -> List[str]:
        """Discover valid scenes: read split file, filter to scenes that exist in data/."""
        scene_ids = self._read_split_file(self.cfg.roots, self.cfg.split)
        data_dir = self.cfg.roots / "data"
        valid = [s for s in scene_ids if (data_dir / s).exists()]
        if len(valid) < len(scene_ids):
            print(f"Warning: {len(scene_ids) - len(valid)} scenes from split not found in data/")
        return valid

    @staticmethod
    def _fps_select(positions: np.ndarray, num_select: int) -> np.ndarray:
        """Furthest point sampling on 3D camera positions.

        Greedily selects points that maximize the minimum distance to
        the already-selected set, starting from the first point.

        Args:
            positions: [N, 3] array of camera positions.
            num_select: Number of points to select.

        Returns:
            [num_select] array of selected indices.
        """
        n = len(positions)
        if num_select >= n:
            return np.arange(n)

        selected = [0]
        min_dists = np.full(n, np.inf)

        for _ in range(num_select - 1):
            last = positions[selected[-1]]
            dists = np.linalg.norm(positions - last, axis=1)
            min_dists = np.minimum(min_dists, dists)
            min_dists[selected] = -1  # exclude already selected
            selected.append(int(np.argmax(min_dists)))

        return np.array(selected)

    def _parse_frames(
            self,
            frames: list[dict],
            scene_dir: Path,
            w: int,
            h: int,
            fl_x: float,
            fl_y: float,
            cx: float,
            cy: float,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Parse a list of frames into extrinsics, intrinsics, and images.

        Returns:
            extrinsics_list: list of [4, 4] tensors (c2w in OpenCV convention)
            intrinsics_list: list of [3, 3] tensors (normalized)
            images_list: list of [C, H, W] tensors (uint8)
        """
        # Build normalized intrinsic matrix (same for all frames in a scene)
        K = np.array([
            [fl_x / w, 0.0, cx / w],
            [0.0, fl_y / h, cy / h],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        intrinsics_tensor = torch.from_numpy(K)

        extrinsics_list = []
        intrinsics_list = []
        images_list = []

        for frame in frames:
            if self.cfg.filter_bad_frames and frame.get("is_bad", False):
                continue

            # Parse c2w and convert OpenGL -> OpenCV
            c2w_gl = np.array(frame["transform_matrix"], dtype=np.float32)
            c2w_cv = c2w_gl @ _GL_TO_CV

            extrinsics_list.append(torch.from_numpy(c2w_cv))
            intrinsics_list.append(intrinsics_tensor.clone())

            # Load image
            img_path = scene_dir / "images" / frame["file_path"]
            image = imageio.imread(str(img_path))[..., :3]

            # Subsample if needed
            if self.cfg.subsample_factor > 1:
                factor = self.cfg.subsample_factor
                image = image[::factor, ::factor]

            image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
            images_list.append(image)

        return extrinsics_list, intrinsics_list, images_list

    def _load_scene(self, scene_name: str) -> dict:
        """Load a single scene and return it in chunk format."""
        scene_dir = self.cfg.roots / "data" / scene_name
        assert scene_dir.exists(), f"Scene directory {scene_dir} does not exist."

        print(f"Loading ScanNet scene '{scene_name}' from {scene_dir}")

        # Load transforms.json
        transforms_path = scene_dir / "transforms.json"
        with open(transforms_path) as f:
            transforms = json.load(f)

        w, h = transforms["w"], transforms["h"]
        fl_x, fl_y = transforms["fl_x"], transforms["fl_y"]
        cx, cy = transforms["cx"], transforms["cy"]

        train_frames = transforms["frames"]
        test_frames = transforms.get("test_frames", [])

        # Filter bad frames before FPS (to get correct positions)
        if self.cfg.filter_bad_frames:
            train_frames_valid = [f for f in train_frames if not f.get("is_bad", False)]
        else:
            train_frames_valid = train_frames

        # FPS on camera positions to select context views
        if len(train_frames_valid) > self.cfg.num_context_views:
            positions = np.array([
                np.array(f["transform_matrix"], dtype=np.float32)[:3, 3]
                for f in train_frames_valid
            ])
            fps_indices = self._fps_select(positions, self.cfg.num_context_views)
            selected_train_frames = [train_frames_valid[i] for i in fps_indices]
            print(f"  FPS selected {len(selected_train_frames)}/{len(train_frames_valid)} context views")
        else:
            selected_train_frames = train_frames_valid
            print(f"  Using all {len(selected_train_frames)} context views (< {self.cfg.num_context_views})")

        # Parse context frames (selected training frames)
        ctx_ext, ctx_int, ctx_imgs = self._parse_frames(
            selected_train_frames, scene_dir, w, h, fl_x, fl_y, cx, cy
        )

        # Parse target frames (test frames)
        tgt_ext, tgt_int, tgt_imgs = self._parse_frames(
            test_frames, scene_dir, w, h, fl_x, fl_y, cx, cy
        )

        context_end_idx = len(ctx_ext)
        all_ext = ctx_ext + tgt_ext
        all_int = ctx_int + tgt_int
        all_imgs = ctx_imgs + tgt_imgs

        extrinsics = torch.stack(all_ext, dim=0)
        intrinsics = torch.stack(all_int, dim=0)

        if self.image_shape is None and len(all_imgs) > 0:
            self.image_shape = [all_imgs[0].shape[1], all_imgs[0].shape[2]]

        print(f"  Loaded {context_end_idx} context + {len(tgt_ext)} target views")

        return {
            "key": scene_name,
            "cameras": (extrinsics, intrinsics),
            "images": all_imgs,
            "context_end_idx": context_end_idx,
        }

    def _process_scene(self, chunk: dict):
        """Process a single scene chunk and yield examples."""
        extrinsics, intrinsics = chunk["cameras"]
        scene = chunk["key"]

        # Delegate to view sampler to determine context/target split
        context_indices, target_indices = self.view_sampler.sample(
            scene, extrinsics, intrinsics,
        )

        # Assert no overlap between context and target views
        context_set = set(context_indices.tolist())
        target_set = set(target_indices.tolist())
        overlap = context_set & target_set
        assert len(overlap) == 0, (
            f"Scene '{scene}': {len(overlap)} target views leaked into context: {overlap}"
        )

        # Load and normalize images
        context_images = torch.stack(
            [chunk["images"][i.item()] for i in context_indices]
        ).float() / 255.0

        target_images = torch.stack(
            [chunk["images"][i.item()] for i in target_indices]
        ).float() / 255.0

        example_out = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
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
            scene_names = [
                scene_name
                for scene_index, scene_name in enumerate(self.scene_names)
                if scene_index % worker_info.num_workers == worker_info.id
            ]
        else:
            scene_names = self.scene_names

        for scene_name in scene_names:
            chunk = self._load_scene(scene_name)
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
