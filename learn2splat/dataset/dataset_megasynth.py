from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Optional

import numpy as np
import torch
from jaxtyping import Bool, Float, UInt8
from torch import Tensor

from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg


@dataclass
class DatasetMegasynthCfg(DatasetDL3DVCfg):
    name: Literal["megasynth"]
    # MegaSynth chunks carry per-view float16 depth (.npy bytes). Opt in to attach
    # it to the context/target batch dicts under the "depth" key.
    load_depth: bool = False


class DatasetMegasynth(DatasetDL3DV):
    """Reader for MegaSynth `.torch` chunks (learn2splat/scripts/convert_megasynth.py).

    The chunk schema is identical to DL3DV (per-example `key`, `cameras` [V,18] OpenCV-w2c,
    `images` as RGB byte tensors), so this reuses DatasetDL3DV wholesale. The only addition
    is per-view depth: each example also holds `depths` (float16 HxW `.npy` bytes); when
    `cfg.load_depth` is set, the sampled context/target depths are decoded and attached.
    """

    cfg: DatasetMegasynthCfg

    def _process_example_to_batch(
        self,
        example: dict,
        extrinsics: Tensor,
        intrinsics: Tensor,
        context_indices: Tensor,
        target_indices: Tensor,
    ) -> Optional[dict]:
        example_out = super()._process_example_to_batch(
            example, extrinsics, intrinsics, context_indices, target_indices
        )
        if example_out is not None and self.cfg.load_depth:
            try:
                for tag, indices in (("context", context_indices), ("target", target_indices)):
                    depth = self.convert_depths([example["depths"][i.item()] for i in indices])
                    example_out[tag]["depth"] = depth
                    # Construct + cache the valid-depth mask next to the depth (see BatchedViews).
                    example_out[tag]["depth_mask"] = self.depth_validity_mask(depth)
            except (OSError, KeyError):
                return None
        return example_out

    def convert_depths(
        self,
        depths: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "view height width"]:
        """Decode per-view float16 `.npy` depth bytes into a [V, H, W] float tensor.

        Invalid/background pixels are 0.0 (set by the converter); mask with `depth > 0`.
        """
        return torch.stack([
            torch.from_numpy(np.load(BytesIO(depth.numpy().tobytes())).astype(np.float32))
            for depth in depths
        ])

    @staticmethod
    def depth_validity_mask(depth: Float[Tensor, "view height width"]) -> Bool[Tensor, "view height width"]:
        """Valid-depth mask cached next to the depth (see BatchedViews.depth_mask). The MegaSynth
        converter writes 0.0 for invalid/background pixels, so valid = depth > 0. Datasets with a
        different invalid-depth convention should override this with their own masking strategy."""
        return depth > 0
