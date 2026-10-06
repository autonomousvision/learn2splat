from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

import torch
from jaxtyping import Float, Int64, Bool
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).

class BatchedViewsDict(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch view 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch view 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch view channel height width"]  # batch view channel height width
    near: Float[Tensor, "batch view"]  # batch view
    far: Float[Tensor, "batch view"]  # batch view
    index: Int64[Tensor, "batch view"]  # batch view
    scene_scale: Float[Tensor, "batch"]  # batch


@dataclass
class BatchedViews:
    """
    BatchedViews represents a batch of views. The batch dimension allows for efficient processing of multiple
    scenes simultaneously.

    This class is a wrapper around the BatchedViewsDict TypedDict.
    Some dict like behavior is still missing, can be added as needed.
    """
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    scene_scale: Float[Tensor, "batch "] | None

    x_flipped: Bool[Tensor, "batch"]

    viewpoint_stack: Int64[Tensor, "batch _"] | None = None  # batch subset
    used_indices_list: list[Int64[Tensor, "_"]] | None = None  # list of tensors of shape (subset,)
    depth: Float[Tensor, "batch _ _ _"] | None = None  # batch view height width (optional GT depth)
    # Dataset-defined valid-depth mask, constructed and cached next to `depth` (each dataset chooses
    # its own masking strategy, e.g. MegaSynth uses depth > 0). None when the dataset provides none.
    depth_mask: Bool[Tensor, "batch _ _ _"] | None = None  # batch view height width

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def batchify_views(self, indices: Int64[Tensor, "batch _"]) -> "BatchedViews":
        """Select a subset of views for each example in the batch."""

        scene_batch = indices.size(0)
        scene_batch_idx = torch.arange(scene_batch, device=indices.device)[:, None]

        return BatchedViews(
            extrinsics=self.extrinsics[scene_batch_idx, indices],
            intrinsics=self.intrinsics[scene_batch_idx, indices],
            image=self.image[scene_batch_idx, indices],
            near=self.near[scene_batch_idx, indices],
            far=self.far[scene_batch_idx, indices],
            index=self.index[scene_batch_idx, indices],
            x_flipped=self.x_flipped,
            scene_scale=self.scene_scale,
            depth=self.depth[scene_batch_idx, indices] if self.depth is not None else None,
            depth_mask=self.depth_mask[scene_batch_idx, indices] if self.depth_mask is not None else None,
        )

    @classmethod
    def from_dict(cls, data: BatchedViewsDict) -> "BatchedViews":
        b = data["extrinsics"].size(0)
        device = data["extrinsics"].device
        return cls(
            extrinsics=data["extrinsics"],
            intrinsics=data["intrinsics"],
            image=data["image"],
            near=data["near"],
            far=data["far"],
            index=data["index"],
            x_flipped=data.get("x_flipped", torch.zeros(b, dtype=torch.bool, device=device)),
            viewpoint_stack=data.get("viewpoint_stack", None),
            used_indices_list=data.get("used_indices_list", None),
            scene_scale=data.get("scene_scale", None),
            depth=data.get("depth", None),
            depth_mask=data.get("depth_mask", None),
        )

    def reset_viewpoint_stack_if_needed(self, strategy, batch_size) -> None:

        if self.viewpoint_stack is None or self.viewpoint_stack.size(1) < batch_size:
            # Create a new viewpoint stack
            batch = self.extrinsics.size(0)
            num_views = self.extrinsics.size(1)

            if strategy == "random":
                # Create a random permutation of viewpoints for each example in the batch, by sorting random values
                rand_matrix = torch.rand(batch, num_views, device=self.extrinsics.device)
                new_stack = torch.argsort(rand_matrix, dim=1)
            else:
                new_stack = torch.arange(num_views, device=self.extrinsics.device).unsqueeze(0).expand(batch,
                                                                                                       -1).clone()

            # Assign the new viewpoint stack
            if self.viewpoint_stack is None or self.viewpoint_stack.size(1) == 0:
                self.viewpoint_stack = new_stack
            else:
                # Concatenate the existing stack with the new one
                self.viewpoint_stack = torch.cat([self.viewpoint_stack, new_stack], dim=1)


class BatchedExample(TypedDict, total=False):
    target: BatchedViews | BatchedViewsDict
    context: BatchedViews | BatchedViewsDict
    scene: list[str]
    source: list[str]  # mixed-dataset source key per scene; absent for single datasets


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str
    source: str  # mixed-dataset source key; absent for single datasets


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
