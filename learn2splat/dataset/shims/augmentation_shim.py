import torch
from jaxtyping import Float
from torch import Tensor

from ..data_types import AnyExample, AnyViews


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    return {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
        "x_flipped": True,
    }


def mark_unflipped(views: AnyViews) -> AnyViews:
    """No-op augmentation that still records x_flipped=False, so the key is
    always present and downstream code (e.g. BatchedViews.from_dict) need not
    fall back to a default."""
    return {**views, "x_flipped": False}


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly horizontally-flip the scene (50% chance). Either way, x_flipped
    is recorded on every view set so the key is always present."""
    flip = torch.rand(tuple(), generator=generator) >= 0.5
    transform = reflect_views if flip else mark_unflipped
    out = {
        **example,
        "context": transform(example["context"]),
        "target": transform(example["target"]),
    }
    if "context_remain" in example:
        out["context_remain"] = transform(example["context_remain"])
    return out
