from dataclasses import asdict

from ..data_types import BatchedExample, BatchedViews, UnbatchedViews, BatchedViewsDict, UnbatchedExample


def apply_patch_shim_to_views(views: BatchedViews | UnbatchedViews | BatchedViewsDict,
                              patch_size: int | list[int]) -> BatchedViews | UnbatchedViews | BatchedViewsDict:
    *_, h, w = views["image"].shape

    if isinstance(patch_size, int):
        patch_size_x = patch_size
        patch_size_y = patch_size
    else:
        patch_size_x, patch_size_y = patch_size

    h_new = (h // patch_size_x) * patch_size_x
    row = (h - h_new) // 2
    w_new = (w // patch_size_y) * patch_size_y
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][..., row: row + h_new, col: col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[..., 0, 0] *= w / w_new  # fx
    intrinsics[..., 1, 1] *= h / h_new  # fy

    if isinstance(views, BatchedViews):
        return BatchedViews.from_dict({
            **asdict(views),
            "image": image,
            "intrinsics": intrinsics,
        })
    else:
        return {
            **views,
            "image": image,
            "intrinsics": intrinsics,
        }


def apply_patch_shim(batch: BatchedExample | UnbatchedExample,
                     patch_size: int | list[int]) -> BatchedExample | UnbatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }
