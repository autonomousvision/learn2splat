import argparse
import json
import os
from collections import OrderedDict
from glob import glob

import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from jaxtyping import install_import_hook
from torch import Tensor
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
        ("learn2splat",),
        ("beartype", "beartype"),
):
    from learn2splat.dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample
    from learn2splat.paths import asset_path


def convert_poses(
        poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def partition_list(lst, n_bins):
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0")
    if len(lst) < n_bins:
        raise ValueError("Number of bins cannot exceed the length of the list")

    bin_size = len(lst) // n_bins
    borders = [lst[0]]  # First border is always the first index
    for i in range(1, n_bins):
        border_index = min(
            i * bin_size, len(lst) - 1
        )  # Ensure last bin doesn't exceed list length
        borders.append(lst[border_index])
    borders.append(lst[-1])  # Last border is always the last index
    return borders


def find_train_and_test_index(chunk_path, scene_name=None, num_context_views=5,
                              num_target_skip=1, num_target_views=28,
                              start_frame=None,
                              frame_distance=None,
                              render_video=False,
                              uniform_sample=False,
                              ):
    chunk = torch.load(chunk_path)
    out_dict = OrderedDict()
    for example in chunk:
        cur_scene_name = example["key"]
        if scene_name is not None and cur_scene_name != scene_name:
            continue

        extrinsics, intrinsics = convert_poses(example["cameras"])

        # bounded evaluation to make the task easier
        if start_frame is not None:
            assert frame_distance is not None
            end_frame = start_frame + frame_distance

            extrinsics = extrinsics[start_frame:end_frame]

        n_views = extrinsics.shape[0]

        if uniform_sample:
            index_context = [int(x) for x in np.linspace(0, n_views, num_context_views, dtype=int)]
        else:
            index_context = sorted(farthest_point_sample(
                extrinsics[:, :3, -1].unsqueeze(0), num_context_views
            ).squeeze(0).tolist())

        if render_video:
            assert start_frame is not None
            assert frame_distance is not None
            index_target = list(range(start_frame, end_frame))
        else:
            index_target_all = [x for x in range(n_views) if x not in index_context]

            if uniform_sample:
                index_target_select = [(index_context[i] + index_context[i + 1]) // 2 for i in
                                       range(len(index_context) - 1)]
            else:
                index_target_select = partition_list(index_target_all, num_target_views)

            if start_frame is not None:
                # the original index in the full sequence
                index_context = [idx + start_frame for idx in index_context]
                index_target_select = [idx + start_frame for idx in index_target_select]

            assert (
                    len(index_target_select) >= num_target_views
            ), f"double check {cur_scene_name} at {chunk_path}: target len: {len(index_target_select)} from {len(index_target_all)}"
            index_target = index_target_select[:num_target_views]

        out_dict[cur_scene_name] = {"context": index_context, "target": index_target}

    return out_dict


def generate_index_file(args):
    n_ctx = args.num_context_views
    if args.uniform_sample:
        args.num_target_views = n_ctx - 1
    n_tgt = args.num_target_views

    out_dir = str(asset_path("dl3dv_evaluation"))
    os.makedirs(out_dir, exist_ok=True)
    data_dir = "datasets/dl3dv/test"
    chunk_paths = sorted(glob(os.path.join(data_dir, "*.torch")))
    out_dict_all = OrderedDict()
    for chunk_path in tqdm(chunk_paths):
        out_dict = find_train_and_test_index(
            chunk_path, scene_name=None, num_context_views=n_ctx,
            num_target_views=n_tgt,
            start_frame=args.start_frame,
            frame_distance=args.frame_distance,
            render_video=args.render_video,
            uniform_sample=args.uniform_sample,
        )
        out_dict_all.update(out_dict)

    if args.start_frame is not None:
        if args.render_video:
            save_file = f"dl3dv_start_{args.start_frame}_distance_{args.frame_distance}_ctx_{n_ctx}v_video.json"
        else:
            save_file = f"dl3dv_start_{args.start_frame}_distance_{args.frame_distance}_ctx_{n_ctx}v_tgt_{n_tgt}v.json"
    else:
        save_file = f"dl3dv_ctx_{n_ctx}v_tgt_{n_tgt}v.json"

    if args.uniform_sample:
        save_file = save_file[:-5] + '_uniform.json'

    out_path = os.path.join(out_dir, save_file)

    with open(out_path, "w") as f:
        json.dump(out_dict_all, f)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_target_views", type=int, default=28, help="test skip")
    parser.add_argument("--num_context_views", type=int, default=5, help="test skip")
    parser.add_argument('--render_video', action='store_true')

    # bounded evaluation to make the task easier
    parser.add_argument('--start_frame', default=None, type=int)
    parser.add_argument('--frame_distance', default=None, type=int)
    parser.add_argument('--uniform_sample', action='store_true')

    args = parser.parse_args()

    generate_index_file(args)
