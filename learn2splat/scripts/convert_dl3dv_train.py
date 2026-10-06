import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
from tqdm import tqdm

from learn2splat.scripts.convert_dl3dv_utils import Example, get_size, load_images, load_metadata, is_image_shape_matched

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="original dataset directory")
parser.add_argument("--output_dir", type=str, help="processed dataset directory")
parser.add_argument(
    "--img_subdir",
    type=str,
    default="images_8",
    help="image directory name",
    choices=[
        "images_4",
        "images_8",
    ],
)
parser.add_argument("--n_test", type=int, default=10, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")

args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(2e8)


def legal_check_for_all_scenes(root_dir, target_shape):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/*")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        # img_dir = os.path.join(sub_folder, 'images_8')
        img_dir = os.path.join(sub_folder, "images_4")
        if not is_image_shape_matched(Path(img_dir), target_shape):
            print(f"image shape does not match for {sub_folder}")
            continue
        pose_file = os.path.join(sub_folder, "transforms.json")
        if not os.path.isfile(pose_file):
            print(f"cannot find pose file for {sub_folder}")
            continue

        valid_folders.append(sub_folder)

    return valid_folders


if __name__ == "__main__":
    if "images_8" in args.img_subdir:
        target_shape = (270, 480)  # (h, w)
    elif "images_4" in args.img_subdir:
        target_shape = (540, 960)
    else:
        raise ValueError

    print("checking all scenes...")
    valid_scenes = legal_check_for_all_scenes(INPUT_DIR, target_shape)
    print("valid scenes:", len(valid_scenes))

    # test scenes
    test_scenes = "your_test_set_index.json"
    with open(test_scenes, "r") as f:
        overlap_scenes = json.load(f)

    assert len(overlap_scenes) == 140, "test scenes should contain 140 scenes"

    for stage in ["train"]:

        error_logs = []
        image_dirs = valid_scenes

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []


        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []


        for image_dir in tqdm(image_dirs, desc=f"Processing {stage}"):
            key = os.path.basename(image_dir.strip("/"))
            # skip test scenes
            if key in overlap_scenes:
                print(f"scene {key} in benchmark, skip.")
                continue

            image_dir = Path(image_dir) / "images_8"  # 270x480
            # image_dir = Path(image_dir) / 'images_4'  # 540x960

            num_bytes = get_size(image_dir)

            # Read images and metadata.
            try:
                images = load_images(image_dir)
            except:
                print("image loading error")
                continue
            meta_path = image_dir.parent / "transforms.json"
            if not meta_path.is_file():
                error_msg = f"---------> [ERROR] no meta file in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            example = load_metadata(meta_path)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
            except:
                error_msg = f"---------> [ERROR] Some images missing in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue

            # Add the key to the example.
            example["key"] = "dl3dv_" + key

            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()
