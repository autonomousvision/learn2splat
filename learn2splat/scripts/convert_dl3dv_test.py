import argparse
import os
from glob import glob
from pathlib import Path

import torch
from tqdm import tqdm

from learn2splat.scripts.convert_dl3dv_utils import is_image_shape_matched, Example, get_size, load_images, load_metadata

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
parser.add_argument("--n_test", type=int, default=8, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")

args = parser.parse_args()

INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)

# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def legal_check_for_all_scenes(root_dir, target_shape, img_dir):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/nerfstudio")))
    # NOTE that
    # 07d9f9724ca854fae07cb4c57d7ea22bf667d5decd4058f547728922f909956b
    # images_4 folder has resolution 270x480

    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        # Use a per-scene variable; reusing img_dir here would clobber the parameter and
        # build a wrong path for every scene after the first.
        scene_img_dir = os.path.join(sub_folder, img_dir)  # e.g. <scene>/nerfstudio/images_4
        if not is_image_shape_matched(Path(scene_img_dir), target_shape):
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
    valid_scenes = legal_check_for_all_scenes(INPUT_DIR, target_shape, args.img_subdir)
    print("valid scenes:", len(valid_scenes))

    for stage in ["test"]:

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
            key = os.path.basename(os.path.dirname(image_dir.strip("/")))

            image_dir = Path(image_dir) / args.img_subdir  # 270x480

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
            example["key"] = key

            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()
