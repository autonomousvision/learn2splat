"""Convert an extracted MegaSynth split into resplat/DL3DV-style `.torch` chunks.

MegaSynth ships pre-rendered RGB-D scenes; this packs each scene into the exact
on-disk format the learn2splat/resplat dataloader already consumes (the `Example`
TypedDict in `learn2splat/scripts/convert_dl3dv_utils.py`, decoded by
`learn2splat/dataset/dataset_re10k.py`). Images and depths are stored as *encoded
bytes* rather than decoded tensors, so the loader decodes only the handful of
views it samples per iteration -- giving cheap per-view random access and a
small on-disk footprint (~31 MB/scene; ~21 GB for a 672-scene split).

Input -- one extracted split, one folder per scene keyed by uuid:
    <split>/<uuid>/hanwen/opencv_cameras.json        # per-view intrinsics + w2c
    <split>/<uuid>/hanwen/renderings/<idx>_rgba.png  # 512x512 RGBA, alpha opaque
    <split>/<uuid>/hanwen/renderings/<idx>_depth.exr # 512x512x3 float32 (replicated)

Output -- DL3DV-style dataset tree (one example/scene per `.torch` by default):
    <out>/{train,test}/<NNNNNN>.torch   # a list[Example]; one scene per file
    <out>/{train,test}/index.json       # {example_key: chunk_filename}

Each `Example` is a dict with these keys (V = number of views in the scene):
    key:        f"megasynth_{uuid}"      # unique scene id; the index.json key
    url:        uuid                     # provenance string (kept for DL3DV parity)
    timestamps: int64[V]                 # integer view indices; align images/depths
    cameras:    float32[V, 18]           # packed pose per view, layout below
    images:     list[V] of uint8[*]      # RGB JPG/PNG *file bytes* (alpha dropped)
    depths:     list[V] of uint8[*]      # float16 HxW depth as *.npy bytes*

The 18-D camera vector per view matches `convert_poses` in dataset_re10k.py:
    [ fx/W, fy/H, cx/W, cy/H,            # intrinsics, normalized by image W/H
      0, 0,                              # two reserved/padding slots
      w2c[0, :], w2c[1, :], w2c[2, :] ]  # the 3x4 of a 4x4 OpenCV world-to-camera
The loader rebuilds K (3x3) from the first four entries and inverts the 3x4 w2c
into an OpenCV camera-to-world. MegaSynth already stores OpenCV w2c, so (unlike
the DL3DV converter) no Blender->OpenCV basis change is applied here.

Conventions / decisions this converter makes:
  * near/far are NOT stored here -- the dataset config supplies them at train time.
  * Alpha is dropped: MegaSynth scene renders are fully opaque (alpha == 255), so
    it carries no information.
  * Depth: the EXR's three channels are identical, so we keep one, cast to
    float16, and map the `1e10` background sentinel (and any non-finite value) to
    0.0. It decodes via the loader's TartanAir `np.load` path; downstream
    `depth > 0` selects the valid (foreground) pixels.

Requires an env with torch + OpenCV built with the OpenEXR codec (the repo's
`learn2splat` env qualifies). Run from the repo root with that env's Python:
    python learn2splat/scripts/convert_megasynth.py \
        --input_dir  <megasynth_extracted>/split_0 \
        --output_dir datasets/megasynth
"""

import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def encode_rgb(png_path: Path, fmt: str, quality: int) -> torch.Tensor:
    """Load an RGBA render, drop the (opaque) alpha, re-encode as RGB bytes."""
    im = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    if fmt == "jpg":
        im.save(buf, format="JPEG", quality=quality)
    else:
        im.save(buf, format="PNG", optimize=True)
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


def encode_depth(exr_path: Path) -> torch.Tensor:
    """Read a float32 EXR depth, collapse to single-channel float16 .npy bytes.

    Background/empty pixels carry a `1e10` sentinel (and would overflow float16);
    map those plus any non-finite values to 0.0, the standard invalid-depth marker
    that depth losses mask on (`depth > 0`). Real depth is 0.1..~tens, float16-safe.
    """
    d = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"failed to read EXR: {exr_path}")
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    d[~np.isfinite(d) | (d > 65504.0)] = 0.0  # background sentinel -> invalid
    buf = io.BytesIO()
    np.save(buf, d.astype(np.float16))
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


def view_index(name: str) -> int:
    """`00000007_rgba.png` / `00000007_depth.exr` -> 7."""
    return int(os.path.basename(name).split("_")[0])


def build_example(scene_dir: Path, fmt: str, quality: int) -> dict:
    """Build one DL3DV-style `Example` dict from a single MegaSynth scene folder.

    Walks the scene's camera frames (one per rendered view), packing each view's
    intrinsics + extrinsics into the 18-D vector documented in the module
    docstring and encoding its RGB and depth as file bytes. `timestamps` records
    the integer view index parsed from each frame's filename so the loader can
    line up `images[i]` / `depths[i]` with `cameras[i]`.
    """
    uuid = scene_dir.name
    hanwen = scene_dir / "hanwen"
    renderings = hanwen / "renderings"
    cams = json.loads((hanwen / "opencv_cameras.json").read_text())

    timestamps, cameras = [], []
    images, depths = [], []
    for fr in cams["frames"]:
        idx = view_index(fr["file_path"])
        w, h = float(fr["w"]), float(fr["h"])
        # entries 0-5: intrinsics normalized by image size (fx/W, fy/H, cx/W, cy/H) + 2 pad
        cam = [fr["fx"] / w, fr["fy"] / h, fr["cx"] / w, fr["cy"] / h, 0.0, 0.0]
        w2c = np.asarray(fr["w2c"], dtype=np.float64)  # 4x4 OpenCV world-to-camera
        cam.extend(w2c[:3].flatten().tolist())  # entries 6-17: 3x4 (drop homogeneous row), row-major

        timestamps.append(idx)
        cameras.append(np.asarray(cam, dtype=np.float64))
        images.append(encode_rgb(renderings / f"{idx:08d}_rgba.png", fmt, quality))
        depths.append(encode_depth(renderings / f"{idx:08d}_depth.exr"))

    return {
        "key": f"megasynth_{uuid}",
        "url": uuid,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(np.stack(cameras), dtype=torch.float32),
        "images": images,
        "depths": depths,
    }


def save_chunk(chunk: list[dict], index: dict, out_stage: Path, chunk_index: int) -> None:
    """Write one `.torch` chunk (a list of examples) and record each key in `index`.

    `index` is mutated in place to map every example's key to this chunk's
    filename; it is later serialized as the stage's index.json.
    """
    name = f"{chunk_index:06d}.torch"
    torch.save(chunk, out_stage / name)
    for ex in chunk:
        index[ex["key"]] = name


def main() -> None:
    """CLI entry point: discover scenes, split train/test, convert, write index.json.

    Scenes are discovered deterministically (sorted by uuid) so the split is
    stable across runs; the test split is every `--n_test`-th scene. Examples are
    grouped into `.torch` files of `--scenes_per_chunk` scenes each (default 1 =
    one file per scene), and a per-stage `index.json` mapping key -> filename is
    written. Scenes that fail to convert are reported and skipped, not fatal.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="extracted split dir (contains <uuid>/ scene folders)")
    p.add_argument("--output_dir", required=True, help="output dataset dir (will hold train/ and test/)")
    p.add_argument("--rgb_format", choices=["jpg", "png"], default="jpg", help="RGB encoding (default jpg)")
    p.add_argument("--rgb_quality", type=int, default=95, help="JPG quality (default 95)")
    p.add_argument("--n_test", type=int, default=20, help="every n_test-th scene goes to the test split")
    p.add_argument("--scenes_per_chunk", type=int, default=1, help="examples per .torch file (default 1)")
    p.add_argument("--limit", type=int, default=0, help="convert at most N scenes (0 = all); for smoke tests")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    scenes = sorted([d for d in in_dir.iterdir() if (d / "hanwen" / "opencv_cameras.json").is_file()])
    if args.limit:
        scenes = scenes[: args.limit]

    test_set = set(scenes[:: args.n_test])
    splits = {
        "train": [s for s in scenes if s not in test_set],
        "test": [s for s in scenes if s in test_set],
    }
    print(f"{len(scenes)} scenes -> train {len(splits['train'])}, test {len(splits['test'])}")

    for stage, stage_scenes in splits.items():
        out_stage = out_dir / stage
        out_stage.mkdir(parents=True, exist_ok=True)
        index: dict[str, str] = {}
        chunk: list[dict] = []
        chunk_index = 0
        for scene_dir in tqdm(stage_scenes, desc=f"convert {stage}"):
            try:
                chunk.append(build_example(scene_dir, args.rgb_format, args.rgb_quality))
            except Exception as e:  # noqa: BLE001 - skip and report bad scenes
                print(f"[skip] {scene_dir.name}: {type(e).__name__}: {e}")
                continue
            if len(chunk) >= args.scenes_per_chunk:
                save_chunk(chunk, index, out_stage, chunk_index)
                chunk_index += 1
                chunk = []
        if chunk:
            save_chunk(chunk, index, out_stage, chunk_index)
        (out_stage / "index.json").write_text(json.dumps(index))
        n_files = len(list(out_stage.glob("*.torch")))
        print(f"  wrote {n_files} .torch files -> {out_stage} ({len(index)} keys)")


if __name__ == "__main__":
    main()
