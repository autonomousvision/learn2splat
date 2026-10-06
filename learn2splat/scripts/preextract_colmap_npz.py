"""Pre-extract COLMAP point clouds for all scenes and save as .npz files.

Run once before training to avoid slow COLMAP parsing at every iteration:

    python learn2splat/scripts/preextract_colmap_npz.py --root <path/to/scenes> [--normalize] [--workers 8]

Each scene directory is expected to contain a `sparse/0/` (or `sparse/`)
sub-directory with the standard COLMAP binary model files.

For every scene a file `colmap_points_cache.npz` (or
`colmap_points_cache_norm.npz` when --normalize is used) is written next to
the scene directory.  The InitializerColmap class will pick these files up
automatically and skip the full SceneManager parse.
"""

import argparse
import concurrent.futures
import sys
import traceback
from pathlib import Path

import numpy as np

# Make sure the project root is on sys.path so that src.* imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from learn2splat.dataset.colmap.utils import Parser


def _npz_path(scene_dir: Path, normalize: bool) -> Path:
    suffix = "_norm" if normalize else ""
    return scene_dir / f"colmap_points_cache{suffix}.npz"


def process_scene(scene_dir: Path, normalize: bool, overwrite: bool) -> str:
    npz = _npz_path(scene_dir, normalize)
    if npz.exists() and not overwrite:
        return f"SKIP  {scene_dir.name}"
    try:
        parser = Parser(
            data_dir=str(scene_dir),
            factor=1,
            normalize=normalize,
            load_images=False,
            dl3dv_settings=False,
            verbose=False,
        )
        np.savez_compressed(
            npz,
            points=parser.points,
            points_rgb=parser.points_rgb,
            camtoworlds=parser.camtoworlds,
        )
        return f"OK    {scene_dir.name}  ({parser.points.shape[0]} pts)"
    except Exception as e:
        return f"ERROR {scene_dir.name}: {e}\n{traceback.format_exc()}"


def find_scene_dirs(root: Path) -> list[Path]:
    """Return all direct children of root that look like a COLMAP scene."""
    scenes = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sparse = child / "sparse" / "0"
        if not sparse.exists():
            sparse = child / "sparse"
        if sparse.exists():
            scenes.append(child)
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="Root directory containing one sub-dir per scene.")
    parser.add_argument("--normalize", action="store_true", help="Apply world-space normalisation (matches normalize_world_space: true in config).")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if .npz already exists.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4).")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    scenes = find_scene_dirs(root)
    if not scenes:
        print(f"No COLMAP scene directories found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scenes)} scenes under {root}")
    print(f"normalize={args.normalize}  overwrite={args.overwrite}  workers={args.workers}\n")

    ok = skip = error = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_scene, s, args.normalize, args.overwrite): s for s in scenes}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            msg = fut.result()
            prefix = msg[:5].strip()
            if prefix == "OK":
                ok += 1
            elif prefix == "SKIP":
                skip += 1
            else:
                error += 1
            print(f"[{i}/{len(scenes)}] {msg}")

    print(f"\nDone. OK={ok}  skipped={skip}  errors={error}")


if __name__ == "__main__":
    main()

