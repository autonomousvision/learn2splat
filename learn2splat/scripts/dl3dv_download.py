# ---------------------------------------------------------------------------
# Adapted (with modifications) from the DL3DV-10K download scripts:
#   https://github.com/DL3DV-10K/Dataset/blob/main/scripts/download.py
#   https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/blob/main/download.py
#
# Original work © the DL3DV-10K authors, licensed under the Creative Commons
# Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0):
#   https://creativecommons.org/licenses/by-nc/4.0/
#
# This file is a modified version of that work and is distributed under the
# same CC BY-NC 4.0 license. It is provided "as is", and may be used for NonCommercial purposes only.
#
# NOTE: This per-file license differs from the repository's main license;
# ---------------------------------------------------------------------------
"""Unified DL3DV downloader.

Downloads DL3DV data from Hugging Face for two sources:

  --source all        the 10K corpus repos (DL3DV-ALL-{480P,960P,2K,4K},
                      DL3DV-ALL-ColmapCache), selected by batch subset
                      (1K..11K) or a single --hash.
  --source benchmark  the 140-scene DL3DV-10K-Benchmark (the evaluation set).

For either source, --content chooses what to fetch:

  images       image frames + poses (the data WITHOUT the SfM).
  sfm          only the COLMAP sparse SfM (cameras/images/points3D.bin),
               rearranged to <scene>/sparse for the colmap initializer.
  images+sfm   both.

Both DL3DV repos are gated: you must request access on Hugging Face and be
logged in (`huggingface-cli login` or HF_TOKEN). This script downloads nothing
itself — it requires each user to obtain their own access — so it does not
redistribute the dataset.

Example:
  python -m learn2splat.scripts.dl3dv_download --source benchmark \
      --odir datasets/dl3dv-benchmark --content images+sfm --clean_cache
  python -m learn2splat.scripts.dl3dv_download --source all \
      --odir datasets/dl3dv-colmap-sfm --content sfm --subset 1K --clean_cache
"""

import argparse
import os
import pathlib
import shutil
import traceback
import urllib.request
import zipfile
from os.path import join

import pandas as pd
from huggingface_hub import HfApi, HfFileSystem, snapshot_download
from tqdm import tqdm

api = HfApi()

# --- source: all (10K corpus) -------------------------------------------------
RESOLUTION2REPO = {
    '480P': 'DL3DV/DL3DV-ALL-480P',
    '960P': 'DL3DV/DL3DV-ALL-960P',
    '2K': 'DL3DV/DL3DV-ALL-2K',
    '4K': 'DL3DV/DL3DV-ALL-4K',
}
COLMAP_CACHE_REPO = 'DL3DV/DL3DV-ALL-ColmapCache'
ALL_META_LINK = 'https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv'

# --- source: benchmark (140-scene eval set) -----------------------------------
BENCHMARK_REPO = 'DL3DV/DL3DV-10K-Benchmark'

SFM_BIN_FILES = {"cameras.bin", "images.bin", "points3D.bin"}


# =============================================================================
# Shared helpers
# =============================================================================
def hf_download_path(repo: str, rel_path: str, odir: str, max_try: int = 5):
    """Download a single file from a HF dataset repo, retrying on failure."""
    counter = 0
    while True:
        if counter >= max_try:
            print(f"ERROR: Download {repo}/{rel_path} failed.")
            return False
        try:
            api.hf_hub_download(repo_id=repo, filename=rel_path, repo_type='dataset',
                                local_dir=odir, cache_dir=join(odir, '.cache'))
            return True
        except KeyboardInterrupt:
            print('Keyboard Interrupt. Exit.')
            raise SystemExit(1)
        except BaseException:
            traceback.print_exc()
            counter += 1


def download_from_url(url: str, ofile: str):
    """Download a file from a plain URL to ofile."""
    try:
        urllib.request.urlretrieve(url, ofile)
        return True
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")
        return False


def clean_huggingface_cache(output_dir: str):
    """Remove the local HF cache dir to save space (no good hub API for this)."""
    cur_cache_dir = join(output_dir, '.cache')
    if os.path.exists(cur_cache_dir):
        shutil.rmtree(cur_cache_dir)


def verify_access(repo: str) -> bool:
    """Return True if the logged-in user can list the (gated) repo."""
    fs = HfFileSystem()
    try:
        fs.ls(f'datasets/{repo}')
        return True
    except BaseException:
        return False


# =============================================================================
# SfM cleanup / validation (shared by both sources; layouts differ per source)
# =============================================================================
def sfm_cleanup_scene(scene_dir: pathlib.Path):
    """Keep only the COLMAP sparse SfM files, then move them to <scene>/sparse.

    Used by the 'all' source, whose ColmapCache zips contain
    <scene>/colmap/sparse/* alongside many other files.
    """
    print(f"Cleaning up SfM scene at {scene_dir.resolve()}")
    scene_dir = scene_dir.resolve()
    if not scene_dir.exists():
        print(f"[WARN] {scene_dir} does not exist")
        return

    # First pass: delete everything except the sparse .bin files and transforms.json.
    for path in scene_dir.rglob("*"):
        if path.is_file():
            is_bin_file = (path.name in SFM_BIN_FILES and
                           path.parent.name.isdigit() and
                           path.parent.parent.name == "sparse")
            is_transforms_file = (path.name == "transforms.json" and path.parent == scene_dir)
            if is_bin_file or is_transforms_file:
                continue
            path.unlink()

    # Second pass: remove empty directories bottom-up.
    for path in sorted(scene_dir.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()

    # Third pass: lift the SfM up to <dataset>/<hash>/ (out of the scratch/subset scene dir):
    #   <scene>/colmap/sparse  -> <dataset>/<hash>/sparse
    #   <scene>/transforms.json -> <dataset>/<hash>/transforms.json  (next to the SfM, for future use)
    dataset_dir = scene_dir.parent.parent
    colmap_dir = scene_dir / "colmap"
    sparse_dir = colmap_dir / "sparse"
    if sparse_dir.exists():
        target_scene_dir = dataset_dir / scene_dir.name
        target_scene_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(sparse_dir), str(target_scene_dir / "sparse"))
        transforms = scene_dir / "transforms.json"
        if transforms.is_file():
            shutil.move(str(transforms), str(target_scene_dir / "transforms.json"))
        if not any(colmap_dir.iterdir()):
            colmap_dir.rmdir()
        if not any(scene_dir.iterdir()):
            scene_dir.rmdir()


def validate_sfm_structure(scene_dir: pathlib.Path, unsucc_count: int = 0) -> bool:
    """Validate that a cleaned scene dir holds sparse/0/{cameras,images,points3D}.bin."""
    scene_dir = scene_dir.resolve()
    if not scene_dir.exists():
        print(f"[WARN: {unsucc_count}] {scene_dir} does not exist")
        return False

    sparse_0_dir = scene_dir / "sparse" / "0"
    for bin_file in SFM_BIN_FILES:
        if not (sparse_0_dir / bin_file).is_file():
            print(f"[ERROR: {unsucc_count}] {bin_file} is missing in {sparse_0_dir}")
            return False
    return True


# =============================================================================
# Source: all (10K corpus)
# =============================================================================
def all_to_download_item(hash_name: str, reso: str, batch: str, file_type: str) -> dict:
    if file_type == 'images':
        repo = RESOLUTION2REPO[reso]
        rel_path = f'{batch}/{hash_name}.zip'
    elif file_type == 'sfm':
        repo = COLMAP_CACHE_REPO
        rel_path = f'{batch}/{hash_name}.zip'
    else:
        raise ValueError(f'Unknown file_type {file_type!r} for source=all.')
    return {'repo': repo, 'rel_path': rel_path, 'file_type': file_type}


def all_get_download_list(subset_opt, hash_name, reso_opt, file_types, output_dir, limit=None):
    """Build the download list for the 'all' source from the DL3DV-valid.csv meta."""
    cache_folder = join(output_dir, '.cache')
    meta_file = join(cache_folder, 'DL3DV-valid.csv')
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.exists(meta_file):
        assert download_from_url(ALL_META_LINK, meta_file), 'Download meta file failed.'

    df = pd.read_csv(meta_file)

    if hash_name:
        assert hash_name in df['hash'].values, f'Hash {hash_name} not found in the meta file.'
        rows = [(hash_name, df[df['hash'] == hash_name]['batch'].values[0])]
    else:
        subdf = df[df['batch'] == subset_opt]
        rows = [(r['hash'], subset_opt) for _, r in subdf.iterrows()]

    if limit is not None:
        rows = rows[:limit]

    ret = []
    for h, batch in rows:
        for file_type in file_types:
            ret.append(all_to_download_item(h, reso_opt, batch, file_type))
    return ret


def _zip_common_prefix(zip_ref, hash_name):
    """Return the zip's single top-level dir if it equals hash_name, else None."""
    zip_contents = zip_ref.namelist()
    if zip_contents and '/' in zip_contents[0]:
        potential_prefix = zip_contents[0].split('/')[0] + '/'
        if all(p.startswith(potential_prefix) for p in zip_contents if not p.endswith('/')):
            if potential_prefix.rstrip('/') == hash_name:
                return hash_name
    return None


def all_download(download_list, output_dir, is_clean_cache):
    """Download and unzip the 'all'-source items.

    Images and SfM go to separate directories so the (destructive) SfM cleanup never
    touches the images:
      images → <odir>/<subset>/<hash>/   (image frames + transforms.json)
      sfm    → <odir>/<hash>/sparse/      (cleaned; extracted via a scratch dir)
    """
    output_dir = pathlib.Path(output_dir)
    succ_count = 0
    for item in tqdm(download_list, desc='Downloading'):
        repo, rel_path, file_type = item['repo'], item['rel_path'], item['file_type']
        hash_name = os.path.splitext(os.path.basename(rel_path))[0]
        subset_name = os.path.dirname(rel_path)

        # Skip if the (per-content) final target already exists.
        if file_type == 'sfm':
            if validate_sfm_structure(output_dir / hash_name):
                succ_count += 1
                continue
        else:  # images
            if (output_dir / subset_name / hash_name).exists():
                succ_count += 1
                continue

        if not hf_download_path(repo, rel_path, str(output_dir)):
            print(f'Download {rel_path} failed')
            continue
        succ_count += 1
        if is_clean_cache:
            clean_huggingface_cache(str(output_dir))

        if not rel_path.endswith('.zip'):
            continue
        zip_file = str(output_dir / rel_path)

        if file_type == 'sfm':
            # Extract into a scratch dir; sfm_cleanup_scene keeps only the sparse SfM and
            # moves it to <odir>/<hash>/sparse, away from the images at <odir>/<subset>/<hash>.
            scratch_base = output_dir / '.sfm_scratch'
            scene_dir = scratch_base / hash_name
            scene_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                dest = scratch_base if _zip_common_prefix(zip_ref, hash_name) else scene_dir
                zip_ref.extractall(str(dest))
            sfm_cleanup_scene(scene_dir)
            shutil.rmtree(scratch_base, ignore_errors=True)
        else:  # images
            target_dir = output_dir / subset_name / hash_name
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                dest = output_dir / subset_name if _zip_common_prefix(zip_ref, hash_name) else target_dir
                zip_ref.extractall(str(dest))
        os.remove(zip_file)

    print(f'Summary: {succ_count}/{len(download_list)} files downloaded successfully')
    return succ_count == len(download_list)


# =============================================================================
# Source: benchmark (140-scene eval set)
# =============================================================================
def _benchmark_surface_sfm(odir, scene_hash):
    """Surface a benchmark scene's COLMAP SfM for the colmap initializer.

    <scene>/nerfstudio/colmap/sparse  ->  <scene>/sparse
    transforms.json is copied to the scene root (next to the SfM, for future colmap/test use)
    while the original stays under nerfstudio/ where the image-conversion step reads it.
    """
    scene_root = pathlib.Path(odir) / scene_hash
    nerf_dir = scene_root / 'nerfstudio'
    nerf_transforms = nerf_dir / 'transforms.json'
    if nerf_transforms.is_file():
        shutil.copy2(str(nerf_transforms), str(scene_root / 'transforms.json'))
    dst_sparse = scene_root / 'sparse'
    if dst_sparse.exists():
        shutil.rmtree(dst_sparse)
    shutil.move(str(nerf_dir / 'colmap' / 'sparse'), str(dst_sparse))
    colmap_dir = nerf_dir / 'colmap'
    if colmap_dir.exists() and not any(colmap_dir.iterdir()):
        colmap_dir.rmdir()


def benchmark_download(subset_opt, hash_name, contents, output_dir, is_clean_cache, limit=None):
    """Download benchmark scenes (full set or a single --hash) for the requested contents.

    The benchmark stores each scene's frames as individual files, so we fetch them with a single
    parallel snapshot_download using per-scene glob patterns. Only the nerfstudio assets are
    pulled, skipping the heavy gaussian_splat/ MVS+dense cache:
      images → nerfstudio/images_4/* (+ transforms.json)
      sfm    → nerfstudio/colmap/sparse/* (+ transforms.json), then surfaced to <scene>/sparse.
    """
    # benchmark-meta.csv gives the scene list (for --hash validation and --limit).
    if not hf_download_path(BENCHMARK_REPO, 'benchmark-meta.csv', output_dir):
        print('ERROR: Download benchmark-meta.csv failed.')
        return False
    hashlist = pd.read_csv(join(output_dir, 'benchmark-meta.csv'))['hash'].tolist()

    if subset_opt == 'hash':
        if hash_name not in hashlist:
            print(f'ERROR: hash {hash_name} not in benchmark-meta.csv')
            return False
        download_list = [hash_name]
    else:
        download_list = hashlist
    if limit is not None:
        download_list = download_list[:limit]

    # Glob patterns for exactly the files we want (snapshot_download matches with fnmatch,
    # where '*' also spans '/').
    patterns = []
    for h in download_list:
        if 'images' in contents:
            patterns += [f'{h}/nerfstudio/images_4/*', f'{h}/nerfstudio/transforms.json']
        if 'sfm' in contents:
            patterns += [f'{h}/nerfstudio/colmap/sparse/*', f'{h}/nerfstudio/transforms.json']

    print(f'Downloading {len(download_list)} benchmark scene(s) in parallel...')
    snapshot_download(repo_id=BENCHMARK_REPO, repo_type='dataset', local_dir=output_dir,
                      allow_patterns=patterns, cache_dir=join(output_dir, '.cache'))

    if 'sfm' in contents:
        for h in download_list:
            _benchmark_surface_sfm(output_dir, h)
    if is_clean_cache:
        clean_huggingface_cache(output_dir)
    return True


# =============================================================================
# CLI
# =============================================================================
def _contents_from_arg(content: str) -> list[str]:
    return {'images': ['images'], 'sfm': ['sfm'], 'images+sfm': ['images', 'sfm']}[content]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=['all', 'benchmark'], required=True,
                        help='all = 10K corpus repos; benchmark = 140-scene eval set')
    parser.add_argument('--odir', type=str, default=None,
                        help='output directory (default: datasets/dl3dv-colmap-sfm for --source all, '
                             'datasets/dl3dv-benchmark for --source benchmark)')
    parser.add_argument('--content', choices=['images', 'sfm', 'images+sfm'], default='images+sfm',
                        help='what to download: image frames+poses, the COLMAP SfM, or both')
    parser.add_argument('--hash', type=str, default='', help='download a single scene by hash')
    parser.add_argument('--clean_cache', action='store_true',
                        help='remove the HF cache after each download to save space')
    parser.add_argument('--limit', default=None,
                        help='download at most N scenes (a debug-sized set); empty or 0 means no limit')
    # source=all options
    parser.add_argument('--subset', choices=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K',
                                             '9K', '10K', '11K'],
                        help='[all] batch subset to download (ignored if --hash is set)')
    parser.add_argument('--resolution', choices=['480P', '960P', '2K', '4K'], default='480P',
                        help='[all] image resolution repo')
    # (benchmark images are always fetched at images_4 / 960x540, skipping the heavy
    #  gaussian_splat MVS+dense cache.)
    args = parser.parse_args()

    # Default output dir matches the existing per-source conventions:
    # all → train SfM dir, benchmark → test SfM dir.
    if args.odir is None:
        args.odir = {'all': 'datasets/dl3dv-colmap-sfm',
                     'benchmark': 'datasets/dl3dv-benchmark'}[args.source]

    # Empty string or 0 means "no limit" (download the whole subset / benchmark).
    limit = None if args.limit in (None, '', '0', 0) else int(args.limit)

    os.makedirs(args.odir, exist_ok=True)
    contents = _contents_from_arg(args.content)

    if args.source == 'all':
        if not args.hash and not args.subset:
            parser.error('source=all requires --subset (or --hash).')
        file_types = ['images' if c == 'images' else 'sfm' for c in contents]
        # Verify access to every repo we will hit.
        repos = set()
        if 'images' in contents:
            repos.add(RESOLUTION2REPO[args.resolution])
        if 'sfm' in contents:
            repos.add(COLMAP_CACHE_REPO)
        for repo in repos:
            if not verify_access(repo):
                print(f'No access to https://huggingface.co/datasets/{repo} — request access and '
                      f'log in (huggingface-cli login).')
                raise SystemExit(1)
        download_list = all_get_download_list(args.subset, args.hash, args.resolution,
                                              file_types, args.odir, limit=limit)
        ok = all_download(download_list, args.odir, args.clean_cache)
    else:
        try:
            user = api.whoami()
            print(f'Logged in as {user["name"]}')
        except Exception:
            print('ERROR: Hugging Face login failed. Check your connection and token '
                  '(huggingface-cli login / HF_TOKEN).')
            raise SystemExit(1)
        subset_opt = 'hash' if args.hash else 'full'
        ok = benchmark_download(subset_opt, args.hash, contents, args.odir, args.clean_cache,
                                limit=limit)

    if ok:
        print('Download Done. Refer to', args.odir)
    else:
        print(f'Download to {args.odir} failed. See error message above.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
