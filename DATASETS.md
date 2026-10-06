# Datasets

> **Note:** This page started from [ReSplat's `DATASETS.md`](https://github.com/cvg/resplat/blob/main/DATASETS.md) and was extended with DL3DV download tooling (images **and** COLMAP SfM) and with the COLMAP-style evaluation datasets used here.

We train on [DL3DV](https://github.com/DL3DV-10K/Dataset) and evaluate zero-shot on four datasets: [RealEstate10K](https://google.github.io/realestate10k/index.html) (stored as `.torch` chunks) and three COLMAP-style scene collections, [MipNeRF-360](https://jonbarron.info/mipnerf360/), [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) and [LLFF](https://github.com/Fyusion/LLFF).

We also support the synthetic [MegaSynth](https://github.com/hwjiang1510/MegaSynth) RGB-D dataset via our own download/convert tooling; see [its section below](#megasynth-synthetic-rgb-d-torch-chunks).

Two on-disk formats are used:
- `.torch` chunks: RealEstate10K, DL3DV and MegaSynth are converted to PyTorch chunk files that this codebase loads directly. Conversion scripts (DL3DV, MegaSynth) and pointers to upstream tooling (RealEstate10K) are given below.
- COLMAP scenes: MipNeRF-360, DTU and LLFF are read in place from their COLMAP reconstructions, with no conversion step. Each scene is a directory holding a `sparse/` reconstruction and an `images/` (or `images_N/`) folder.

Expected folder structure:

```
├── datasets
│   ├── dl3dv                       # training (.torch chunks)
│   │   ├── train
│   │   │   ├── 000000.torch
│   │   │   ├── ...
│   │   │   └── index.json
│   │   └── test
│   │       ├── 000000.torch
│   │       ├── ...
│   │       └── index.json
│   ├── megasynth                   # synthetic RGB-D (.torch chunks, one scene per file)
│   │   ├── train
│   │   │   ├── 000000.torch
│   │   │   ├── ...
│   │   │   └── index.json
│   │   └── test
│   │       ├── 000000.torch
│   │       ├── ...
│   │       └── index.json
│   ├── re10k                       # evaluation (.torch chunks)
│   │   └── 720p_chunks
│   │       ├── 000000.torch
│   │       ├── ...
│   │       └── index.json
│   ├── mipnerf360                  # evaluation (COLMAP scenes)
│   │   ├── bicycle
│   │   │   ├── sparse/0/           # cameras.bin, images.bin, points3D.bin
│   │   │   └── images_4/           # images_<subsample_factor>
│   │   ├── garden
│   │   └── ...
│   ├── DTU                         # evaluation (COLMAP scenes, same layout)
│   └── nerf_llff_data              # evaluation (COLMAP scenes, same layout)
```

It's recommended to symlink `YOUR_DATASET_PATH` to `datasets`:
```
ln -s YOUR_DATASET_PATH datasets
```
Or point at your data directly with `dataset.roots=[...]` in the config (each launch script under `scripts/testing/` sets this for you).

---

## Training data: DL3DV

We train on DL3DV at a resolution of 256×448 (and 512×960 for high-resolution qualitative results).

The script [`scripts/prepare_and_test_dl3dv.sh`](scripts/prepare_and_test_dl3dv.sh) runs the full pipeline — **download → convert to `.torch` chunks → build the `index.json` → `mode=test`/`train`** — end to end, on a small debug subset by default or the full dataset. Commands for the full dataset follow.

> The DL3DV repos are gated, so first run `huggingface-cli login` and request access to the relevant repos.

### End-to-end script (full or debug subset)

```bash
bash scripts/prepare_and_test_dl3dv.sh
```

By default it downloads `LIMIT=5` benchmark scenes under `datasets/dl3dv-debug/` (mirroring the real dataset layout) and runs the dense COLMAP-init + learned-optimizer path. Set `LIMIT=""` at the top of the script to run the full dataset at the real `datasets/` paths. The steps it runs, and the script each launches:

1. **Download** benchmark scenes — images (`images_4`, 540×960) + COLMAP SfM — with [`learn2splat/scripts/dl3dv_download.py`](learn2splat/scripts/dl3dv_download.py).
2. **Convert** the downloaded images to `.torch` chunks with [`learn2splat/scripts/convert_dl3dv_test.py`](learn2splat/scripts/convert_dl3dv_test.py).
3. **Build the index** (`index.json`) with [`learn2splat/scripts/generate_dl3dv_index.py`](learn2splat/scripts/generate_dl3dv_index.py).
4. *(optional)* drop the raw image frames once the chunks hold them — the dataset reads the chunks, only the COLMAP `sparse/` is still needed (by the initializer). The dense checkpoint is fetched from Hugging Face on first use (an `hf://` ref).
5. **`mode=test`** — optimize each scene and score novel views (`python -m learn2splat.main +experiment=test_dl3dv ...`).
6. **Training smoke test** — a few meta-steps to confirm the train loop runs (`+experiment=train_dl3dv mode=train ...`).

### Full dataset (manual)

Run the same steps at the real `datasets/` paths across all scenes. Download the split you need (`--content` is `images`, `sfm`, or `images+sfm` — frames are read by the **dataset**, the COLMAP SfM by the **colmap initializer**):

```bash
# Test set (140-scene benchmark): images (images_4, 540×960) + COLMAP SfM
python -m learn2splat.scripts.dl3dv_download --source benchmark --content images+sfm \
  --odir datasets/dl3dv-benchmark

# Training corpus (DL3DV-ALL), one batch subset at a time, zip-based
python -m learn2splat.scripts.dl3dv_download --source all --subset 1K --content images+sfm \
  --resolution 480P --odir datasets/dl3dv-colmap-sfm
```

Then convert and index as in steps 2–3 above, picking the converter per split: `convert_dl3dv_test.py` for the [DL3DV-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark) test split (140 scenes), or `convert_dl3dv_train.py` for the [DL3DV-480p](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P) training split (270×480, the 140 test scenes excluded). Update the dataset paths inside these scripts before running.

If you would like to train on the high-resolution DL3DV dataset, you will need to download the [DL3DV-960P](https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P) version (540x960 resolution). Follow the same procedure for data processing, but update the `images_8` folder to `images_4`.

Please follow the [DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md) if you use this dataset in your project and kindly [reference the DL3DV paper](https://github.com/DL3DV-10K/Dataset?tab=readme-ov-file#bibtex).

ReSplat provides a preprocessed subset in `.torch` chunks ([download](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv_960p_test_subset.zip)) containing two test scenes to quickly run inference. Please note that this released subset is intended solely for research purposes. We disclaim any responsibility for the misuse, inappropriate use, or unethical application of the dataset by individuals or entities who download or access it. We kindly ask users to adhere to the [DL3DV license](https://github.com/DL3DV-10K/Dataset/blob/main/License.md).

---

## Evaluation data

### RealEstate10K (`.torch` chunks)

RealEstate10K follows the same **download → convert to `.torch` chunks → build index** pipeline as DL3DV. Follow [pixelSplat](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) and [MVSplat](https://github.com/donydchen/mvsplat) to obtain the chunks. We evaluate on the high-resolution 720p (720×1280) version; the test launcher reads it from `datasets/re10k/720p_chunks` ([`scripts/testing/re10k/launch.sh`](scripts/testing/re10k/launch.sh)). 

For 720p, use the downloading script [here](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download) (it fetches 360p by default — change `360p` to `720p` in [this line](https://github.com/yilundu/cross_attention_renderer/blob/master/data_download/generate_realestate.py#L137)), then convert with the tools [here](https://github.com/dcharatan/real_estate_10k_tools/tree/main/src).

ReSplat provides a preprocessed subset in `.torch` chunks ([download](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/re10k_720p_test_subset.zip)) containing two test scenes to quickly run inference.

The evaluation view pairs are provided in [`assets/re10k_start_0_distance_200_ctx_8v_tgt_8v.json`](assets/re10k_start_0_distance_200_ctx_8v_tgt_8v.json).

### COLMAP scenes: MipNeRF-360, DTU, LLFF

These three benchmarks are read directly from their COLMAP reconstructions by the `colmap` dataset loader ([`learn2splat/dataset/dataset_colmap.py`](learn2splat/dataset/dataset_colmap.py)) — there is **no conversion to `.torch`**. Download each from its original source and arrange it so that every scene is a subdirectory containing:

```
<scene_name>/
├── sparse/0/        # COLMAP: cameras.bin, images.bin, points3D.bin
└── images_N/        # downsampled frames; N = dataset.subsample_factor (falls back to images/)
```

The loader discovers scenes automatically: any subdirectory of the dataset root with a `sparse/` (or `images/`) folder is treated as one scene. The COLMAP SfM points double as the input for the dense COLMAP initializer.

| Dataset      | Expected root                 | Scenes | Subsample | Sources |
| ------------ | ----------------------------- | ------ | --------- | ------- |
| MipNeRF-360  | `datasets/mipnerf360`         | 9      | 4         | [Mip-NeRF 360 project page](https://jonbarron.info/mipnerf360/) (`360_v2` release) |
| DTU          | `datasets/DTU`                | 15     | 2         | [DTU MVS](https://roboimagedata.compute.dtu.dk/?page_id=36), as per-scene COLMAP reconstructions |
| LLFF         | `datasets/nerf_llff_data`     | 8      | 4         | [LLFF](https://github.com/Fyusion/LLFF) / the original `nerf_llff_data` release |

Each launcher under [`scripts/testing/`](scripts/testing/) (`mipnerf360/`, `dtu/`, `llff/`) sets the dataset root, subsample factor, scene-space normalization, and view sampler.

---

## MegaSynth (synthetic RGB-D, `.torch` chunks)

[MegaSynth](https://github.com/hwjiang1510/MegaSynth) is a large synthetic dataset of non-semantic 3D scenes built from geometric primitives (~700K scenes), released **pre-rendered** as RGB-D.

### Download → extract

[`learn2splat/scripts/download_megasynth.sh`](learn2splat/scripts/download_megasynth.sh) fetches splits from Hugging Face:

```bash
# one split (~40 GB) downloaded and unzipped into <out>/extracted/
learn2splat/scripts/download_megasynth.sh /datasets/megasynth 0 --extract
# a range "0-3", a comma list "0,4,7", or "all" (~4 TB) also work
```

### Convert to `.torch` chunks

[`learn2splat/scripts/convert_megasynth.py`](learn2splat/scripts/convert_megasynth.py) packs an extracted split into the same `.torch` schema as DL3DV (`key`, `cameras` `[V,18]` OpenCV `w2c`, `images` as RGB bytes) with a `depths` list (single-channel float16 depth as `.npy` bytes). It writes one scene per `.torch` file with an `index.json`. Every 20th scene goes to `test`, the rest to `train` (`--n_test`). RGB is re-encoded JPG by default (`--rgb_format png` for lossless). The opaque alpha is dropped and the depth's background sentinel is mapped to `0.0` (mask with `depth > 0`).

```bash
# requires an env with torch + OpenCV built with the OpenEXR codec
python learn2splat/scripts/convert_megasynth.py \
  --input_dir  /datasets/megasynth/extracted/split_0 \
  --output_dir datasets/megasynth
```

### Reader and config

The reader [`learn2splat/dataset/dataset_megasynth.py`](learn2splat/dataset/dataset_megasynth.py) (registered as `megasynth`) is a thin subclass of the DL3DV reader that additionally decodes per-view depth when `dataset.load_depth=true`. Defaults live in [`learn2splat/config/dataset/megasynth.yaml`](learn2splat/config/dataset/megasynth.yaml) (512×512, `index.json`).

Please follow the [MegaSynth license/terms](https://github.com/hwjiang1510/MegaSynth) if you use this dataset.

---
