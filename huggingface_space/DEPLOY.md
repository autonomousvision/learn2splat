# Deploying the Learn2Splat demo to a Hugging Face Space

`autonomousvision/Learn2Splat` is a **Docker SDK** Space on **GPU** hardware.
It runs `demo.py`'s gradio GUI; HF proxies one port (`app_port`) to the
container.

CUDA extensions are **not compiled in the Space** — the HF Docker builder runs
out of RAM compiling them. They are prebuilt into `wheels/` on a matching
machine (§2); the image just installs the binaries, so Space builds are fast
(~5 min) and never OOM.

## 1. The Space repo

A Docker Space builds from its own git repo. At its root it needs:

- `Dockerfile`, `README.md`, `.dockerignore` — from this `huggingface_space/` folder
- `wheels/` — the prebuilt CUDA wheels from §2
- the learn2splat source: `demo.py`, `learn2splat/`, `submodules/`, `requirements.txt`,
  `pyproject.toml`, `LICENSE`

`README.md` here is the Space card — it replaces the project README.

## 2. Build the CUDA wheels

Build on a machine matching the image: **Python 3.12, torch 2.7.1+cu128,
CUDA 12.8, Ubuntu 22.04 / glibc 2.35**. The `learn2splat` env from `setup.sh`
on an Ubuntu/Pop!_OS 22.04 host qualifies.

`fused_ssim`'s `setup.py` ignores `TORCH_CUDA_ARCH_LIST` and auto-detects the
*build machine's* GPU — on anything but an sm_86 card it bakes the wrong arch.
Force it first: in `submodules/fused-ssim/setup.py`, replace
`arch = f"sm_{compute_capability[0]}{compute_capability[1]}"` with
`arch = "sm_86"`.

The last two wheels below are the **inria** and **fastgs** rasterizer backends
(the ones `setup.sh` builds with `WITH_OPTIONAL_RASTERIZERS=1`) — they provide
the demo's `inria` / `fastgs` Renderer options; without them only `gsplat` is
offered. The fastgs wheel builds
from the *nested* `FastGS` submodule, so initialize submodules first:
`git -C "$SRC" submodule update --init --recursive`. Both honor
`TORCH_CUDA_ARCH_LIST`, so unlike `fused_ssim` they need no manual arch patch.

Then build every wheel for the A10G (`sm_86`):

```bash
SRC=/path/to/learn2splat
export TORCH_CUDA_ARCH_LIST=8.6
# Clear cached build objects from any prior compile (e.g. setup.sh builds each
# submodule for the *build host's* GPU). Otherwise pip wheel silently reuses them
# and bakes the wrong arch — the build succeeds, then fails at runtime on the A10G.
git -C "$SRC" submodule foreach --recursive 'rm -rf build *.egg-info'
python -m pip wheel --no-build-isolation --no-deps -w "$SRC/wheels" \
  "git+https://github.com/nerfstudio-project/nerfacc" \
  "git+https://github.com/nerfstudio-project/gsplat.git" \
  "$SRC"/submodules/pycolmap "$SRC"/submodules/fused-ssim \
  "$SRC"/submodules/simple-knn "$SRC"/submodules/pointops \
  "$SRC"/submodules/fused_knn_attn \
  "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization@26ce026ae9d3cfa56a103279b863a9f320c3e555" \
  "git+https://github.com/fastgs/FastGS.git@44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f#subdirectory=submodules/diff-gaussian-rasterization_fastgs"
```

Revert the `fused-ssim/setup.py` edit afterwards. Verify each CUDA wheel is
`sm_86` — extract its `.so` and run `cuobjdump --list-elf <so> | grep sm_`.

## 3. Assemble and push

```bash
git clone https://huggingface.co/spaces/autonomousvision/Learn2Splat
cd Learn2Splat
SRC=/path/to/learn2splat
git -C "$SRC" submodule update --init --recursive
cp -r "$SRC"/{demo.py,learn2splat,submodules,requirements.txt,pyproject.toml,LICENSE} .
cp -r "$SRC"/wheels .
cp "$SRC"/huggingface_space/{Dockerfile,.dockerignore,README.md} .
git add -A && git commit -m "Learn2Splat interactive demo" && git push
```

## 4. Space settings

- **Hardware: Nvidia A10G** (24 GB) — the GUI holds the dense and sparse
  checkpoints in VRAM together. The wheels are built for the A10G's `sm_86`;
  a different GPU needs the wheels rebuilt for its architecture.
- **Visibility: Public** is recommended. A private Space gates the app behind a
  third-party auth cookie that browsers blocking third-party cookies drop, which
  can break gradio's event stream; Public removes the auth requirement.
- Optionally set a sleep timer to avoid idle GPU billing.

## 5. Notes

- **First boot** downloads the demo scene + both checkpoints from
  `hf://autonomousvision/learn2splat` (~1 GB, a few minutes) before the GUI
  opens.
- **gradio mode** (`demo.py --with-gui gradio`): the learn2splat decoder renders the
  live optimization on the GPU and gradio streams the frames as images; the
  finished splats load into an interactive `Model3D` viewer (a ~30 MB PLY
  served over HTTP per run).
- **Shared GPU state**: all visitors share one set of in-VRAM checkpoints and a
  single serialized GPU lane (`concurrency_id="gpu"`), so runs queue rather than
  overlap (which also bounds VRAM use).
