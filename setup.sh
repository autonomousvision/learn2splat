#!/usr/bin/env bash
# Default environment setup for learn2splat.
# Tested on PyTorch 2.7.1, CUDA 12.8, Ubuntu 22.04.
#
# Run from the repository root:
#   bash setup.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-learn2splat}"   # set ENV_NAME to install into a differently named env
PY_VERSION=3.12
CUDA_CHANNEL="nvidia/label/cuda-12.8.0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\n==> %s\n' "$*"; }

# --- Locate conda --------------------------------------------------------
# `conda activate` is a shell function only available in interactive shells
# (loaded by `conda init`). In a non-interactive script we must load the
# hook ourselves before activating. Using `conda shell.bash hook` (rather
# than sourcing conda.sh by path) works regardless of where conda is
# installed, including system/RPM installs where the path is non-standard.
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
  cat >&2 <<'EOF'
ERROR: could not locate the conda executable.

This script needs conda available in the shell that runs it. Neither the
CONDA_EXE environment variable nor a `conda` on PATH was found.

Fix: run this from a shell where conda is initialized (your prompt should
show the active environment, e.g. "(base)"). If conda has never been set up
in this shell, run `conda init bash` once, then restart the shell, or
manually source your conda install, e.g.:

    source /path/to/miniconda3/etc/profile.d/conda.sh

Then re-run:  bash setup.sh
EOF
  exit 1
fi

# --- Conda environment ---------------------------------------------------
# conda's generated activate/deactivate scripts (e.g. the gcc_linux-64 ones)
# reference variables like _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED without
# guarding them, which is fatal under `set -u`. These hooks also re-run on
# `conda install` (when gcc-related packages change), so keep nounset off for
# the entire conda section and restore it before the pip steps.
set +u
eval "$("$CONDA_BIN" shell.bash hook)"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]|/envs/${ENV_NAME}$"; then
  log "conda env '${ENV_NAME}' already exists; skipping creation."
else
  log "Creating conda env '${ENV_NAME}' (python ${PY_VERSION})"
  conda create -n "$ENV_NAME" "python=${PY_VERSION}" -y
fi
conda activate "$ENV_NAME"

log "Installing CUDA toolkit + build tools"
conda install -c "$CUDA_CHANNEL" --strict-channel-priority -y \
  cuda-nvcc cuda-cudart-dev cuda-cccl libcublas-dev libcusparse-dev \
  libcusolver-dev libcurand-dev libcufft-dev cuda-nvrtc cuda-nvrtc-dev \
  cuda-cudart-static
conda install -y ninja conda-forge::glm
set -u

# --- PyTorch + Python deps ----------------------------------------------
# Use `python -m pip` (not bare `pip`) so installs always target the active
# env's interpreter even if a stale pip shim is earlier on PATH.
PIP=(python -m pip)

log "Installing PyTorch (cu128)"
"${PIP[@]}" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

log "Installing Python requirements"
"${PIP[@]}" install -r "$REPO_ROOT/requirements.txt"

# Skip the build if already exists.
pip_install_pinned() {  # <import name> <version> <pip target>
  if python -c "import $1, sys; sys.exit(0 if $1.__version__ == '$2' else 1)" 2>/dev/null; then
    log "$1 $2 already installed; skipping."
  else
    "${PIP[@]}" install --no-build-isolation "$3"
  fi
}
pip_install_pinned nerfacc 0.5.3 \
  git+https://github.com/nerfstudio-project/nerfacc@57ccfa14feb94975836ea6913149a86737220f2b
pip_install_pinned gsplat 1.5.3 \
  git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# --- Submodules ----------------------------------------------------------
# simple-knn / fused-ssim / pycolmap are git submodules. pointops and fused_knn_attn
# have no upstream repository: their source is part of this repo, so a plain clone has them.
# Initialize the submodules here so a checkout without `git clone --recurse-submodules`
# still builds. Skipped when there's no .git (e.g. a source archive), where the
# submodules are expected to be in place already.
if [ -e "$REPO_ROOT/.git" ]; then  # a dir in a clone, a file in a worktree
  log "Initializing git submodules"
  git -C "$REPO_ROOT" submodule update --init --recursive
fi

log "Installing submodules"
"${PIP[@]}" install "$REPO_ROOT/submodules/pycolmap"
"${PIP[@]}" install --no-build-isolation "$REPO_ROOT/submodules/fused-ssim"
"${PIP[@]}" install --no-build-isolation "$REPO_ROOT/submodules/simple-knn"
"${PIP[@]}" install --no-build-isolation "$REPO_ROOT/submodules/pointops"
"${PIP[@]}" install --no-build-isolation "$REPO_ROOT/submodules/fused_knn_attn"

# Gaussian rasterizers for the two optional decoders (decoder=fastgs, decoder=inria).
# The default gsplat decoder needs neither, so these two CUDA builds are opt-in:
#   WITH_OPTIONAL_RASTERIZERS=1 bash setup.sh
# Both are pinned forks of the original 3DGS rasterizer; FastGS keeps theirs in a
# subdirectory of their repo.
if [ "${WITH_OPTIONAL_RASTERIZERS:-0}" = "1" ]; then
  log "Installing the optional rasterizers (fastgs, inria)"
  "${PIP[@]}" install --no-build-isolation \
    "git+https://github.com/fastgs/FastGS.git@44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f#subdirectory=submodules/diff-gaussian-rasterization_fastgs"
  "${PIP[@]}" install --no-build-isolation \
    "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization@26ce026ae9d3cfa56a103279b863a9f320c3e555"
else
  log "Skipping the optional rasterizers (decoder=fastgs / decoder=inria). Re-run with WITH_OPTIONAL_RASTERIZERS=1 to build them."
fi

# --- learn2splat package -------------------------------------------------------
# Editable install so source edits take effect immediately and the `learn2splat`
# console command + `import learn2splat` (incl. bundled Hydra configs) are available.
# --no-deps: requirements.txt was already installed above; avoid re-resolving
# (which could disturb the CUDA-specific torch build).
log "Installing learn2splat (editable)"
"${PIP[@]}" install --no-build-isolation --no-deps -e "$REPO_ROOT"

log "Setup complete. Activate with:  conda activate ${ENV_NAME}"
