# Learn2Splat — interactive demo for a Hugging Face Space (Docker SDK, GPU).
#
# Installs the learn2splat package + prebuilt CUDA-extension wheels, then runs
# demo.py's gradio GUI: SfM-initialize a COLMAP scene and refine the Gaussians
# with the learned optimizer live in the browser.
#
# The CUDA extensions are NOT compiled here — the HF Docker builder runs out of
# RAM doing it. They are prebuilt into wheels/ on a machine matching this image
# (Python 3.12, torch 2.7.1+cu128, glibc 2.35); see huggingface_space/DEPLOY.md.
#
# Build context = the Space repo root (learn2splat source + wheels/, see DEPLOY.md).
# Hardware: pick a GPU in the Space settings — A10G (24 GB) recommended; the
# GUI holds the dense and sparse checkpoints in VRAM at once.

# CUDA 12.8 + Ubuntu 22.04 — matches the wheels' build environment.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.12 (via deadsnakes) — learn2splat uses PEP 695 generic syntax that
# Ubuntu 22.04's stock Python 3.10 cannot parse. Also: build tools, extension
# headers (libglm-dev), and the OpenCV runtime libs (libgl1, libglib2.0-0 —
# learn2splat's COLMAP loader imports cv2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git build-essential ninja-build libglm-dev \
        libgl1 libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: run as a non-root user (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch
WORKDIR /home/user/app

# All Python work happens in a venv on PATH (no system-Python writes).
RUN python3.12 -m venv /home/user/venv
ENV PATH=/home/user/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8) — pinned to setup.sh.
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Python requirements (copied first so this layer caches across code edits).
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

# Prebuilt CUDA-extension wheels — gsplat, nerfacc, pycolmap, fused-ssim,
# simple-knn, pointops, fused_knn_attn, and the inria + fastgs rasterizers
# (diff_gaussian_rasterization[_fastgs], for the demo's inria/fastgs renderers).
# Built on a matching machine (see DEPLOY.md) so the HF builder never compiles
# CUDA and never OOMs. The wildcard install picks up whatever wheels are present.
COPY --chown=user:user wheels/ ./wheels/
RUN pip install --no-deps ./wheels/*.whl

# The learn2splat repo, then learn2splat itself (pure Python — editable install).
COPY --chown=user:user . .
RUN pip install --no-build-isolation --no-deps -e .

# gradio serves the GUI here — must equal app_port in README.md.
EXPOSE 7860

# gradio mode: the learn2splat decoder renders the live optimization on the GPU and
# gradio streams the frames as images (column 2); the finished splats load into
# an interactive Model3D viewer (column 3). demo.py binds 0.0.0.0 on this port.
CMD ["python", "demo.py", "--with-gui", "gradio", "--gui-port", "7860"]
