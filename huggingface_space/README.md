---
title: Learn2Splat
emoji: 🪴
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive demo of the Learn2Splat learned 3DGS optimizer
---

# Learn2Splat — interactive demo

A learned optimizer for 3D Gaussian Splatting. This Space SfM-initializes a
COLMAP scene and refines the Gaussians live in your browser: pick the
Learn2Splat optimizer (dense or sparse checkpoint) or a 3DGS Adam baseline,
press **Start**, and watch the decoder render converge — then explore the
finished splats in an interactive 3D viewer.

Runs `demo.py --with-gui gradio` from the
[Learn2Splat repository](https://github.com/autonomousvision/learn2splat);
the optimization is rendered on the GPU and streamed by gradio, and the
result loads into a `Model3D` splat viewer.

> Requires GPU hardware. The demo holds two checkpoints in VRAM at once —
> an A10G (24 GB) is recommended.
