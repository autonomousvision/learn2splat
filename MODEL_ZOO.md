# Model Zoo

Pretrained **Learn2Splat** checkpoints, hosted on [Hugging Face 🤗](https://huggingface.co/autonomousvision/learn2splat). See the [README](README.md) for the method.

| Model | Regime | Checkpoint |
| --- | --- | --- |
| `Learn2Splat` | Sparse and dense views | [`joint/checkpoints/epoch_7-step_150000.ckpt`](https://huggingface.co/autonomousvision/learn2splat/blob/main/joint/checkpoints/epoch_7-step_150000.ckpt) |
| `ReSplat` initializer | Feed-forward init for the sparse regime | [`resplat_init/checkpoints/epoch_20-step_100000.ckpt`](https://huggingface.co/autonomousvision/learn2splat/blob/main/resplat_init/checkpoints/epoch_20-step_100000.ckpt) |

`Learn2Splat` is one optimizer trained jointly over both regimes, so the same weights serve the
sparse and dense settings. Pair it with `scene_trainer/scene_optimizer=learn2splat_dense` on a COLMAP
initialization (see the README's evaluation example).

> **Note:** the `ReSplat` initializer is an *earlier* version of ReSplat: its results reproduce our arXiv numbers and are lower than the final ReSplat paper. We release the exact checkpoint the released model was trained against; updating the code to use ReSplat's final checkpoint is planned.