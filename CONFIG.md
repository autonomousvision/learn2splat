# Configuration

Learn2Splat is configured with [Hydra](https://hydra.cc/). Every run is composed from small
YAML fragments under [`learn2splat/config/`](learn2splat/config/) and the result is validated against the typed
dataclasses in [`learn2splat/config.py`](learn2splat/config.py) and
[`learn2splat/scene_trainer/scene_trainer_cfg.py`](learn2splat/scene_trainer/scene_trainer_cfg.py).

The entry point is always:

```bash
python -m learn2splat.main <overrides...>
```

This page explains how the config is structured, which building blocks you can swap in, and how to
override anything from the command line. For the full list of test-time output flags, see the
[Test output flags](README.md#test-output-flags) section of the README.

## How a run is composed

A run is assembled in two steps:

1. **Base config** — [`learn2splat/config/main.yaml`](learn2splat/config/main.yaml) sets the defaults and lists
   the *config groups* that make up a run (dataset, initializer, optimizer, decoder, loss, …).
   Each group points at one YAML file from its directory.
2. **Experiment** — an `+experiment=<name>` file under
   [`learn2splat/config/experiment/`](learn2splat/config/experiment/) overrides the groups and values for a
   concrete setup (which dataset, which optimizer, train vs. test, resolution, …).

For example,

```bash
# Evaluate on MipNeRF-360 with a pretrained checkpoint
python -m learn2splat.main +experiment=test_mipnerf360 \
  scene_trainer/scene_optimizer=learn2splat_dense \
  checkpointing.pretrained_optimizer=hf://autonomousvision/learn2splat/joint/checkpoints/epoch_7-step_150000.ckpt

# Train the sparse Learn2Splat optimizer on DL3DV
python -m learn2splat.main +experiment=train_l2s_sparse_dl3dv
```

### Available experiments

| Experiment | Mode | Purpose |
|---|---|---|
| `test_re10k` | test | Evaluate on RealEstate10K (512×960). |
| `test_dl3dv` | test | Evaluate on DL3DV. |
| `test_colmap` | test | Evaluate zero-shot on a COLMAP-style scene. |
| `test_mipnerf360` | test | `test_colmap` with the MipNeRF-360 roots, resolution and batch sizes. |
| `train_dl3dv` | train | Shared DL3DV training base (initializer + optimizer). |
| `train_mixed_sparse_dense` | train | Train one optimizer over both regimes at once (the released checkpoint). |
| `train_l2s_sparse_dl3dv` | train | Train the sparse Learn2Splat optimizer (checkpoint buffer + rollout). |
| `train_l2s_sparse_dl3dv_no_delta` | train | Ablation: no delta loss. |
| `train_l2s_sparse_dl3dv_no_loss` | train | Ablation: no auxiliary loss. |

Experiment files carry `# @package _global_`, so their keys are applied at the root of the config.
An experiment can also extend another one (e.g. `train_l2s_sparse_dl3dv` builds on `train_dl3dv`).

## Config groups

Each group is a directory under [`learn2splat/config/`](learn2splat/config/); the value you select is the file
name (without `.yaml`). The pipeline component is then looked up by name in a registry (the
`__init__.py` of the matching Python package). Select a group on the command line with a slash, e.g.
`scene_trainer/scene_optimizer=knn_based`.

### Dataset — `dataset=<name>`

[`learn2splat/config/dataset/`](learn2splat/config/dataset/): `re10k`, `dl3dv`, `scannet`, `colmap`, `megasynth`,
and `mixed` (draws scenes from several of the others, weighted by `sources`).
Each dataset composes a default **view sampler** from
[`dataset/view_sampler/`](learn2splat/config/dataset/view_sampler/) (`bounded`, `boundedv2`,
`boundedv2_360`, `evaluation`, `all`, `dense`, `arbitrary`, `ids`). Use the `evaluation` sampler with
a fixed index JSON (under `assets/`) for reproducible benchmarks:

```bash
dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_re10k.json
```

Key dataset fields: `image_shape`, `roots`, `near`/`far`, `view_sampler.num_context_views`,
`view_sampler.num_target_views`, `test_chunk_interval` (subsample the test set, e.g. `10` = 1/10th).

### Scene initializer — `scene_trainer/scene_initializer=<name>`

Produces the initial Gaussians from the input views.
[`scene_initializer/`](learn2splat/config/scene_trainer/scene_initializer/):

| Name | Initializer |
|---|---|
| `resplat_v1` / `resplat_v2` | Learned feed-forward (ReSplat). The default for Learn2Splat. |
| `colmap` | Structure-from-Motion point cloud. |
| `ply` | Load Gaussians from a `.ply`. |
| `pointcloud` | Load from a point cloud. |
| `depth` | Unproject the dataset's ground-truth depth maps. |
| `random` | Random initialization. |
| `edgs` | EDGS initializer. |

With `dataset=mixed` you can additionally set `scene_trainer.scene_initializers`, a map from source
key to initializer, so scenes from different sources are initialized differently while sharing one
optimizer.

### Scene optimizer — `scene_trainer/scene_optimizer=<name>`

Iteratively refines the Gaussians.
[`scene_optimizer/`](learn2splat/config/scene_trainer/scene_optimizer/):

| Name | Optimizer                                                                                                                                                 |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `learn2splat` | **The released Learn2Splat optimizer** (named `l2s` inside the configs).                                                                                  |
| `learn2splat_dense` | The same optimizer with a random instead of zero initial state, for COLMAP initialization.                         |
| `knn_based` | The kNN-based architecture that `learn2splat` and `resplat_v1`/`resplat_v2` are built on. Select it directly only to configure the architecture yourself. |
| `adam` | Adam baseline (vanilla 3DGS).                                                                                                                             |
| `adam_tuned` | The same baseline with grid-searched learning rates.                                                                                                      |
| `fastgs` | Adam baseline driving the FastGS refiner + decoder.                                                                                                       |
| `resplat_v1` / `resplat_v2` | ReSplat optimizer.                                                                                                                                        |
| `none` | Initialization only — no optimization.                                                                                                                    |

The learned optimizers compose an **`lr_scheduler`** sub-group
([`scene_optimizer/lr_scheduler/`](learn2splat/config/scene_trainer/scene_optimizer/lr_scheduler/):
`expon`, `ddim`, `none`) and a **`refiner`** sub-group
([`scene_optimizer/refiner/`](learn2splat/config/scene_trainer/scene_optimizer/refiner/): `default`,
`mcmc`, `edgs`, `none`) that controls adaptive density control (ADC). **ADC is experimental** — its
behavior is not guaranteed and we plan to improve it in a later stage of the release.

### Decoder — `scene_trainer/decoder=<name>`

Renders Gaussians to images. [`decoder/`](learn2splat/config/scene_trainer/decoder/):

- `gsplat` — gsplat backend (default).
- `inria` — optional alternative; needs `diff_gaussian_rasterization` installed.
- `fastgs` — optional alternative; needs the FastGS rasterizer installed.

### Loss — `loss=[<name>,...]`

A list, composed from [`loss/`](learn2splat/config/loss/): `mse`, `ssim`, `lpips`, `deltas`, `stability`,
`gaussians`, `iso_scales`, `sh0`, `sgd`, `render_depth`. Each loss carries its own `weight` (and loss-specific
fields, e.g. `lpips.apply_after_step`, `mse.l1_loss`). Compose and reweight them inline:

```bash
loss='[mse,lpips]' loss.lpips.weight=0.5
```

### Postprocessing — `meta_trainer/test/postprocessing=<name>`

An extra refinement applied **after** the learned optimizer at test time.
[`test/postprocessing/`](learn2splat/config/meta_trainer/test/postprocessing/): `none`, `adam`, `sgd`,
`vanilla_3dgs`, `vanilla_3dgs_sgd`.

### Checkpoint buffer — `meta_trainer/train/ckpt_buffer_cfg=<name>`

The few-shot checkpoint buffer used during meta-training.
[`train/ckpt_buffer_cfg/`](learn2splat/config/meta_trainer/train/ckpt_buffer_cfg/): `default`, `none`.
Setting `meta_trainer.train.use_ckpt_buffer=true` enables it (and forces
`ddp_find_unused_parameters_true` under DDP).

## Overriding from the command line

Hydra dot-notation overrides any field; prefix `+` to add a key not present in the defaults:

```bash
# change scalar values
scene_trainer.num_update_steps=2000  meta_trainer.test.save_render_image=true

# swap a whole config group
scene_trainer/scene_optimizer=adam  dataset/view_sampler=evaluation

# replace a list-valued group
loss='[mse,ssim,lpips]'
```

## Key top-level fields

These live at the root of the config (see [`learn2splat/config/main.yaml`](learn2splat/config/main.yaml) and the
`RootCfg` dataclass):

| Field | Meaning |
|---|---|
| `mode` | `train` or `test`. |
| `output_dir` | Where outputs are written. `placeholder` derives a path from the checkpoint. |
| `seed` | Global random seed. |
| `checkpointing.pretrained_model` | Full model checkpoint (initializer + optimizer); local path or `hf://…`. |
| `checkpointing.pretrained_optimizer` | Optimizer-only checkpoint. |
| `checkpointing.pretrained_initializer` | Initializer-only checkpoint. |
| `meta_optimizer.lr` | Meta-optimizer learning rate (training). |

### `scene_trainer` flags

| Field | Meaning |
|---|---|
| `num_update_steps` | Optimization iterations per scene (`0` = init-only). |
| `train_scene_opt` | Whether the optimizer has gradients enabled during meta-training. The initializer is always frozen. |
| `train_min_refine` / `train_max_refine` | Range of refinement steps sampled per scene during training. |
| `opt_batch_strategy` | Sub-batching of input views: `random`, `sequential`, `neighbors`, `fps`. |
| `sh_degree_interval` | Steps between SH-degree increments (`0` = disabled). |

## How checkpoint configs are merged

When a checkpoint is supplied **as a local path**, the architecture it was trained with is loaded
from the config saved next to it (`config.yaml`), so the model is rebuilt and rendered exactly as
trained. The merge strategy depends on the mode (see `merge_config_from_file` in
[`learn2splat/config.py`](learn2splat/config.py)):

- **Test** (`mode=test`): your CLI config is the base (dataset, test flags, …). Only the
  **optimizer/initializer architecture** and the **decoder** are patched in from the checkpoint. An
  explicit CLI override still wins.
- **Train** (resuming / fine-tuning): the checkpoint config takes priority for every field it
  defines (preserving the trained architecture); the CLI only fills in fields added since training.

Checkpoint source priority for the architecture is
`resume` > `pretrained_model` > `pretrained_optimizer` (with `pretrained_initializer` overriding the
initializer only).

For an `hf://` checkpoint, choose the architecture on the command line, e.g.
`scene_trainer/scene_optimizer=learn2splat_dense` for the released checkpoint on a COLMAP
initialization. The `config.yaml` downloaded next to the `.ckpt` records the settings it was trained
with.

## Config versioning

Saved checkpoint configs are tagged with a `version`, which
[`learn2splat/config_migrate.py`](learn2splat/config_migrate.py) checks on load against `CURRENT_CFG_VERSION`.
The released checkpoints are at version 2.5. Fresh runs from `main.yaml` are always at the
current version, so you can ignore this unless you change the config schema.