# Test-eval metrics

- For the **runtime/memory** numbers (`*_ms`, `peak_vram_mb`, the per-step
`time` curve, and GPU warm-up handling) see [TIMINGS.md](TIMINGS.md). 
- Relevant code is in `meta_trainer.on_test_end` / `meta_test_step` and `scripts/testing/aggregate_metrics.py`.

The number to report is in `<out>/<opt>/metrics/averaged/<input>_<opt>.json`, which
`aggregate_metrics.py` builds by merging the per-scene files. Both `launch.sh` paths produce it, as a
dependent job under SLURM and inline after a local run; after a direct `python -m learn2splat.main` run,
run `aggregate_metrics.py` yourself.

`<out>/metrics/<input>_<opt>_<metric>.json` (shape `[scenes, steps]`) holds only the scenes one
process saw, and every process writes that same path, so under a SLURM array it ends up holding the
chunk of whichever job finished last.

- `<out>` — the run's output dir. For the dl3dv evaluation scripts:
  `BASE_OUT/<rel_dir>/<OUT_SUFFIX>_<step>[_speedup]` (see
  `_common/out_dirs.sh::ours_out_dir`).
- `<opt>` — the optimizer class name, lowercased: `learn2splatoptimizer`,
  `adamoptimizer`, … (this is also a directory level and a metric-key prefix).
- `<input>` — `target` (novel views; the headline metric) or `context` (input
  views; only when `meta_trainer.test.eval_context_views=true`).
- `<metric>` — `psnr`, `ssim`, `alex_lpips`, `vgg_lpips`, `iterations`, `time`,
  `gaussians`, `nonzero_grads`.

## Three locations for `.json`s

### 1. Per-process split files (`on_test_end`, needs `compute_scores=true`)
```
<out>/metrics/<input>_<opt>_<metric>.json        e.g. metrics/target_learn2splatoptimizer_psnr.json
```
- A nested list of shape **`[scenes, steps]`** (the scenes this process
  saw), NaN-padded when scenes stopped early (see [Partial scenes](#partial-scenes-early-stopped-optimization)).
  `scenes` = number evaluated in this job; `steps` = number of saved
  optimization checkpoints (set by `meta_trainer.test.save_every_steps` /
  `save_every_freq`).
- Covers every scene of a single-process run; only one chunk under a SLURM array.

### 2. Per-scene files (always written; drive the skip-if-exists check)
```
<out>/<opt>/metrics/<scene_hash>/<input>_<opt>.json
                                  e.g. .../<scene>/target_learn2splatoptimizer.json
```
- Content: a **dict** keyed by `<opt>_<metric>`, each a list of shape **`[steps]`**
  (this one scene), plus scalar benchmark keys:
  ```json
  {
    "learn2splatoptimizer_psnr":   [ ... 19 values ... ],
    "learn2splatoptimizer_ssim":   [ ... ],
    "learn2splatoptimizer_alex_lpips": [ ... ],
    "learn2splatoptimizer_vgg_lpips":  [ ... ],
    "learn2splatoptimizer_iterations": [ ... ],
    "learn2splatoptimizer_time":       [ ... ],   // cumulative ms
    "learn2splatoptimizer_gaussians":  [ ... ],   // count over steps
    "learn2splatoptimizer_nonzero_grads": [ ... ],
    "peak_vram_mb": 903.77,        // scalar (whole-scene benchmark)
    "decoder_ms": 2365.8,          // scalar
    "optimizer_ms": 4080.6,        // scalar
    "optimizer_net_ms": 6446.9,    // scalar
    "scene_start_ms": 0.48         // scalar
  }
  ```
- All array tasks write into the **same** `<opt>/metrics/` tree (one subdir per
  scene hash), so after the whole array finishes this directory holds every
  scene — this is what makes cross-job aggregation possible.
- With `meta_trainer.test.eval_initialization=true` the same dict also carries the
  initializer's keys (`<init>_psnr`, …, each a length-1 list — the initializer is a
  single step), so the optimizer's gain over the initialization is in one file.

### 3. Aggregated across scenes (`aggregate_metrics.py`)
```
<out>/<opt>/metrics/averaged/<input>_<opt>.json
```
- Produced by `python scripts/testing/aggregate_metrics.py <out>`. The `launch.sh` scripts run it automatically (`_common/aggregate.sh`) as an `afterok` dependent
  job under SLURM, or right after the test returns in local mode. After a direct `python -m
  learn2splat.main` run, run it yourself.
- Stacks every per-scene dict (layer 2) and averages over scenes per key:
  ```json
  { "num_scenes": 140,
    "learn2splatoptimizer_psnr": [ ... mean-over-scenes per step, shape [steps] ... ],
    "learn2splatoptimizer_ssim": [ ... ],
    "...": "...",
    "peak_vram_mb": <mean scalar>, "decoder_ms": <mean scalar>, ... }
  ```
- Per-step arrays stay `[steps]` averaged over scenes.
- Per-scene scalars become a single mean scalar.

## Partial scenes (early-stopped optimization)

A scene can stop before the last saved step when its optimization fails/diverges.
The code still keeps the inner steps completed (that scene's per-step row is shorter than a
fully-completed scene).

Both the per-process reduction (`MetaTrainer._reduce_partial_metric`) and the
cross-scene aggregation (`aggregate_metrics.py`) handle this **identically**:

1. NaN-pad every row to the longest (`[scenes, max_steps]`).
2. Take a **NaN-propagating** mean over scenes, so the per-step mean is `NaN` at any step
   not reached by every scene (not comparable).
3. Record how many scenes reached each step: `num_scenes_per_step` (written once
   alongside the averaged arrays) and the `scenes_per_step`.

## Reading the results

[`scripts/show_results_table.py`](../show_results_table.py) prints these files as a table. Pass one run directory for metrics across all iterations, or several to
compare them scene by scene at one iteration:

```bash
python scripts/show_results_table.py results/mipnerf360
python scripts/show_results_table.py <run_A> <run_B> --iter 2000 --labels ours adam
```

It reads the per-scene files (#2) and averages them itself, so it works whether or not
`aggregate_metrics.py` has run.

To read the numbers directly:

```python
import json, numpy as np
d = json.load(open(".../averaged/target_learn2splatoptimizer.json"))
n = d["num_scenes"]
psnr = np.asarray(d["learn2splatoptimizer_psnr"])[-1]   # final-step, scene-averaged
ssim = np.asarray(d["learn2splatoptimizer_ssim"])[-1]
lpips = np.asarray(d["learn2splatoptimizer_vgg_lpips"])[-1]
```