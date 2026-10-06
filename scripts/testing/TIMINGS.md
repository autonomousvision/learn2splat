# Test-eval timings: what is measured, units, and GPU warm-up

- Quality metrics are explained in [METRICS.md](METRICS.md).
- Code:
  - Per-iteration timing: `learn2splat/scene_trainer/optimizer/optimizer.py`.
  - Per-scene scalars + warm-up zeroing: `MetaTrainer._run_optimizer` and
    `MetaTrainer._zero_first_iter_warmup` in `learn2splat/meta_trainer/meta_trainer.py`.
  - Per-step curve: `MetaTrainer._compute_and_store_metrics` in the same file.
  - Timing primitive: `Benchmarker` in `learn2splat/misc/benchmarker.py`.

## Two benchmarkers

Two instances of the `Benchmarker` class:

- **Per scene** (`Optimizer.benchmarker`, and the same on `PostProcessing3DGS`): the per-iteration
  timings and stats, plus the one-off scene setup. Reset at the start of each scene.
- **Per test pass** (`MetaTrainer.benchmarker`, shared with `SceneTrainer`): one entry per scene per
  tag, covering the initializer, the init render and the per-scene scalars below. This is the one
  written to `benchmark.json`.

## How an iteration is timed

- Each optimization iteration is bracketed with `torch.cuda.Event`s and read after a
  `torch.cuda.synchronize()`. The numbers are real GPU time, not async launch time.
- Each iteration `i` splits into three values:
  - `decoder_time_log[i]`: ms rendering the Gaussians to get gradients (the gsplat decoder).
  - `optimizer_time_log[i]`: ms in the update step. Equals `iter_time_log[i] - decoder_time_log[i]`.
  - `iter_time_log[i]`: total ms for the iteration (decoder + update).
- `scene_start_ms`: ms for `on_scene_start`, measured once per scene. Covers Adam/state init and
  optimizer preprocessing. It does not cover the KNN lookup.
- The KNN lookup runs inside the optimization iterations, not in `on_scene_start`. Each scene
  rebuilds it on its first iteration and every `knn_idx_update_every` steps, so its cost lands in
  `iter_time_log`.
- The extra `save_every` renders (views written when saving images/videos) run after the timed
  region. They are in none of these logs.

## GPU warm-up handling

- The first optimization iteration of the test pass pays one-time GPU costs: kernel JIT, cuDNN
  autotune, CUDA context setup.
- `_zero_first_iter_warmup` zeros `iter[0]` and `decoder[0]` for the first optimized scene only.
- A flag (`_timing_warmup_done`, reset in `on_test_epoch_start`) ensures this happens once per pass.
- The per-scene scalars and the per-step curve both read the same optimizer benchmarker, so both
  are warm-up-clean for the first scene.
- Later scenes keep their full per-iteration timings, including their `iter[0]` (which holds that
  scene's KNN rebuild).
- `scene_start_ms` is not zeroed. It is recurring per-scene setup, not one-time warm-up.

## Per-step cumulative time — `<opt>_time`

- File: `<out>/<opt>/metrics/<scene>/<input>_<opt>.json` (METRICS.md layer 2).
- The `<opt>_time` array is cumulative ms up to each saved step: `sum(iter_time_log[:step])`.
- It is the time axis for time-vs-quality plots. It is paired step-for-step with `<opt>_psnr`,
  `<opt>_ssim`, etc.
- Read `[-1]` for the final-step time.

## Per-scene scalars

Written per scene into both the per-scene JSON and the run's `<out>/benchmark.json`; `peak_vram_mb`
also goes to `<out>/peak_memory.json`.

| Key | Meaning | Units |
|---|---|---|
| `scene_start_ms` | `on_scene_start`: Adam/state init, preprocessing | ms |
| `decoder_ms` | `sum(decoder_time_log)` over the scene's iterations | ms |
| `optimizer_ms` | `sum(optimizer_time_log)` over the scene's iterations | ms |
| `optimizer_net_ms` | `scene_start_ms + decoder_ms + optimizer_ms` | ms |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated` | MiB |

`optimizer_net_ms` is the single per-scene compute number. `aggregate_metrics.py` averages
each scalar over scenes, including `peak_vram_mb`.

## Train-time eval (separate path)

During training, `MetaTrainer.run_full_test_sets_eval` reports averaged runtimes to wandb, handling
warm-up as `mode=test` does.
