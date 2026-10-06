# Testing Scripts

Batch evaluation scripts used in the paper (RealEstate10K, DL3DV, MipNeRF360, DTU, LLFF, custom COLMAP
scenes) — runs our checkpoints + baselines across scenes and aggregates the metrics. For a single direct
run, use the `python -m learn2splat.main +experiment=test_<dataset> mode=test …` command in the top-level
[README](../../README.md#evaluation) instead.

## Run

The scripts are built for **SLURM** (submit job arrays). If `sbatch` isn't available they
automatically run the first job locally with `bash`.

```bash
# from the repo root run
# <VAR>=<VALUE> bash scripts/testing/<dataset>/launch.sh
NUM_STEPS=100 bash scripts/testing/mipnerf360/launch.sh
```

available `<dataset>`: `dl3dv | re10k | mipnerf360 | dtu | llff | custom`.

- **With SLURM:** submits a job array, then an aggregation job that depends on it.
- **Locally:** runs `bash test.sh` for task 0 of each method, then aggregates once it returns. This
  happens automatically when `sbatch` is missing, or with `RUN_BASH=1` to force it even where
  `sbatch` exists.
- **Dry run:** `DRY_RUN=1` prints what would be submitted/run without doing it.

## Configure

- **`<dataset>/launch.sh`** — the only file you normally edit: adding/removing checkpoints (`OURS_RUNS`), baselines
  (`BASELINE_EXPERIMENTS`), and the evaluation axes (`SCENE_NUM`, `NUM_STEPS`, `OPT_BATCH_SIZE`, `INIT`, …).
- **`<dataset>/test.sh`** — the dataset's Hydra config + the SLURM directives.
- **`_common/`** — shared submission/run engines, opts builders, output-dir paths, and aggregation. Each
  file is documented inline.

## Outputs

Written under each run's `output_dir`. See [METRICS.md](METRICS.md) (quality metrics + JSON layout) and
[TIMINGS.md](TIMINGS.md) (runtime/memory + GPU warm-up).