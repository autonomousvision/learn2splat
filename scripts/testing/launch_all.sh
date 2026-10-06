#!/usr/bin/env bash
# =============================================================================
# launch_all.sh — run every dataset's evaluation in one go.
#
# Each dataset's launch.sh sweeps its own settings matrix (see <dataset>/launch.sh)
# and submits its SLURM array(s) + dependent aggregation job.
#
# Examples:
#   bash scripts/testing/launch_all.sh                       # all datasets
#   DATASETS="dl3dv dtu" bash scripts/testing/launch_all.sh  # a subset
#   DRY_RUN=1 bash scripts/testing/launch_all.sh             # preview the full fan-out
#
# Chain after training (Pattern A — submit everything from the LOGIN node; the
# eval arrays sit pending in the queue and only run once training succeeds, so
# the just-trained checkpoint exists by the time they start):
#   DENSE_ID=$(sbatch  --parsable scripts/training/submit_train.slurm scripts/training/learn2splat_dense.sh)
#   SPARSE_ID=$(sbatch --parsable scripts/training/submit_train.slurm scripts/training/learn2splat_sparse.sh)
#   EVAL_DEPENDENCY="afterok:${DENSE_ID}:${SPARSE_ID}" bash scripts/testing/launch_all.sh
#
# To evaluate ONLY the freshly-trained checkpoints (skip the archived runs and
# baselines), override the run lists from the environment:
#   NEW="learn2splat/dl3dv/sparse_8views_resplat_init|checkpoints/epoch_10-step_100000.ckpt|true"
#   NEW="${NEW}:learn2splat/dl3dv/dense_64views_sfm_init|checkpoints/epoch_10-step_100000.ckpt|false"
#   OURS_RUNS_OVERRIDE="$NEW" BASELINE_EXPERIMENTS_OVERRIDE=none \
#     EVAL_DEPENDENCY="afterok:${DENSE_ID}:${SPARSE_ID}" bash scripts/testing/launch_all.sh
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common/pretty_print.sh"

DATASETS=(${DATASETS:-dl3dv mipnerf360 dtu llff re10k})

for d in "${DATASETS[@]}"; do
  launch="${SCRIPT_DIR}/${d}/launch.sh"
  [[ -f "$launch" ]] || { print_warn "skip ${d}: no ${launch}"; continue; }
  print_header "launch_all → ${d}"
  bash "$launch"
done