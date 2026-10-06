#!/usr/bin/env bash
# =============================================================================
# submit_train_and_eval.sh — submit training, then auto-queue the eval gated on it
# =============================================================================
# Run from the LOGIN node. For each training script it:
#   1. submits training and captures the SLURM job id (--parsable),
#   2. reads output_dir / max_steps / init from the training script to build the
#      eval's ckpt entry  <rel_dir>|checkpoints/*-step_<max_steps>.ckpt|<init>
#      (the "*" stands in for the epoch number, resolved when the eval runs),
#   3. queues scripts/testing/launch_all.sh with EVAL_DEPENDENCY=afterok:<id>...
#      so the eval waits for training and is cancelled if training fails.
# Only the just-trained checkpoint(s) are evaluated; baselines are skipped.
#
# Usage:
#   bash scripts/training/submit_train_and_eval.sh scripts/training/learn2splat_dense.sh
#   bash scripts/training/submit_train_and_eval.sh \
#        scripts/training/learn2splat_dense.sh scripts/training/learn2splat_sparse.sh
#   bash scripts/training/submit_train_and_eval.sh <script> --resume   # resume training
#   bash scripts/training/submit_train_and_eval.sh <script> --no-eval  # train only, no eval
#
# Override the eval via env (passed to launch_all.sh): DATASETS, ROW,
# OURS_RUNS_OVERRIDE, BASELINE_EXPERIMENTS_OVERRIDE.
# =============================================================================
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

resume_flag=""
do_eval=1
train_scripts=()
for arg in "$@"; do
  case "$arg" in
    --resume)  resume_flag="--resume" ;;
    --no-eval) do_eval=0 ;;
    *)         train_scripts+=("$arg") ;;
  esac
done
[[ ${#train_scripts[@]} -eq 0 ]] && { echo "usage: $0 <train_script> [<train_script> ...] [--resume] [--no-eval]" >&2; exit 1; }

# Build the eval ckpt entry from a training script: rel_dir|ckpt_glob|init_state
derive_entry() {
  local script="$1" out_dir max_steps init
  out_dir=$(grep -oE "output_dir=['\"]?[^'\" ]+" "$script" | head -1 | sed -E "s/^output_dir=['\"]?//")
  max_steps=$(grep -oE "meta_trainer\.max_steps=[0-9_]+" "$script" | head -1 | sed -E 's/.*max_steps=//; s/_//g')
  if grep -qE "scene_initializer=colmap" "$script"; then init="false"; else init="true"; fi
  [[ -z "$out_dir" || -z "$max_steps" ]] && { echo "[derive] could not read output_dir/max_steps from $script" >&2; return 1; }
  echo "${out_dir#checkpoints/}|checkpoints/*-step_${max_steps}.ckpt|${init}"
}

dep="afterok"
ours=""
mkdir -p logs   # the #SBATCH --output paths are under logs/

for ts in "${train_scripts[@]}"; do
  jid=$(sbatch --parsable "${REPO_ROOT}/scripts/training/submit_train.slurm" "$ts" $resume_flag)
  echo "submitted training job ${jid}  (${ts})"
  dep="${dep}:${jid}"
  entry=$(derive_entry "$ts") && ours="${ours:+${ours}:}${entry}"
done

if [[ $do_eval -eq 0 ]]; then echo "--no-eval: skipping eval"; exit 0; fi

echo "queuing eval  dependency=${dep}"
echo "  ours: ${ours}"
# Default eval: dl3dv ROW 0 (8-low) + mipnerf360 (dense; ignores ROW), antialiased decoder.
DATASETS="${DATASETS:-dl3dv mipnerf360}" \
  ROW="${ROW:-0}" \
  DECODER="${DECODER:-antialiased}" \
  OURS_RUNS_OVERRIDE="${OURS_RUNS_OVERRIDE:-$ours}" \
  BASELINE_EXPERIMENTS_OVERRIDE="${BASELINE_EXPERIMENTS_OVERRIDE:-none}" \
  EVAL_DEPENDENCY="${dep}" \
  bash "${REPO_ROOT}/scripts/testing/launch_all.sh"