#!/usr/bin/env bash
# =============================================================================
# re10k/test.sh  — Evaluate models on RE10K
# =============================================================================
# Edit launch.sh to change checkpoints, baselines, and axes.
# This script contains only RE10K-specific config — shared run logic
# lives in _common/run_experiments.sh.
# =============================================================================

#SBATCH --job-name=re10k-test
#SBATCH --partition=***   # set your cluster GPU partition(s)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1
# #SBATCH --exclude=***   # optional: blacklist specific nodes
#SBATCH --output=logs/%A_%a_re10k_test.out
#SBATCH --error=logs/%A_%a_re10k_test.err

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
COMMON_DIR="${REPO_ROOT}/scripts/testing/_common"
source "${COMMON_DIR}/env_setup.sh"
source "${COMMON_DIR}/pretty_print.sh"

# ---------------------------------------------------------------------------
# ① EVALUATION AXES
# ---------------------------------------------------------------------------
NUM_VIEWS="${NUM_VIEWS:-8}"
INIT="${INIT:-resplat}"
SCENE_NUM="${SCENE_NUM:-72}"
TEST_CHUNK_INTERVAL="${TEST_CHUNK_INTERVAL:-25}"
SCENES_PER_JOB="${SCENES_PER_JOB:-100}"
NUM_STEPS="${NUM_STEPS:-1000}"
OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
METHOD="${METHOD:-ours}"
DATASET_ROOT="${DATASET_ROOT:-datasets/re10k/720p_chunks}"
EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"

# ---------------------------------------------------------------------------
# ② INDEX PATH  (derived from NUM_VIEWS)
# ---------------------------------------------------------------------------
case "$NUM_VIEWS" in
  8)  INDEX_PATH="assets/re10k_start_0_distance_200_ctx_8v_tgt_8v.json" ;;
  *)
    echo "[test.sh] Unknown NUM_VIEWS: ${NUM_VIEWS}. Expected 4, 6, or 8." >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# ③ COMMON Hydra overrides
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/common_opts.sh"   # populates COMMON_OPTS with shared opts

# re10k-specific additions
COMMON_OPTS+=(
  # ensures test_start_idx scene-counter works correctly across SLURM array jobs
  "data_loader.test.num_workers=0"

  # Experiment
  "+experiment=test_re10k"

  # Dataset
  "dataset.roots=[${DATASET_ROOT}]"
  "dataset.highres=true"
  "dataset.test_chunk_interval=${TEST_CHUNK_INTERVAL}"

  # View sampler
  "dataset/view_sampler=evaluation"
  "dataset.view_sampler.num_context_views=${NUM_VIEWS}"
  "dataset.view_sampler.index_path=${INDEX_PATH}"

  # --- test flags ---
  # Save frequency matched to dl3dv/test.sh (dense ~29-checkpoint schedule) so the
  # timing curves are comparable across sparse datasets.
  "meta_trainer.test.save_every_freq=[2,10,100,200]"
  "meta_trainer.test.save_every_steps=[0,10,100,1000]"
  "meta_trainer.test.save_at_iters=null"
  # "meta_trainer.test.save_at_iters=[0,4,10,100,1000,2000]"  # old re10k schedule (6 checkpoints)
  "meta_trainer.test.eval_context_views=${EVAL_CONTEXT_VIEWS}"
  "meta_trainer.test.save_render_image=false"
)

# ---------------------------------------------------------------------------
# ④ INIT overrides (appends to COMMON_OPTS)
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/init_opts.sh"  # For now, we only support resplat init

# ---------------------------------------------------------------------------
# ⑤ RUN (shared logic)
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/run_experiments.sh"

