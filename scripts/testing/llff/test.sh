#!/usr/bin/env bash
# =============================================================================
# llff/test.sh  — Evaluate models on LLFF (Local Light Field Fusion)
# =============================================================================
# Edit launch.sh to change checkpoints, baselines, and axes.
# This script contains only LLFF-specific config — shared run logic
# lives in _common/run_experiments.sh.
# =============================================================================

#SBATCH --job-name=llff-test
#SBATCH --partition=***   # set your cluster GPU partition(s)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1
# #SBATCH --exclude=***   # optional: blacklist specific nodes
#SBATCH --output=logs/%A_%a_llff_test.out
#SBATCH --error=logs/%A_%a_llff_test.err

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
SCENE_NUM="${SCENE_NUM:-8}"
SCENES_PER_JOB="${SCENES_PER_JOB:-8}"
NUM_STEPS="${NUM_STEPS:-2000}"
OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
SUBSAMPLE="${SUBSAMPLE:-4}"
TARGET_EVERY="${TARGET_EVERY:-8}"
METHOD="${METHOD:-ours}"
DATASET_ROOT="${DATASET_ROOT:-datasets/nerf_llff_data}"
DENSE_COLMAP_PATH="${DENSE_COLMAP_PATH:-datasets/nerf_llff_data}"
EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"
NORMALIZE="${NORMALIZE:-true}"
INIT="${INIT:-dense_colmap}"
VIEW_SAMPLER="${VIEW_SAMPLER:-dense}"
NUM_VIEWS="${NUM_VIEWS:--1}"
INDEX_PATH="${INDEX_PATH:-}"

# ---------------------------------------------------------------------------
# ② OUTPUT ROOT
# ---------------------------------------------------------------------------
# BASE_OUT should be declared in launch.sh
#[[ "$OPT_BATCH_SIZE" == "-1" ]] && BS_STR="all" || BS_STR="${OPT_BATCH_SIZE}"
#BASE_OUT="results/llff/${BS_STR}_all_${SCENE_NUM}scenes"

# ---------------------------------------------------------------------------
# ③ COMMON Hydra overrides
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/common_opts.sh"   # populates COMMON_OPTS with shared opts

# llff-specific additions
COMMON_OPTS+=(
  # Experiment
  "+experiment=test_colmap"

  # Dataset
  "data_loader.test.num_workers=0"
  "dataset.roots=${DATASET_ROOT}"
  "dataset.subsample_factor=${SUBSAMPLE}"
  "dataset.normalize_world_space=${NORMALIZE}"

  # --- test flags ---
  "meta_trainer.test.metrics_batch_size=4"
  # "meta_trainer.test.save_every_freq=[2,10,100,200]"
  # "meta_trainer.test.save_every_steps=[0,10,100,1000]"
  "meta_trainer.test.save_at_iters=[0,2,4,6,8,10,20,30,40,50,100,500,1000,2000,3000,4000,5000]"
  "meta_trainer.test.eval_context_views=${EVAL_CONTEXT_VIEWS}"
  "meta_trainer.test.save_render_image=false"
)

# ---------------------------------------------------------------------------
# View sampler
# ---------------------------------------------------------------------------
if [[ "$VIEW_SAMPLER" == "dense" ]]; then
  COMMON_OPTS+=(
    "dataset/view_sampler=dense"
    "dataset.view_sampler.target_every=${TARGET_EVERY}"
    "dataset.view_sampler.num_context_views=-1"
    "dataset.view_sampler.num_target_views=-1"
  )
else
  COMMON_OPTS+=( "dataset/view_sampler=${VIEW_SAMPLER}" )
  [[ "$NUM_VIEWS" != "-1" ]] && COMMON_OPTS+=( "dataset.view_sampler.num_context_views=${NUM_VIEWS}" )
  [[ -n "$INDEX_PATH"     ]] && COMMON_OPTS+=( "dataset.view_sampler.index_path=${INDEX_PATH}" )
fi

# ---------------------------------------------------------------------------
# ④ init opts
# ---------------------------------------------------------------------------
# resplat encoder requires H and W divisible by shim_patch_size*downscale_factor=64
RESPLAT_EXTRA_OPTS=( "dataset.crop_size=64" )
DENSE_COLMAP_EXTRA_OPTS=(
  "scene_trainer.scene_initializer.normalize_world_space=${NORMALIZE}"
  "scene_trainer.scene_initializer.filter_zero_rgb=false"  # Important, some scenes are only black gaussians
  "scene_trainer.scene_initializer.dl3dv_settings=false"
)

source "${COMMON_DIR}/init_opts.sh"

# ---------------------------------------------------------------------------
# ⑤ RUN (shared logic)
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/run_experiments.sh"

