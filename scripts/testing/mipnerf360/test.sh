#!/usr/bin/env bash
# =============================================================================
# mipnerf360/test.sh  — Evaluate models on MipNeRF-360
# =============================================================================
# Edit launch.sh to change checkpoints, baselines, and axes.
# This script contains only MipNeRF-360-specific config — shared run logic
# lives in _common/run_experiments.sh.
# =============================================================================

#SBATCH --job-name=mipnerf360-test
#SBATCH --partition=***   # set your cluster GPU partition(s)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1
# #SBATCH --exclude=***   # optional: blacklist specific nodes
#SBATCH --output=logs/%A_%a_mipnerf360_test.out
#SBATCH --error=logs/%A_%a_mipnerf360_test.err

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
SCENE_NUM="${SCENE_NUM:-9}"
SCENES_PER_JOB="${SCENES_PER_JOB:-9}"
NUM_STEPS="${NUM_STEPS:-5000}"
OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
SUBSAMPLE="${SUBSAMPLE:-4}"
HIGH_SUBSAMPLE_SCENES="${HIGH_SUBSAMPLE_SCENES:-}"
SUBSAMPLE_HIGH="${SUBSAMPLE_HIGH:-${SUBSAMPLE}}"
TARGET_EVERY="${TARGET_EVERY:-8}"
# Views per render call for the init and saved-step renders. MipNeRF-360 scenes carry a few
# hundred views, and rasterizing them all at once needs one buffer of
# views x H x W x 4 channels x 4 bytes -- 6.2 GiB for bonsai at subsample 2, past what a single
# allocation can serve. Chunking costs only extra kernel launches on renders that are not timed.
ITER_BATCH_SIZE="${ITER_BATCH_SIZE:-32}"

# Scenes in HIGH_SUBSAMPLE_SCENES are downscaled by SUBSAMPLE_HIGH, every other scene by
# SUBSAMPLE. The split is handed to the dataset as one map so all scenes still run in a single
# process. Giving each scene its own process would also give each its own warm-up discard
# (_zero_first_iter_warmup zeroes the first iteration of the first scene only), which none of
# the other datasets get, so the per-scene runtimes would not be comparable.
SUBSAMPLE_MAP_OPTS=()
if [[ -n "${HIGH_SUBSAMPLE_SCENES}" ]]; then
  _entries=()
  for s in ${HIGH_SUBSAMPLE_SCENES}; do _entries+=("${s}:${SUBSAMPLE_HIGH}"); done
  SUBSAMPLE_MAP_OPTS=("dataset.subsample_factor_per_scene={$(IFS=,; echo "${_entries[*]}")}")
fi
METHOD="${METHOD:-ours}"
DATASET_ROOT="${DATASET_ROOT:-datasets/mipnerf360}"
DENSE_COLMAP_PATH="${DENSE_COLMAP_PATH:-datasets/mipnerf360}"
EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"
INIT="${INIT:-dense_colmap}"
PLY_PATH="${PLY_PATH:-}"
PLY_FILENAME="${PLY_FILENAME:-gaussians.ply}"
PLY_EXTRA_OPTS=()
VIEW_SAMPLER="${VIEW_SAMPLER:-dense}"
NUM_VIEWS="${NUM_VIEWS:--1}"
INDEX_PATH="${INDEX_PATH:-}"

# ---------------------------------------------------------------------------
# ② OUTPUT ROOT
# ---------------------------------------------------------------------------
# BASE_OUT should be declared in launch.sh
#[[ "$OPT_BATCH_SIZE" == "-1" ]] && BS_STR="bsall" || BS_STR="bs${OPT_BATCH_SIZE}"
#BASE_OUT="results/mipnerf360/${BS_STR}_${SCENE_NUM}scenes"

# ---------------------------------------------------------------------------
# ③ COMMON Hydra overrides
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/common_opts.sh"   # populates COMMON_OPTS with shared opts

# mipnerf360-specific additions
COMMON_OPTS+=(
  # Experiment
  "+experiment=test_colmap"

  # Dataset
  "data_loader.test.num_workers=0"
  "dataset.roots=${DATASET_ROOT}"
  "dataset.subsample_factor=${SUBSAMPLE}"
  "${SUBSAMPLE_MAP_OPTS[@]}"
  "dataset.normalize_world_space=false"

  # Render the full view set in chunks (see ITER_BATCH_SIZE above)
  "scene_trainer.iter_batch_size=${ITER_BATCH_SIZE}"

  # --- test flags ---
  "meta_trainer.test.metrics_batch_size=4"
  "meta_trainer.test.save_render_image=false"
  "meta_trainer.test.save_at_iters=[0,2,4,6,8,10,20,30,40,50,100,500,1000,2000,3000,4000,5000]"
  "meta_trainer.test.eval_context_views=${EVAL_CONTEXT_VIEWS}"
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
  "scene_trainer.scene_initializer.normalize_world_space=false"
  # Should be default when setting scene_initializer=colmap, but set explicitly to be safe
  "scene_trainer.scene_initializer.filter_zero_rgb=false"
  "scene_trainer.scene_initializer.dl3dv_settings=false"
)
source "${COMMON_DIR}/init_opts.sh"

# ---------------------------------------------------------------------------
# ⑤ RUN (shared logic)
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/run_experiments.sh"
