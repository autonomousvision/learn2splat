#!/usr/bin/env bash
# =============================================================================
# dl3dv/test.sh  — Evaluate models on DL3DV
# =============================================================================
# Edit launch.sh to change checkpoints, baselines, and axes.
# This script contains only DL3DV-specific config — shared run logic lives in
# _common/run_experiments.sh.
# =============================================================================

#SBATCH --job-name=dl3dv-test
#SBATCH --partition=***   # set your cluster GPU partition(s)
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --gres=gpu:1
# #SBATCH --exclude=***   # optional: blacklist specific nodes
#SBATCH --output=logs/%A_%a_dl3dv_test.out
#SBATCH --error=logs/%A_%a_dl3dv_test.err

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
# ① EVALUATION AXES  (defaults; override via export before sbatch/bash)
# ---------------------------------------------------------------------------
NUM_VIEWS="${NUM_VIEWS:-8}"
RESOLUTION="${RESOLUTION:-low_res}"   # low_res | high_res
INIT="${INIT:-resplat_v1}"
DENSE_COLMAP_PATH="${DENSE_COLMAP_PATH:-datasets/dl3dv-benchmark}"
SPARSE_COLMAP_PATH="${SPARSE_COLMAP_PATH:-datasets/dl3dv_start_0_distance_40_ctx_8v_tgt_8v_256x448}"
SCENE_NUM="${SCENE_NUM:-140}"
SCENES_PER_JOB="${SCENES_PER_JOB:-140}"
NUM_STEPS="${NUM_STEPS:-2000}"
OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
METHOD="${METHOD:-ours}"
EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"

# Validate sparse colmap is used with 8 views low res (only combination supported right now)
if [[ "$INIT" == "sparse_colmap" ]]; then
  if [[ "$NUM_VIEWS" != "8" || "$RESOLUTION" != "low_res" ]]; then
    echo "[test.sh] sparse_colmap init is only supported with NUM_VIEWS=8 and RESOLUTION=low_res (for now)" >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# ② INDEX PATH + RESOLUTION OPTS  (derived from NUM_VIEWS + RESOLUTION)
# ---------------------------------------------------------------------------
case "${NUM_VIEWS}_${RESOLUTION}" in
  "8_low_res")
    INDEX_PATH="assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json"
    RESOLUTION_OPTS=(
      "dataset.roots=[datasets/dl3dv-480p-chunks]"
    )
    ;;
  "8_high_res")
    INDEX_PATH="assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json"
    RESOLUTION_OPTS=(
      "dataset.image_shape=[512,960]"
      "dataset.ori_image_shape=[540,960]"
      "dataset.roots=[datasets/dl3dv_960p_test_subset]"
    )
    ;;
  "32_low_res")
    INDEX_PATH="assets/dl3dv_evaluation/dl3dv_start_0_distance_160_ctx_32v_tgt_24v.json"
    RESOLUTION_OPTS=(
      "dataset.roots=[datasets/dl3dv-480p-chunks]"
    )
    ;;
  "-1_low_res")
    INDEX_PATH=""
    RESOLUTION_OPTS=(
      "dataset.roots=[datasets/dl3dv-480p-chunks]"
    )
    ;;
  *)
    echo "[test.sh] Unknown NUM_VIEWS/RESOLUTION combination: ${NUM_VIEWS}/${RESOLUTION}" >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# ③ COMMON Hydra overrides
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/common_opts.sh"   # populates COMMON_OPTS with shared opts

# dl3dv-specific additions
COMMON_OPTS+=(
  # ensures test_start_idx scene-counter works correctly across SLURM array jobs
  "data_loader.test.num_workers=0"

  # Experiment
  "+experiment=test_dl3dv"

  # Dataset
  "dataset.opencv_pose_format=false"

  # Resolution
  "${RESOLUTION_OPTS[@]}"

  "meta_trainer.test.metrics_batch_size=48"
  "meta_trainer.test.save_render_image=false"
  "meta_trainer.test.save_every_freq=[2,10,100,200]"
  "meta_trainer.test.save_every_steps=[0,10,100,1000]"
  "meta_trainer.test.save_at_iters=null"
  "meta_trainer.test.eval_context_views=${EVAL_CONTEXT_VIEWS}"
)

if [[ "${NUM_VIEWS}" == "-1" ]]; then
  COMMON_OPTS+=(
    "dataset/view_sampler=dense"
    "dataset.view_sampler.target_every=8"
    "dataset.view_sampler.num_context_views=-1"
    "dataset.view_sampler.num_target_views=-1"
  )
else
  COMMON_OPTS+=(
    "dataset/view_sampler=evaluation"
    "dataset.view_sampler.num_context_views=${NUM_VIEWS}"
    "dataset.view_sampler.index_path=${INDEX_PATH}"
  )
fi

# ---------------------------------------------------------------------------
# ④ INIT overrides (appends to COMMON_OPTS)
# ---------------------------------------------------------------------------
DENSE_COLMAP_EXTRA_OPTS=(
  "scene_trainer.scene_initializer.normalize_world_space=false"
  "scene_trainer.scene_initializer.dl3dv_settings=true"
  "scene_trainer.scene_initializer.filter_zero_rgb=true"
  )
SPARSE_COLMAP_EXTRA_OPTS=(
  # Loading colmap reconstruction extracted from sparse views
  # Haofei data:
  # dataset/dl3dv_start_0_distance_40_ctx_8v_tgt_8v_256x448
  
)
source "${COMMON_DIR}/init_opts.sh"

# ---------------------------------------------------------------------------
# ⑤ RUN (shared logic)
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/run_experiments.sh"
