#!/usr/bin/env bash
# =============================================================================
# custom/launch.sh  — configure and launch the custom evaluation
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../_common/launch_paths.sh"

# ---------------------------------------------------------------------------
# Dry run: 1 = print only, 0 = submit
# ---------------------------------------------------------------------------
export DRY_RUN="${DRY_RUN:-0}"
ECHO_FORMAT="plain"

# ---------------------------------------------------------------------------
# Speed-up flags for "ours" (fused attention + lazy KNN update): 1 = on, 0 = off
# ---------------------------------------------------------------------------
export SPEED_UP=1

# ---------------------------------------------------------------------------
# ① EVALUATION AXES
# ---------------------------------------------------------------------------
export INIT="dense_colmap"   # dense_colmap | resplat_v1
export VIEW_SAMPLER="dense"  # dense | evaluation | ...
export NUM_VIEWS=-1
export SCENE_NUM=1
export SCENES_PER_JOB=1
export NUM_STEPS=2000
export OPT_BATCH_SIZE=8
export SUBSAMPLE=1
export TARGET_EVERY=8
export EVAL_CONTEXT_VIEWS=false
export NORMALIZE=false

export DATASET_ROOT="datasets/custom"
export DENSE_COLMAP_PATH="datasets/custom"
export EXTRA_OPTS=(
  "dataset.scene_name=new_scene"   # scene dir under DATASET_ROOT
  "meta_trainer.test.save_render_image=true"
)

# ---------------------------------------------------------------------------
# ② OUR CHECKPOINTS  ("rel_dir|ckpt_name|use_resplat_features" — comment out to disable)
# ---------------------------------------------------------------------------
OURS_RUNS=(
  # The released Learn2Splat checkpoint (one model for both the sparse and dense regimes).
  "hf://autonomousvision/learn2splat/joint|checkpoints/epoch_7-step_150000.ckpt|false"
)

# ---------------------------------------------------------------------------
# ③ BASELINE EXPERIMENTS  (comment out to disable)
# ---------------------------------------------------------------------------
BASELINE_EXPERIMENTS=(
  "adam"
  "adam_tuned"
)

# ---------------------------------------------------------------------------
# BASE_OUT
# ---------------------------------------------------------------------------
[[ "$OPT_BATCH_SIZE" == "-1" ]] && BS_STR="all" || BS_STR="${OPT_BATCH_SIZE}"
[[ "$VIEW_SAMPLER"   == "dense" || "$NUM_VIEWS" == "-1" ]] && NV_STR="all" || NV_STR="${NUM_VIEWS}"
export BASE_OUT="${RESULTS_ROOT}/custom/${BS_STR}_${NV_STR}_${SCENE_NUM}_scenes/${INIT}_init"

# Extra key-value pairs to print in the header
EXTRA_PRINT_KVS=("INIT:${INIT}" "VIEW_SAMPLER:${VIEW_SAMPLER}" "NUM_VIEWS:${NUM_VIEWS}" "SUBSAMPLE:${SUBSAMPLE}" "TARGET_EVERY:${TARGET_EVERY}" "DATASET_ROOT:${DATASET_ROOT}")

DATASET_NAME="Custom"
source "${COMMON_DIR}/launch_common.sh"

# ---------------------------------------------------------------------------
# Submit — comment out whichever mode you don't need
# ---------------------------------------------------------------------------
submit "ours"      "${ARRAY_OURS}"      "${N_OURS}"
submit "baselines" "${ARRAY_BASELINES}" "${N_BASELINES}"