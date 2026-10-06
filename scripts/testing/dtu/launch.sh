#!/usr/bin/env bash
# =============================================================================
# dtu/launch.sh  — configure and launch the DTU evaluation
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../_common/launch_paths.sh"

# ---------------------------------------------------------------------------
# Dry run: 1 = print only, 0 = submit
# ---------------------------------------------------------------------------
export DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------------------
# Speed-up flags for "ours" (fused attention + lazy KNN update): 1 = on, 0 = off
# ---------------------------------------------------------------------------
export SPEED_UP=1

# ---------------------------------------------------------------------------
# ① EVALUATION AXES
# ---------------------------------------------------------------------------
export INIT="${INIT:-dense_colmap}"       # dense_colmap | resplat_v1
export VIEW_SAMPLER="${VIEW_SAMPLER:-dense}"  # dense | evaluation | boundedv2_360 | ...
export NUM_VIEWS="${NUM_VIEWS:--1}"       # context views used when VIEW_SAMPLER != dense (also sets BASE_OUT subdir)
export INDEX_PATH="${INDEX_PATH:-}"       # required for VIEW_SAMPLER=evaluation, e.g. assets/dtu_evaluation/dtu_ctx8_tgt8_seq_skip4.json
export SCENE_NUM=15
export SCENES_PER_JOB=15
export NUM_STEPS="${NUM_STEPS:-2000}"
export OPT_BATCH_SIZE="${OPT_BATCH_SIZE:-8}"
export SUBSAMPLE=2
export TARGET_EVERY=8
export DATASET_ROOT="datasets/DTU"
export DENSE_COLMAP_PATH="datasets/DTU"
export EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"
export NORMALIZE=true
[[ "$NORMALIZE" == "true" ]] && _norm_str="_normalize_scene" || _norm_str=""
export OUT_SUFFIX="${NUM_STEPS}${_norm_str}"  # appended to output dir name (for organizational purposes; optional)


# ---------------------------------------------------------------------------
# ② OUR CHECKPOINTS  ("rel_dir|ckpt_name" — comment out to disable)
# ---------------------------------------------------------------------------
OURS_RUNS=(
  # The released Learn2Splat checkpoint (one model for both the sparse and dense regimes).
  "hf://autonomousvision/learn2splat/joint|checkpoints/epoch_7-step_150000.ckpt|true"
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
_index_id="$(basename "${INDEX_PATH}" .json)"   # empty unless VIEW_SAMPLER=evaluation
export BASE_OUT="${RESULTS_ROOT}/dtu/${BS_STR}_${NV_STR}_${SCENE_NUM}_scenes/${_index_id:+${_index_id}/}${INIT}_init"

# Extra key-value pairs to print in the header
EXTRA_PRINT_KVS=("INIT:${INIT}" "VIEW_SAMPLER:${VIEW_SAMPLER}" "NUM_VIEWS:${NUM_VIEWS}" "SUBSAMPLE:${SUBSAMPLE}" "TARGET_EVERY:${TARGET_EVERY}")

DATASET_NAME="DTU"
source "${COMMON_DIR}/launch_common.sh"

# ---------------------------------------------------------------------------
# Submit — comment out whichever mode you don't need
# ---------------------------------------------------------------------------
submit "ours"      "${ARRAY_OURS}"      "${N_OURS}"
submit "baselines" "${ARRAY_BASELINES}" "${N_BASELINES}"
