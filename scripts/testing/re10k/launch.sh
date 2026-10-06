#!/usr/bin/env bash
# =============================================================================
# re10k/launch.sh  — configure and launch the RE10K evaluation
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
export INIT="resplat_v1"  # Currently, no colmap initialization available
export NUM_VIEWS=8
export SCENE_NUM=72
export TEST_CHUNK_INTERVAL=25
export SCENES_PER_JOB=72
export NUM_STEPS="${NUM_STEPS:-2000}"
export OPT_BATCH_SIZE=8
export EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"

# ---------------------------------------------------------------------------
# ② OUR CHECKPOINTS  ("rel_dir|ckpt_name[|init_state_from_resplat]" — comment out to disable)
#
#   init_state_from_resplat (optional, default: auto):
#     auto  → inferred from INIT (resplat→true, colmap→false)
#     true  → ckpt was trained WITH resplat init (no init_state_* overrides)
#     false → ckpt was trained WITHOUT resplat init (adds init_state_wo_features=true etc.)
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
# BASE_OUT — replicated here for dry-run printing (must match test.sh logic)
# ---------------------------------------------------------------------------
[[ "$NUM_VIEWS"      == "-1" ]] && NV_STR="all" || NV_STR="${NUM_VIEWS}"
[[ "$OPT_BATCH_SIZE" == "-1" ]] && BS_STR="all" || BS_STR="${OPT_BATCH_SIZE}"
export BASE_OUT="${RESULTS_ROOT}/re10k/${BS_STR}_${NV_STR}_${TEST_CHUNK_INTERVAL}int/${INIT}_init"

# Extra key-value pairs to print in the header
EXTRA_PRINT_KVS=("INIT:${INIT}" "NUM_VIEWS:${NUM_VIEWS}" "SCENE_NUM:${SCENE_NUM}")

DATASET_NAME="RE10K"
source "${COMMON_DIR}/launch_common.sh"

# ---------------------------------------------------------------------------
# Submit — comment out whichever mode you don't need
# ---------------------------------------------------------------------------
submit "ours"      "${ARRAY_OURS}"      "${N_OURS}"
submit "baselines" "${ARRAY_BASELINES}" "${N_BASELINES}"

