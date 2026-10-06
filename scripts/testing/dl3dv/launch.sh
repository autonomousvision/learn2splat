#!/usr/bin/env bash
# =============================================================================
# dl3dv/launch.sh  — configure and launch the DL3DV evaluation
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../_common/launch_paths.sh"

# ---------------------------------------------------------------------------
# Dry run: 1 = print only, 0 = submit
# ---------------------------------------------------------------------------
export DRY_RUN="${DRY_RUN:-0}"
ECHO_FORMAT="plain"  # {"plain", "vscode"}
RUN_BASH=0  # if 1, runs bash test.sh directly (no array, no parallelism; good for debugging)

# ---------------------------------------------------------------------------
# Speed-up flags for "ours" (fused attention + lazy KNN update): 1 = on, 0 = off
# ---------------------------------------------------------------------------
export SPEED_UP=1

# ---------------------------------------------------------------------------
# ① EVALUATION AXES
# ---------------------------------------------------------------------------
# Settings matrix — one full launch per row.
#   INIT       : resplat_v1 | dense_colmap | sparse_colmap
#   RESOLUTION : low_res | high_res
# NUM_VIEWS    : 8 | 32 | -1
MATRIX=(
  "INIT=resplat_v1 RESOLUTION=low_res  NUM_VIEWS=8"
  "INIT=resplat_v1 RESOLUTION=high_res NUM_VIEWS=8"
  "INIT=resplat_v1 RESOLUTION=low_res  NUM_VIEWS=32"
  "INIT=dense_colmap RESOLUTION=low_res  NUM_VIEWS=-1"
)
source "${COMMON_DIR}/matrix.sh"   # parent fans out over MATRIX (one launch per row), then exits

export SCENE_NUM="${SCENE_NUM:-140}"
export SCENES_PER_JOB="${SCENES_PER_JOB:-35}"
export NUM_STEPS="${NUM_STEPS:-2000}"
export OPT_BATCH_SIZE=8
# export OUT_SUFFIX="${NUM_STEPS}"  # appended to output dir name (for organizational purposes; optional)
export EXTRA_OPTS=()
export EVAL_CONTEXT_VIEWS="${EVAL_CONTEXT_VIEWS:-false}"
apply_combo   # set INIT / RESOLUTION / NUM_VIEWS from the active matrix row

# opt_batch_size is how many context views the optimizer refines over per step (-1 = all of them
# in one batch). resplat init is the sparse setting: the context set is small and fixed (8 or 32),
# so set opt_batch_size to that count and refine all context views together. colmap init is the
# dense setting: the context set is large, so refine 8 fps-sampled views per step instead of all.
# opt_batch_size becomes part of the output dir name (via BS_STR, e.g. 32_32_low_res...).
case "$INIT" in
  *colmap*) export OPT_BATCH_SIZE=8 ;;
  *)        export OPT_BATCH_SIZE="$NUM_VIEWS" ;;
esac

# ---------------------------------------------------------------------------
# ② OUR CHECKPOINTS  ("rel_dir|ckpt_name[|init_state_from_resplat]")
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
#   "init_only"     # init output only (no optimization); uses INIT setting above
)

# ---------------------------------------------------------------------------
# BASE_OUT — replicated here for dry-run printing (must match test.sh logic)
# ---------------------------------------------------------------------------
[[ "$NUM_VIEWS"      == "-1" ]] && NV_STR="all"  || NV_STR="${NUM_VIEWS}"
[[ "$OPT_BATCH_SIZE" == "-1" ]] && BS_STR="all"  || BS_STR="${OPT_BATCH_SIZE}"
export BASE_OUT="${RESULTS_ROOT}/dl3dv/${BS_STR}_${NV_STR}_${RESOLUTION}_${SCENE_NUM}_scenes/${INIT}_init"

# Extra key-value pairs to print in the header
EXTRA_PRINT_KVS=("INIT:${INIT}" "NUM_VIEWS:${NUM_VIEWS}" "RESOLUTION:${RESOLUTION}" "OUT_SUFFIX:${OUT_SUFFIX}")

DATASET_NAME="DL3DV"
source "${COMMON_DIR}/launch_common.sh"

# ---------------------------------------------------------------------------
# Submit — comment out whichever mode you don't need
# ---------------------------------------------------------------------------
submit "ours"      "${ARRAY_OURS}"      "${N_OURS}"
submit "baselines" "${ARRAY_BASELINES}" "${N_BASELINES}"