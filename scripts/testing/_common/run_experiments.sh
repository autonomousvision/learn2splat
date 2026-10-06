#!/usr/bin/env bash
# =============================================================================
# _common/run_experiments.sh
# =============================================================================
# Shared run logic for all dataset test scripts.
# Source this AFTER setting all required variables:
#
#   Required:
#     BASE_OUT         — output root path
#     COMMON_OPTS      — base Hydra overrides array
#     INIT_STATE_OPTS  — optimizer init-state overrides (set per-run for ours, from init_opts.sh for baselines)
#     METHOD           — "ours" | "baselines"
#     OURS_RUNS        — array of "rel_dir|ckpt_name" entries
#     BASELINE_EXPERIMENTS — array of baseline names
#     SCENE_NUM, SCENES_PER_JOB, NUM_STEPS
#
# =============================================================================

# Output-dir paths: shared with launch_common.sh so the produced dirs match the
# ones predicted for the aggregation job.
source "${COMMON_DIR}/out_dirs.sh"

# ---------------------------------------------------------------------------
# Unpack run lists from colon-delimited env vars (set by launch.sh)
# ---------------------------------------------------------------------------
# Split on ":" but keep the "://" of hf:// entries intact.
IFS=':' read -r -a OURS_RUNS <<< "${OURS_RUNS_ENV//:\/\//@@SCHEME@@}"
OURS_RUNS=("${OURS_RUNS[@]//@@SCHEME@@/://}")
IFS=':' read -r -a BASELINE_EXPERIMENTS <<< "${BASELINE_EXPERIMENTS_ENV:-}"

# ---------------------------------------------------------------------------
# OUT_SUFFIX — leaf output dir (default: NUM_STEPS)
# ---------------------------------------------------------------------------
OUT_SUFFIX="${OUT_SUFFIX:-${NUM_STEPS}}"

# ---------------------------------------------------------------------------
# LR_MAX_STEPS — LR-schedule horizon for the adam/adam_tuned baselines.
# Defaults to the run length (NUM_STEPS). Set it larger (e.g. 30000 while
# NUM_STEPS=10000) to read an early-iteration row from a longer run without the
# LR having already collapsed.
# ---------------------------------------------------------------------------
LR_MAX_STEPS="${LR_MAX_STEPS:-${NUM_STEPS}}"

# ---------------------------------------------------------------------------
# EXTRA_OPTS — additional Hydra overrides passed to every python call
# ---------------------------------------------------------------------------
# Normalize any newlines to spaces first: `read` stops at the first newline, so a multi-line
# EVAL_EXTRA_OPTS would silently drop every override after the first line.
read -r -a EXTRA_OPTS <<< "${EXTRA_OPTS_ENV:+${EXTRA_OPTS_ENV//$'\n'/ }}"

# ---------------------------------------------------------------------------
# Decoder override display (DECODER_OPTS/DECODER_TAG come from decoder_opts.sh,
# sourced via out_dirs.sh above). Empty override -> the decoder saved in each checkpoint.
# ---------------------------------------------------------------------------
if [[ ${#DECODER_OPTS[@]} -gt 0 ]]; then
  DECODER_PRINT="${DECODER:-custom} (eps2d=${DECODER_EPS2D:-0.3}, mode=${DECODER_RASTERIZE_MODE:-antialiased})"
else
  DECODER_PRINT="per-checkpoint (no override)"
fi

# ---------------------------------------------------------------------------
# DRY_RUN     — if 1, print the python command but do not execute it
# ECHO_FORMAT — "plain" (default) or "vscode" (launch.json "args" array)
# ---------------------------------------------------------------------------
_run_python() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if [[ "${ECHO_FORMAT:-plain}" == "vscode" ]]; then
      # Emit a JSON "args" array ready to paste into launch.json
      local out='"args": ['
      local first=1
      for arg in "$@"; do
        [[ $first -eq 0 ]] && out+=", "
        out+="\"${arg}\""
        first=0
      done
      out+="]"
      echo "$out"
    else
      echo "+ python $*"
    fi
    return 0
  fi
  python "$@"
}

# ---------------------------------------------------------------------------
# Scene-chunk math
# task_id = method_idx * N_SCENE_CHUNKS + chunk_idx
# ---------------------------------------------------------------------------
N_SCENE_CHUNKS=$(( (SCENE_NUM + SCENES_PER_JOB - 1) / SCENES_PER_JOB ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK_IDX=$(( TASK_ID % N_SCENE_CHUNKS ))
METHOD_IDX=$(( TASK_ID / N_SCENE_CHUNKS ))
SCENE_START=$(( CHUNK_IDX * SCENES_PER_JOB ))

# Clamp to remaining scenes so the last chunk doesn't overshoot
REMAINING=$(( SCENE_NUM - SCENE_START ))
THIS_CHUNK_SIZE=$(( SCENES_PER_JOB < REMAINING ? SCENES_PER_JOB : REMAINING ))

SCENE_OPTS=(
  "dataset.test_start_idx=${SCENE_START}"
  "meta_trainer.limit_test_batches=${THIS_CHUNK_SIZE}"
)

# ---------------------------------------------------------------------------
# Per-scene iteration (used when SCENE_LIST_ENV is set, e.g. for per-scene subsample)
# Slices the full scene list to the scenes assigned to this chunk.
# ---------------------------------------------------------------------------
IFS=':' read -r -a _ALL_SCENES <<< "${SCENE_LIST_ENV:-}"
_SCENES_FOR_CHUNK=()
if [[ ${#_ALL_SCENES[@]} -gt 0 ]]; then
  _SCENES_FOR_CHUNK=("${_ALL_SCENES[@]:${SCENE_START}:${THIS_CHUNK_SIZE}}")
fi

# Default _subsample_for_scene if the dataset script didn't define one.
if ! declare -f _subsample_for_scene > /dev/null 2>&1; then
  _subsample_for_scene() { echo "${SUBSAMPLE:-4}"; }
fi

# Wrapper around _run_python that either loops per-scene (when _SCENES_FOR_CHUNK
# is populated) or falls back to the standard chunk-based SCENE_OPTS call.
# Use this instead of "_run_python ... ${SCENE_OPTS[@]} ..." in the run blocks.
_run_for_scenes() {
  if [[ ${#_SCENES_FOR_CHUNK[@]} -gt 0 ]]; then
    # Dry run: show one representative scene command, not one per scene.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      local _scene="${_SCENES_FOR_CHUNK[0]}"
      _run_python "$@" \
        "dataset.scene_name=${_scene}" \
        "dataset.subsample_factor=$(_subsample_for_scene "$_scene")" \
        "dataset.test_start_idx=0" \
        "meta_trainer.limit_test_batches=1"
      [[ ${#_SCENES_FOR_CHUNK[@]} -gt 1 ]] && print_info "(+ $(( ${#_SCENES_FOR_CHUNK[@]} - 1 )) more scene(s) in this chunk)"
      return
    fi
    for _scene in "${_SCENES_FOR_CHUNK[@]}"; do
      _scene_sub=$(_subsample_for_scene "$_scene")
      _run_python "$@" \
        "dataset.scene_name=${_scene}" \
        "dataset.subsample_factor=${_scene_sub}" \
        "dataset.test_start_idx=0" \
        "meta_trainer.limit_test_batches=1"
    done
  else
    _run_python "$@" "${SCENE_OPTS[@]}"
  fi
}

# ---------------------------------------------------------------------------
# _resolve_init_state_opts  <run_entry_or_empty>
# ---------------------------------------------------------------------------
# Sets INIT_STATE_OPTS for the current run.
# For "ours":     pass the full OURS_RUNS entry — third field is used if present.
# ---------------------------------------------------------------------------
_resolve_init_state_opts() {
  local entry="$1"
  local _field

  if [[ "$INIT" == "dense_colmap" || "$INIT" == "sparse_colmap" || "$INIT" == "ply" ]]; then
    # Colmap initializer never produces resplat features — always false
    _field="false"
  else
    # resplat: read optional third pipe field
    local _rest="${entry#*|}"
    local _ckpt="${_rest%%|*}"
    _field="${_rest##*|}"
    # No third field when the two sides of ## are equal
    [[ "$_field" == "$_ckpt" || -z "$_field" ]] && _field="auto"
    [[ "$_field" == "auto" ]] && _field="true"
  fi

  if [[ "$_field" == "false" ]]; then
    INIT_STATE_OPTS=(
      "scene_trainer.scene_optimizer.init_state_wo_features=true"
      "scene_trainer.scene_optimizer.init_state_scale=1"
      "scene_trainer.scene_optimizer.init_state_type=random"
    )
  else
    INIT_STATE_OPTS=()
  fi
  # Expose the resolved value for printing
  _RESOLVED_INIT_STATE="$_field"
}

# ===========================================================================
# RUN — "ours"
# ===========================================================================
if [[ "$METHOD" == "ours" ]]; then
  RUN_ENTRY="${OURS_RUNS[$METHOD_IDX]}"
  REL_DIR="${RUN_ENTRY%%|*}"
  _rest="${RUN_ENTRY#*|}"
  CKPT_NAME="${_rest%%|*}"

  _resolve_init_state_opts "${RUN_ENTRY}"

  # rel_dir is either a Hugging Face ref (hf://org/repo/dir) or a dir under checkpoints/.
  if [[ "$REL_DIR" == hf://* ]]; then
    CKPT_PATH="${REL_DIR}/${CKPT_NAME}"
  else
    CKPT_PATH="checkpoints/${REL_DIR}/${CKPT_NAME}"
  fi
  # If the ckpt name contains a "*" (e.g. "*-step_100000.ckpt"), resolve it to the
  # real file on disk — lets the eval follow a checkpoint whose epoch number isn't
  # known until training finishes. OUT_DIR keeps using CKPT_NAME (the step parses
  # out of the glob), so the predicted and produced dirs stay in sync.
  if [[ "$CKPT_NAME" == *"*"* ]]; then
    _match=$(ls -1t ${CKPT_PATH} 2>/dev/null | head -1)
    if [[ -n "$_match" ]]; then CKPT_PATH="$_match"
    elif [[ "${DRY_RUN:-0}" != "1" ]]; then print_warn "no checkpoint matches ${CKPT_PATH}"; fi
  fi
  OUT_DIR=$(ours_out_dir "$REL_DIR" "$CKPT_NAME")

  # Speed-up flags: fused attention is always on; SPEED_UP=1 adds the lazy-KNN flags.
  SPEED_UP_OPTS=("scene_trainer.scene_optimizer.use_fused_attn=true")
  if [[ "${SPEED_UP:-0}" == "1" ]]; then
    SPEED_UP_OPTS+=(
      "scene_trainer.scene_optimizer.knn_idx_update_every=100"
      "scene_trainer.scene_optimizer.update_only_nonzero_grad=false"
    )
  fi

  print_header "OURS — ${REL_DIR}"
  print_kv "ckpt"                    "${CKPT_NAME}"
  print_kv "init_state_from_resplat" "${_RESOLVED_INIT_STATE}"
  print_kv "speed_up"                "${SPEED_UP:-0}"
  print_kv "decoder"                 "${DECODER_PRINT}"
  print_kv "scenes"                  "${SCENE_START}..$(( SCENE_START + SCENES_PER_JOB - 1 ))"
  print_kv "out_dir"                 "${OUT_DIR}"

  _run_for_scenes -m learn2splat.main "${COMMON_OPTS[@]}" "${INIT_STATE_OPTS[@]}" "${SPEED_UP_OPTS[@]}" \
    scene_trainer/scene_optimizer=learn2splat \
    checkpointing.pretrained_optimizer="${CKPT_PATH}" \
    "${EXTRA_OPTS[@]}" \
    output_dir="${OUT_DIR}"

  print_success "Finished: ${REL_DIR}"
fi

# ===========================================================================
# RUN — "baselines"
#
# KEY RULE:
#   - The initializer comes from COMMON_OPTS (see init_opts.sh).
#   - The baselines have no optimizer ckpt, so they name the optimizer config explicitly:
#     adam = the 3DGS learning rates, adam_tuned = the grid-searched ones.
# ===========================================================================
if [[ "$METHOD" == "baselines" ]]; then
  EXP_NAME="${BASELINE_EXPERIMENTS[$METHOD_IDX]}"

  print_header "BASELINE — ${EXP_NAME}"
  print_kv "scenes"  "${SCENE_START}..$(( SCENE_START + SCENES_PER_JOB - 1 ))"
  print_kv "decoder" "${DECODER_PRINT}"

  OUT_DIR=""
  case $EXP_NAME in

    "adam")
      OUT_DIR=$(baseline_full_out_dir "$EXP_NAME")
      print_kv "out_dir" "${OUT_DIR}"
      _run_for_scenes -m learn2splat.main "${COMMON_OPTS[@]}" \
        scene_trainer/scene_optimizer=adam \
        scene_trainer/scene_optimizer/refiner=none \
        scene_trainer.scene_optimizer.lr_scheduler.max_steps="${LR_MAX_STEPS}" \
        "${EXTRA_OPTS[@]}" \
        output_dir="${OUT_DIR}"
      ;;

    "adam_tuned")
      OUT_DIR=$(baseline_full_out_dir "$EXP_NAME")
      print_kv "out_dir" "${OUT_DIR}"
      _run_for_scenes -m learn2splat.main "${COMMON_OPTS[@]}" \
        scene_trainer/scene_optimizer=adam_tuned \
        scene_trainer/scene_optimizer/refiner=none \
        scene_trainer.scene_optimizer.lr_scheduler.max_steps="${LR_MAX_STEPS}" \
        "${EXTRA_OPTS[@]}" \
        output_dir="${OUT_DIR}"
      ;;

    "init_only")
      # Initializer output only — no optimization steps.
      # Uses whatever INIT is set to (e.g. resplat_v1, dense_colmap).
      OUT_DIR=$(baseline_full_out_dir "$EXP_NAME")
      print_kv "out_dir" "${OUT_DIR}"
      _run_for_scenes -m learn2splat.main "${COMMON_OPTS[@]}" \
        dataset.opencv_pose_format=false \
        scene_trainer/scene_optimizer=none \
        scene_trainer.num_update_steps=0 \
        meta_trainer.test.eval_initialization=true \
        "${EXTRA_OPTS[@]}" \
        output_dir="${OUT_DIR}"
      ;;

    *)
      print_error "Unknown baseline: ${EXP_NAME} — skipping."
      ;;
  esac

  if [[ -n "$OUT_DIR" ]]; then
    print_info "Test done. Aggregation will run as a dependent job."
  fi

  print_success "Finished baseline: ${EXP_NAME}"
fi
