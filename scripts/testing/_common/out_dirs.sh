#!/usr/bin/env bash
# =============================================================================
# _common/out_dirs.sh — Output-directory path computation (single source)
# =============================================================================
# Sourced by both launch_common.sh (to predict dirs for the dependent aggregation
# job / dry-run printing) and run_experiments.sh (the actual run). Keeping one
# copy ensures the predicted and produced paths can never drift apart.
#
# Reads from the environment: BASE_OUT, OUT_SUFFIX (defaults to NUM_STEPS) and SPEED_UP.
# =============================================================================

# DECODER_TAG: empty unless an optional decoder override is active, in which case
# it tags the output dir so overridden runs don't collide with per-checkpoint runs.
source "${COMMON_DIR}/decoder_opts.sh"

# ckpt_step_str <ckpt_name_or_path>
#   "checkpoints/last"                 -> "last"
#   "checkpoints/epoch_9-step_90000.ckpt" -> "90000"
ckpt_step_str() {
  local ckpt_basename="${1##*/}"
  if [[ "$ckpt_basename" == "last" ]]; then
    echo "last"
  else
    local no_ext="${ckpt_basename%.ckpt}"
    local step_part="${no_ext##*-}"
    echo "${step_part#step_}"
  fi
}

# ours_out_dir <rel_dir> <ckpt_name>  -> full output dir for an "ours" run
#   ${BASE_OUT}/<rel_dir>/<suffix>_<step>[_speedup]
ours_out_dir() {
  # A Hugging Face ref contributes only its checkpoint folder: hf://org/repo/joint -> joint
  local rel_dir="$1" ckpt_name="$2"
  [[ "$rel_dir" == hf://* ]] && rel_dir="${rel_dir#hf://*/*/}"
  local suffix="${OUT_SUFFIX:-${NUM_STEPS}}"
  local out_dir="${BASE_OUT}/${rel_dir}/${suffix}_$(ckpt_step_str "$ckpt_name")"
  [[ "${SPEED_UP:-0}" == "1" ]] && out_dir="${out_dir}_speedup"
  echo "${out_dir}${DECODER_TAG}"
}

# baseline_full_out_dir <exp>  -> full output dir for a baseline run
#   ${BASE_OUT}/<baseline name>/<suffix>
baseline_full_out_dir() {
  local suffix="${OUT_SUFFIX:-${NUM_STEPS}}"
  echo "${BASE_OUT}/$1/${suffix}${DECODER_TAG}"
}