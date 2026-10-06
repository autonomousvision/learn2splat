#!/usr/bin/env bash
# =============================================================================
# _common/launch_common.sh
# =============================================================================
# Shared submission logic for all dataset launch scripts.
# Source this AFTER setting all required variables:
#
#   Required:
#     DATASET_NAME  — human-readable name for print header (e.g. "DL3DV")
#     BASE_OUT      — output root path (dataset-specific computation)
#     TEST_SCRIPT   — absolute path to the dataset's test.sh
#     AGG_SCRIPT    — absolute path to the dataset's aggregate.sh
#     OURS_RUNS     — bash array of "rel_dir|ckpt_name" entries
#     BASELINE_EXPERIMENTS — bash array of baseline names
#     SCENE_NUM, SCENES_PER_JOB, NUM_STEPS, OPT_BATCH_SIZE
#
#   Optional — extra info printed in the header:
#     EXTRA_PRINT_KVS — associative array of key→value pairs to print
#                        e.g. EXTRA_PRINT_KVS=("NUM_VIEWS:8" "INIT:resplat")
# =============================================================================

# ---------------------------------------------------------------------------
# DRY_RUN / local fallback (can be set before sourcing to override)
# ---------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
RUN_BASH="${RUN_BASH:-0}"

# ---------------------------------------------------------------------------
# Logging — tee all subsequent output to a timestamped log file in BASE_OUT
# ---------------------------------------------------------------------------
_LOG_FILE="${BASE_OUT}/launch_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p "${BASE_OUT}"
exec > >(tee "$_LOG_FILE") 2>&1
echo "[launch] Logging to: ${_LOG_FILE}"

if ! command -v sbatch &>/dev/null; then
  print_warn "sbatch not found — running bash instead. Will only run task 0 of each array."
  RUN_BASH=1
fi

# ---------------------------------------------------------------------------
# External overrides — replace the in-file OURS_RUNS / BASELINE_EXPERIMENTS with
# colon-separated entries from the environment (the in-file arrays are defaults).
# Use "none" to run an empty set (e.g. skip baselines). Examples:
#   OURS_RUNS_OVERRIDE="relA|checkpoints/x.ckpt|true:relB|checkpoints/y.ckpt|false"
#   BASELINE_EXPERIMENTS_OVERRIDE=none
# ---------------------------------------------------------------------------
case "${OURS_RUNS_OVERRIDE:-}" in
  "")   ;;
  none) OURS_RUNS=() ;;
  *)    # Split on ":" but keep the "://" of hf:// entries intact.
        IFS=':' read -r -a OURS_RUNS <<< "${OURS_RUNS_OVERRIDE//:\/\//@@SCHEME@@}"
        OURS_RUNS=("${OURS_RUNS[@]//@@SCHEME@@/://}") ;;
esac
case "${BASELINE_EXPERIMENTS_OVERRIDE:-}" in
  "")   ;;
  none) BASELINE_EXPERIMENTS=() ;;
  *)    IFS=':' read -r -a BASELINE_EXPERIMENTS <<< "${BASELINE_EXPERIMENTS_OVERRIDE}" ;;
esac

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------
export OURS_RUNS_ENV=$(IFS=':'; echo "${OURS_RUNS[*]}")
export BASELINE_EXPERIMENTS_ENV=$(IFS=':'; echo "${BASELINE_EXPERIMENTS[*]}")
# EVAL_EXTRA_OPTS: extra Hydra overrides appended to every eval python call, injected from the
# environment. The per-dataset launch.sh resets the in-file EXTRA_OPTS, so this env hook is what
# survives (e.g. an eval-time scale-clamp override). Pair with OUT_SUFFIX to tag the output dir
# so the overridden run doesn't clobber the canonical one.
export EXTRA_OPTS_ENV="${EXTRA_OPTS[*]}${EVAL_EXTRA_OPTS:+ ${EVAL_EXTRA_OPTS}}"

# Decoder override: export so the SLURM job inherits it (sbatch --export=ALL).
# The job re-resolves the opts + dir tag via decoder_opts.sh from these vars; without
# the export it would silently run with the decoder saved in each checkpoint and write to
# the untagged (canonical) output dir.
[[ -n "${DECODER:-}" ]]                && export DECODER
[[ -n "${DECODER_EPS2D:-}" ]]          && export DECODER_EPS2D
[[ -n "${DECODER_RASTERIZE_MODE:-}" ]] && export DECODER_RASTERIZE_MODE

N_OURS=${#OURS_RUNS[@]}
N_BASELINES=${#BASELINE_EXPERIMENTS[@]}
N_SCENE_CHUNKS=$(( (SCENE_NUM + SCENES_PER_JOB - 1) / SCENES_PER_JOB ))

ARRAY_OURS=$(( N_OURS * N_SCENE_CHUNKS - 1 ))
ARRAY_BASELINES=$(( N_BASELINES * N_SCENE_CHUNKS - 1 ))

# Resolve the DECODER preset (eps2d / rasterize_mode) up front so the header prints the
# real mode, not the "antialiased" fallback. out_dirs.sh re-sources this (idempotent).
source "${COMMON_DIR}/decoder_opts.sh"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
print_header "${DATASET_NAME} Launch"
print_kv "BASE_OUT"       "${BASE_OUT}"
print_kv "SCENE_NUM"      "${SCENE_NUM}"
print_kv "SCENES_PER_JOB" "${SCENES_PER_JOB}  (${N_SCENE_CHUNKS} chunk(s))"
print_kv "NUM_STEPS"      "${NUM_STEPS}"
print_kv "OPT_BATCH_SIZE" "${OPT_BATCH_SIZE}"
[[ -n "${DECODER:-}" || -n "${DECODER_EPS2D:-}" || -n "${DECODER_RASTERIZE_MODE:-}" ]] && \
  print_kv "DECODER" "${DECODER:-custom} (eps2d=${DECODER_EPS2D:-0.3}, mode=${DECODER_RASTERIZE_MODE:-antialiased})"
# Print any extra dataset-specific key-value pairs
for kv in "${EXTRA_PRINT_KVS[@]:-}"; do
  [[ -z "$kv" ]] && continue
  print_kv "${kv%%:*}" "${kv#*:}"
done
[[ $N_OURS      -gt 0 ]] && print_info "ours      — ${N_OURS} run(s)  × ${N_SCENE_CHUNKS} chunk(s) → --array=0-${ARRAY_OURS}"
[[ $N_BASELINES -gt 0 ]] && print_info "baselines — ${N_BASELINES} method(s) × ${N_SCENE_CHUNKS} chunk(s) → --array=0-${ARRAY_BASELINES}"
[[ "$DRY_RUN"  == "1" ]] && print_warn "DRY RUN — no jobs will be submitted."
[[ "$RUN_BASH" == "1" && "$DRY_RUN" != "1" ]] && print_warn "LOCAL MODE — running bash (task 0 only per method)."

# ---------------------------------------------------------------------------
# Email for SLURM job notifications. Set SBATCH_MAIL in your environment or .env
# to receive them. Leave empty to skip.
# ---------------------------------------------------------------------------
SBATCH_MAIL="${SBATCH_MAIL:-}"

# ---------------------------------------------------------------------------
# Output-directory paths: shared with run_experiments.sh so the dirs predicted
# here for the aggregation job match the ones the run actually writes.
# ---------------------------------------------------------------------------
source "${COMMON_DIR}/out_dirs.sh"

# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------
# Colon-joined list of the dirs a method writes, in the form aggregate.sh reads as AGG_DIRS.
# Built with the same out_dirs.sh helpers run_experiments.sh uses to choose those dirs, so the two
# cannot disagree about where the run's output landed.
agg_dirs_for() {
  local method="$1" n_methods="$2" mi dirs=()
  for (( mi=0; mi < n_methods; mi++ )); do
    if [[ "$method" == "ours" ]]; then
      IFS='|' read -r rel_dir ckpt_name _init_state <<< "${OURS_RUNS[$mi]}"
      dirs+=("$(ours_out_dir "$rel_dir" "$ckpt_name")")
    else
      dirs+=("$(baseline_full_out_dir "${BASELINE_EXPERIMENTS[$mi]}")")
    fi
  done
  (IFS=':'; echo "${dirs[*]}")
}

submit() {
  local method="$1" array="$2" n_methods="$3"
  [[ $n_methods -eq 0 ]] && print_warn "Skipping ${method} — no entries." && return
  export METHOD="$method"

  if [[ "$DRY_RUN" == "1" ]]; then
    # Preview one representative command per run (chunk 0 of each method) instead
    # of every method × chunk × scene — enough to eyeball the planned fan-out.
    print_info "[DRY RUN] would submit --array=0-${array} (METHOD=${method}) — one sample command per run:"
    export DRY_RUN ECHO_FORMAT
    for (( mi=0; mi < n_methods; mi++ )); do
      SLURM_ARRAY_TASK_ID=$(( mi * N_SCENE_CHUNKS )) bash "$TEST_SCRIPT"
    done
    return
  fi

  if [[ "$RUN_BASH" == "1" ]]; then
    print_info "Running bash ${TEST_SCRIPT}  (task 0, METHOD=${method})"
    export DRY_RUN ECHO_FORMAT
    SLURM_ARRAY_TASK_ID=0 bash "$TEST_SCRIPT"
    # No dependent job to hang the aggregation off, so run it here, once the test returns.
    AGG_DIRS="$(agg_dirs_for "$method" "$n_methods")" bash "$AGG_SCRIPT"
    return
  fi

  # --- real sbatch submit ---
  # The #SBATCH --output paths are under logs/
  mkdir -p logs

  local job_id
  job_id=$(sbatch --array=0-${array} --parsable  --export=ALL \
    ${EVAL_DEPENDENCY:+--dependency="${EVAL_DEPENDENCY}" --kill-on-invalid-dep=yes} \
    ${SBATCH_MAIL:+--mail-user="${SBATCH_MAIL}" --mail-type=BEGIN,END,FAIL,REQUEUE} \
    "$TEST_SCRIPT")
  print_success "Submitted ${method} array job: ${job_id}  (--array=0-${array})"

  local agg_job_id
  agg_job_id=$(sbatch --dependency=afterok:"${job_id}" --kill-on-invalid-dep=yes --parsable \
    ${SBATCH_MAIL:+--mail-user="${SBATCH_MAIL}" --mail-type=END,FAIL} \
    --export=ALL,AGG_DIRS="$(agg_dirs_for "$method" "$n_methods")" \
    "$AGG_SCRIPT")
  print_success "Submitted aggregation job: ${agg_job_id}  (runs after all tasks of job ${job_id} succeed)"
}
