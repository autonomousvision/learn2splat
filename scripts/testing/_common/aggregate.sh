#!/usr/bin/env bash
# =============================================================================
# _common/aggregate.sh  — Run aggregate_metrics on all output dirs
# Submitted automatically by launch.sh with --dependency=afterok:<array_jobid>
# =============================================================================

#SBATCH --job-name=aggregate
#SBATCH --partition=***   # set your cluster CPU partition(s)
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/%j_aggregate.out
#SBATCH --error=logs/%j_aggregate.err

if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
COMMON_DIR="${REPO_ROOT}/scripts/testing/_common"

source "${COMMON_DIR}/env_setup.sh"
source "${COMMON_DIR}/pretty_print.sh"

IFS=':' read -r -a DIRS <<< "${AGG_DIRS:-}"

print_header "Aggregating metrics"

for dir in "${DIRS[@]}"; do
  if [[ -d "${dir}" ]]; then
    print_info "Aggregating: ${dir}"
    python scripts/testing/aggregate_metrics.py "${dir}"
    print_success "Done: ${dir}"
  else
    print_warn "Output dir not found: ${dir} — skipping."
  fi
done

print_success "All aggregations done."

