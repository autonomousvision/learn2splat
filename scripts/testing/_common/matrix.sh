#!/usr/bin/env bash
# =============================================================================
# _common/matrix.sh
# =============================================================================
# Settings-matrix dispatcher. Source this from a dataset launch.sh AFTER
# defining the MATRIX array and BEFORE the evaluation-axis defaults:
#
#   MATRIX=(
#     "INIT=resplat_v1 RESOLUTION=low_res  NUM_VIEWS=8"
#     "INIT=resplat_v1 RESOLUTION=high_res NUM_VIEWS=8"
#   )
#   source "${COMMON_DIR}/matrix.sh"
#   ... axis defaults (export INIT=..., etc.) ...
#   apply_combo            # overrides the defaults with the current row
#
# Each row is one full launch. The parent process (COMBO unset) re-runs this
# same launch.sh once per row, passing the row as $COMBO; the child (COMBO set)
# continues normally and apply_combo overlays the row on top of the defaults.
# An empty MATRIX means "single launch with the file's defaults".
#
# Run a subset of rows with ROW (0-based index, comma-separated for several):
#   ROW=1   bash <dataset>/launch.sh     # only MATRIX[1]
#   ROW=0,2 bash <dataset>/launch.sh     # MATRIX[0] and MATRIX[2]
# =============================================================================

if [[ -z "${COMBO:-}" && ${#MATRIX[@]} -gt 0 ]]; then
  _rows=("${MATRIX[@]}")
  if [[ -n "${ROW:-}" ]]; then
    _rows=()
    IFS=',' read -ra _idxs <<< "${ROW}"
    for _i in "${_idxs[@]}"; do
      [[ -n "${MATRIX[$_i]:-}" ]] && _rows+=("${MATRIX[$_i]}") || print_warn "[matrix] no row at index ${_i}"
    done
  fi
  for _row in "${_rows[@]}"; do
    print_info "[matrix] launching: ${_row}"
    COMBO="${_row}" bash "$0" || print_warn "[matrix] row failed: ${_row}"
  done
  exit 0
fi

# Overlay the current matrix row (KEY=VAL ...) on top of the axis defaults.
apply_combo() { [[ -n "${COMBO:-}" ]] && eval "export ${COMBO}"; }