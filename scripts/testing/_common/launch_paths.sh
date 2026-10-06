#!/usr/bin/env bash
# _common/launch_paths.sh
# Resolve the paths every dataset's launch.sh needs, then load the shared helpers. Source it as the
# launcher's first line; launch_common.sh (sourced last) submits the jobs. It reads the *caller's*
# path (BASH_SOURCE[1]), so it must be sourced, not run.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/test.sh"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMMON_DIR="${REPO_ROOT}/scripts/testing/_common"
AGG_SCRIPT="${COMMON_DIR}/aggregate.sh"
source "${COMMON_DIR}/pretty_print.sh"
source "${COMMON_DIR}/results_root.sh"
