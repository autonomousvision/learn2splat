#!/usr/bin/env bash
# _common/results_root.sh
# Single root for all evaluation outputs: <root>/<dataset>/...
# Override via the environment to keep a set of runs apart, e.g.
#   RESULTS_ROOT=results/exp bash scripts/testing/launch_all.sh
export RESULTS_ROOT="${RESULTS_ROOT:-results}"