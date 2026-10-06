#!/usr/bin/env bash
# =============================================================================
# _common/decoder_opts.sh — Optional decoder override shared by all launches
# =============================================================================
# Opt-in. With nothing set, the decoder is left untouched: each checkpoint's
# saved rasterizer is used ("ours") and baselines fall back to the gsplat default
# (per-checkpoint behaviour — unchanged from before this file existed).
#
# When set, the override is appended to COMMON_OPTS, so it applies to EVERY method
# in a launch (ours + baselines). That keeps cross-method comparisons matched:
# all renders use the same eps2d / rasterize_mode regardless of what each
# checkpoint was trained with. An explicit CLI decoder override always wins over
# the decoder saved in a checkpoint (see learn2splat/config.py test-mode patching).
#
#   DECODER=antialiased   -> eps2d=0.3, rasterize_mode=antialiased  (gsplat default)
#   DECODER=classic       -> eps2d=0.3, rasterize_mode=classic
# Both presets use the standard eps2d=0.3 and differ only in rasterize_mode.
# To use a non-standard eps2d (e.g. the 0.0001 some of our runs trained with),
# set DECODER_EPS2D directly, or add a preset case below:
#   DECODER_EPS2D / DECODER_RASTERIZE_MODE  set individual fields directly
#   (these win over the preset, and either may be set on its own).
#
# Sets two variables for the sourcing script:
#   DECODER_OPTS  — bash array of hydra overrides (empty when no override)
#   DECODER_TAG   — output-dir suffix so overridden runs don't collide with
#                   per-checkpoint runs (empty when no override)
# =============================================================================

case "${DECODER:-}" in
  "")          ;;
  antialiased) : "${DECODER_EPS2D:=0.3}"; : "${DECODER_RASTERIZE_MODE:=antialiased}" ;;
  classic)     : "${DECODER_EPS2D:=0.3}"; : "${DECODER_RASTERIZE_MODE:=classic}" ;;
  *)           echo "[decoder_opts.sh] Unknown DECODER preset: '${DECODER}' (expected antialiased|classic)" >&2; exit 1 ;;
esac

DECODER_OPTS=()
[[ -n "${DECODER_EPS2D:-}" ]]          && DECODER_OPTS+=("scene_trainer.decoder.eps2d=${DECODER_EPS2D}")
[[ -n "${DECODER_RASTERIZE_MODE:-}" ]] && DECODER_OPTS+=("scene_trainer.decoder.rasterize_mode=${DECODER_RASTERIZE_MODE}")

# Output-dir tag (kept in sync with DECODER_OPTS so predicted and produced dirs match).
# Encodes the actual settings, e.g. mode=antialiased + eps2d=0.3 -> "_antialiased_03".
DECODER_TAG=""
if [[ ${#DECODER_OPTS[@]} -gt 0 ]]; then
  DECODER_TAG="_${DECODER_RASTERIZE_MODE:-eps}"
  [[ -n "${DECODER_EPS2D:-}" ]] && DECODER_TAG="${DECODER_TAG}_${DECODER_EPS2D//./}"
fi
