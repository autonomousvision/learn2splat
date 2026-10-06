#!/usr/bin/env bash
# =============================================================================
# _common/common_opts.sh
# =============================================================================
# Populates COMMON_OPTS with Hydra overrides shared across ALL datasets.
# Source this AFTER setting the required variables:
#
#   Required:
#     NUM_STEPS      — optimisation steps
#     OPT_BATCH_SIZE — optimisation batch size
#     SCENE_NUM      — total number of scenes (used for limit_test_batches default)
#
# After sourcing, append dataset-specific overrides:
#   COMMON_OPTS+=(
#     "dataset=dl3dv"
#     ...
#   )
# =============================================================================

COMMON_OPTS=(
  # --- experiment / mode ---
  # NOTE: +experiment must be set by each dataset's test.sh
  # mode, decoder, and test flags are set in the test experiment configs.
  "checkpointing.pretrained_depth=null"
  "checkpointing.pretrained_model=null"

  # --- Scene trainer ---
  # Trainer
  "scene_trainer.num_update_steps=${NUM_STEPS}"
  "scene_trainer.opt_batch_size=${OPT_BATCH_SIZE}"
  "scene_trainer.opt_batch_strategy=fps"

  # Decoder defaults come from the selected config group (gsplat.yaml or inria.yaml).
  # Override the group via EXTRA_OPTS, e.g. "scene_trainer/decoder=inria".

  # Optimizer
  "scene_trainer.scene_optimizer.input_gradients_chunk_size=-1"

  # --- scene limit ---
  "meta_trainer.limit_test_batches=${SCENE_NUM}"
)

# --- optional decoder override (opt-in; applies to ours + baselines) ---
source "${COMMON_DIR}/decoder_opts.sh"
COMMON_OPTS+=("${DECODER_OPTS[@]}")


