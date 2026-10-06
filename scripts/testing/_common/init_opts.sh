#!/usr/bin/env bash
# _common/init_opts.sh
# ---------------------------------------------------------------------------
# Appends initializer-related Hydra overrides to COMMON_OPTS based on INIT.
# Sourced ONCE per test.sh invocation (section ④), before run_experiments.sh.
#
# Expects:
#   INIT        — resplat_v1 | dense_colmap | sparse_colmap | ply
#   COMMON_OPTS — bash array to append to
#   path var for the chosen INIT: DENSE_COLMAP_PATH / SPARSE_COLMAP_PATH / PLY_PATH
#
# Optional extra-opts arrays, appended in the matching case:
#   RESPLAT_EXTRA_OPTS, DENSE_COLMAP_EXTRA_OPTS, SPARSE_COLMAP_EXTRA_OPTS, PLY_EXTRA_OPTS
#
# Does NOT set INIT_STATE_OPTS: those init_state_* overrides are computed by
# _resolve_init_state_opts() in run_experiments.sh (for "ours").

case "$INIT" in

  # -------------------------------------------------------------------------
  # Learned initializer (ReSplat / Learn2Splat)
  # -------------------------------------------------------------------------
  "resplat_v1")
    COMMON_OPTS+=(
      "scene_trainer/scene_initializer=resplat_v1"
      "checkpointing.pretrained_initializer=hf://autonomousvision/learn2splat/resplat_init/checkpoints/epoch_20-step_100000.ckpt"
      "${RESPLAT_EXTRA_OPTS[@]}"
    )
    ;;

  # -------------------------------------------------------------------------
  # Dense SfM / COLMAP point cloud initializer
  # -------------------------------------------------------------------------
  "dense_colmap")
    COMMON_OPTS+=(
      "scene_trainer/scene_initializer=colmap"
      "scene_trainer.scene_initializer.path=${DENSE_COLMAP_PATH}"
      "scene_trainer.scene_initializer.eval_fixed_gaussians_num=null"
      "${DENSE_COLMAP_EXTRA_OPTS[@]}"
    )
    ;;

  # -------------------------------------------------------------------------
  # Sparse SfM / COLMAP point cloud initializer (reconstructions from haofei)
  # -------------------------------------------------------------------------
  "sparse_colmap")
    COMMON_OPTS+=(
      "scene_trainer/scene_initializer=colmap"
      "scene_trainer.scene_initializer.path=${SPARSE_COLMAP_PATH}"
      "scene_trainer.scene_initializer.filter_zero_rgb=false"
      "scene_trainer.scene_initializer.dl3dv_settings=true"
      "scene_trainer.scene_initializer.points3d_ply_filename=input.ply"
      "scene_trainer.scene_initializer.override_dataset_poses=false"
      "${SPARSE_COLMAP_EXTRA_OPTS[@]}"
    )
    ;;

  # -------------------------------------------------------------------------
  # Pre-computed PLY point cloud (e.g. from a prior 3DGS run)
  # Required: PLY_PATH  — base dir containing per-scene subdirectories
  # Optional: PLY_FILENAME — relative path under scene dir (default: gaussians.ply)
  #           e.g. PLY_FILENAME="iteration_20000/point_cloud.ply"
  # -------------------------------------------------------------------------
  "ply")
    COMMON_OPTS+=(
      "scene_trainer/scene_initializer=ply"
      "scene_trainer.scene_initializer.path=${PLY_PATH}"
      "scene_trainer.scene_initializer.ply_filename=${PLY_FILENAME:-gaussians.ply}"
      "${PLY_EXTRA_OPTS[@]}"
    )
    ;;

  *)
    echo "[init_opts.sh] Unknown INIT type: '${INIT}'. Expected 'resplat_v1', 'dense_colmap', 'sparse_colmap', or 'ply'." >&2
    exit 1
    ;;
esac
