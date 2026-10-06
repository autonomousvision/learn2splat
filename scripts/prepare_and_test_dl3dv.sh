#!/usr/bin/env bash
# =============================================================================
# DL3DV dense pipeline, end to end (defaults to a tiny debug subset).
# Run with:  bash scripts/prepare_and_test_dl3dv.sh
#
# Steps: download benchmark scenes -> convert to chunks -> build index
#        -> (optional) clean raw images -> run mode=test -> show results
#        -> run a short training smoke test.
#
# LIMIT controls how many scenes to download. With a LIMIT the data goes under
# datasets/dl3dv-debug/ using the SAME layout as the full dataset, so switching
# to the full run is just LIMIT="" (which writes to the real datasets/ paths).
#
# Activate the env first (locally: `conda activate resplat`).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

# ----- config (edit me) ------------------------------------------------------
LIMIT=5                  # scenes to download; set LIMIT="" for the FULL dataset
NUM_UPDATE_STEPS=100     # optimizer steps at test time
TRAIN_STEPS=2            # meta-training steps for the smoke test
CLEAN=0                  # 1 = delete the raw image frames once chunks are built
# -----------------------------------------------------------------------------

# Colored step banner.
banner() { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }


# -----------------------------------------------------------------------------
# 0/6 Paths setting
# With a LIMIT, mirror the real dataset tree under datasets/dl3dv-debug/.
# Without one, write straight to the real dataset paths.
if [[ -n "$LIMIT" ]]; then
  DATA_ROOT="datasets/dl3dv-debug"
else
  DATA_ROOT="datasets"
fi
BENCHMARK_DIR="$DATA_ROOT/dl3dv-benchmark"      # COLMAP SfM (colmap initializer path)
CHUNKS_DIR="$DATA_ROOT/dl3dv-480p-chunks"       # .torch chunks (dataset roots)
# -----------------------------------------------------------------------------
# Two data products feed two components:
#   - the .torch CHUNKS ($CHUNKS_DIR) are read by the DATASET (the input/target images);
#   - the COLMAP SfM ($BENCHMARK_DIR/<scene>/sparse) is read by the INITIALIZER (the point cloud).
# The downloaded raw image frames are only an intermediate for building the chunks.

banner "1/6  Download $LIMIT benchmark scene(s): images (images_4, 540x960) + COLMAP SfM"
python -m learn2splat.scripts.dl3dv_download \
  --source benchmark --content images+sfm --limit "$LIMIT" --clean_cache \
  --odir "$BENCHMARK_DIR"

banner "2/6  Convert downloaded images -> .torch chunks"
python -m learn2splat.scripts.convert_dl3dv_test \
  --input_dir "$BENCHMARK_DIR" \
  --output_dir "$CHUNKS_DIR" \
  --img_subdir images_4

banner "3/6  Build the chunk index (index.json)"
python -m learn2splat.scripts.generate_dl3dv_index --data_dir "$CHUNKS_DIR"

# Free the raw image frames now that the chunks hold them. The dataset reads the
# chunks; only the COLMAP SfM (sparse/) is still needed (by the initializer), so
# we drop benchmark/<scene>/nerfstudio and keep sparse/ + transforms.json.
if [[ "$CLEAN" == "1" ]]; then
  banner "Clean: removing raw images (keeping COLMAP SfM for the initializer)"
  rm -rf "$BENCHMARK_DIR"/*/nerfstudio
fi

banner "4/6  Dense checkpoint: fetched from HF on first use (hf:// ref below)"
# Pass the checkpoint as an hf:// ref directly. mode=test reads the model
# architecture from the config.yaml next to the checkpoint; the resolver fetches
# both the .ckpt and its sibling config.yaml into ./checkpoints on first use, so
# no manual download of the whole model folder is needed.
HF_CKPT="hf://autonomousvision/learn2splat/joint/checkpoints/epoch_7-step_150000.ckpt"

# Dense COLMAP-path settings shared by test and train:
#   - ori_image_shape = the chunk's native size (images_4 = 540x960); image_shape = the
#     eval/train resolution (256x448; this 8-view index is low-res, and full-res OOMs a 24GB GPU).
#   - learn2splat is the dense optimizer (the published checkpoint's optimizer is l2s).
#   - COLMAP points have no features, so the optimizer needs init_state_wo_features=true.

banner "5/6  mode=test: optimize each scene and score novel views"
python -m learn2splat.main +experiment=test_dl3dv \
  "dataset.roots=[$CHUNKS_DIR]" \
  'dataset.image_shape=[256,448]' \
  'dataset.ori_image_shape=[540,960]' \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.view_sampler.num_context_views=8 \
  scene_trainer/scene_initializer=colmap \
  scene_trainer.scene_initializer.path="$BENCHMARK_DIR" \
  scene_trainer.scene_initializer.dl3dv_settings=true \
  scene_trainer.scene_initializer.eval_fixed_gaussians_num=null \
  scene_trainer/scene_optimizer=learn2splat \
  scene_trainer.scene_optimizer.init_state_wo_features=true \
  scene_trainer.scene_optimizer.init_state_type=random \
  scene_trainer.scene_optimizer.init_state_scale=1.0 \
  checkpointing.pretrained_optimizer="$HF_CKPT" \
  scene_trainer.num_update_steps="$NUM_UPDATE_STEPS" \
  meta_trainer.test.compute_scores=true \
  output_dir="$DATA_ROOT/results-test"

banner "Test results"
python scripts/show_results_table.py "$DATA_ROOT/results-test"

# 6. Training smoke test: a few meta-steps to confirm the train loop runs.
#    Training reads the "train" split, so reuse the converted scenes as a tiny train set.
banner "6/6  Training smoke test ($TRAIN_STEPS steps)"
cp -rn "$CHUNKS_DIR/test" "$CHUNKS_DIR/train"
python -m learn2splat.main +experiment=train_dl3dv \
  mode=train \
  "dataset.roots=[$CHUNKS_DIR]" \
  'dataset.image_shape=[256,448]' \
  'dataset.ori_image_shape=[540,960]' \
  dataset/view_sampler=evaluation \
  dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
  dataset.view_sampler.num_context_views=8 \
  scene_trainer/scene_optimizer=learn2splat \
  scene_trainer/scene_initializer=colmap \
  scene_trainer.scene_initializer.path="$BENCHMARK_DIR" \
  scene_trainer.scene_initializer.dl3dv_settings=true \
  scene_trainer.scene_initializer.eval_fixed_gaussians_num=null \
  scene_trainer.scene_optimizer.init_state_wo_features=true \
  scene_trainer.scene_optimizer.init_state_type=random \
  scene_trainer.scene_optimizer.init_state_scale=1.0 \
  scene_trainer.train_scene_opt=true \
  scene_trainer.num_update_steps=2 \
  scene_trainer.train_min_refine=1 \
  scene_trainer.train_max_refine=2 \
  "meta_trainer.max_steps=$TRAIN_STEPS" \
  meta_trainer.num_sanity_val_steps=0 \
  meta_trainer.val_check_interval=1.0 \
  meta_trainer.train.eval_model_every_n_val=0 \
  data_loader.train.num_workers=1 \
  wandb.mode=disabled \
  output_dir="$DATA_ROOT/results-train"

banner "Done."