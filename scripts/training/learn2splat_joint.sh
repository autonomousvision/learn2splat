# Trains the released Learn2Splat checkpoint: ONE optimizer covering both view regimes.
#
# The mixed dataset interleaves two DL3DV sources 1:1, each with its own initializer -- dense
# (64 context views, COLMAP SfM init) and sparse (8 context views, resplat_v1 feed-forward init).
# Each source keeps its own data shim, and the optimizer state is seeded per initializer (from
# features for resplat, synthesized for COLMAP), so one set of weights serves both settings.
# Validation runs on the dense source. The recipe, the per-source view samplers and the
# initializers live in learn2splat/config/experiment/train_mixed_sparse_dense.yaml.
#
# Launch through SLURM with:
#   sbatch scripts/training/submit_train.slurm scripts/training/learn2splat_joint.sh
python -m learn2splat.main +experiment=train_mixed_sparse_dense \
meta_trainer.num_sanity_val_steps=1 \
meta_trainer.train.eval_data_length=5 \
meta_trainer.train.loss_on_target_views_num=6 \
meta_trainer.max_steps=150000 \
loss.deltas.weight=2.0 \
wandb.project=learn2splat \
output_dir='checkpoints/learn2splat/joint' \
log_slurm_id=true