# Training scripts for dl3dv with 8 input views
python -m learn2splat.main +experiment=train_l2s_sparse_dl3dv \
meta_trainer.num_sanity_val_steps=1 \
meta_trainer.train.eval_data_length=5 \
meta_trainer.train.loss_on_target_views_num=6 \
meta_trainer.max_steps=100000 \
wandb.project=learn2splat \
wandb.notes="" \
output_dir='checkpoints/learn2splat/dl3dv/sparse_8views_resplat_init' \
meta_trainer.test.save_at_iters=[0,1,5,10,50,100,200,300,400,500,1000] \
log_slurm_id=true