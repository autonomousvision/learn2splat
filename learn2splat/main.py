import contextlib
import os
import random
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.profilers import PyTorchProfiler

from learn2splat.misc.io import cyan
from learn2splat.misc.console import banner, config_table, warn

# Configure beartype and jaxtyping. The import hook instruments every learn2splat
# function with runtime shape/type checks, which adds import and per-call
# overhead. Set LEARN2SPLAT_NO_TYPECHECK=1 to skip it for faster startup when the
# checks aren't needed.
_typecheck = os.environ.get("LEARN2SPLAT_NO_TYPECHECK", "0") != "1"
_import_hook = (
    install_import_hook(("learn2splat",), ("beartype", "beartype"))
    if _typecheck
    else contextlib.nullcontext()
)
with _import_hook:
    from learn2splat.config import setup_cfg, SkipRun
    from learn2splat.dataset.data_module import DataModule
    from learn2splat.loss import get_losses
    from learn2splat.misc.step_tracker import StepTracker
    from learn2splat.misc.wandb_tools import update_checkpoint_path, setup_wandb_logger
    from learn2splat.misc.checkpointing import find_latest_ckpt, load_model_weights
    from learn2splat.meta_trainer.meta_trainer import MetaTrainer


def _fmt_metric(v):
    """Format a progress-bar value as a string: scientific notation for tiny nonzero
    magnitudes (so sub-1e-3 losses don't collapse to ``0.000``), else 3 significant figures.
    Non-numeric values pass through unchanged; tqdm renders string postfix values verbatim."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    if x != 0 and abs(x) < 1e-3:
        return f"{x:.2e}"
    return f"{x:.3g}"


class _ProgressBar(TQDMProgressBar):
    """Default tqdm bar without Lightning's noisy `v_num` (logger version) postfix.

    Also compacts the bar labels (display only; logged/wandb metric names keep their full
    form): drops the shared ``loss/`` prefix and shortens ``target_`` -> ``t_`` and
    ``context_`` -> ``c_``. Values are formatted via :func:`_fmt_metric` (scientific notation
    for tiny magnitudes, so sub-1e-3 losses stay visible instead of rounding to 0.000).
    """

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return {
            k.removeprefix("loss/").replace("target_", "t_").replace("context_", "c_"): _fmt_metric(v)
            for k, v in items.items()
        }


# print torch device info
print(cyan(f"Torch version: {torch.__version__}"))
if torch.cuda.is_available():
    print(cyan(f"CUDA is available. Number of devices: {torch.cuda.device_count()}"))
    for i in range(torch.cuda.device_count()):
        print(cyan(f"Device {i}: {torch.cuda.get_device_name(i)}"))
else:
    print(cyan("CUDA is not available."))


# The config dir is given as an absolute path: `learn2splat.config` as a package name resolves to
# config.py (the typed schema), not to the config/ directory of yaml, so Hydra's package-relative
# lookup finds no yaml when it has no main script to start from (the installed `learn2splat` command).
@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "config"),
    config_name="main",
)
def run(cfg_dict: DictConfig):
    print(cyan("Parsed the command-line config."))
    # Set up configuration.
    try:
        cfg, cfg_dict, eval_cfg = setup_cfg(cfg_dict)
    except SkipRun as e:
        print(cyan(f"Skipping run: {e}"))
        sys.exit(0)

    print_important_cfg_flags(cfg)

    if cfg.debug_cfg:
        print(cyan("=" * 60))
        print(cfg)
        print(cyan("=" * 60))
        print(cyan(f"Config debug mode, exiting.."))
        exit(0)

    # Set up logging with wandb.
    callbacks = []
    logger = setup_wandb_logger(cfg, cfg_dict)
    if isinstance(logger, WandbLogger):
        callbacks.append(LearningRateMonitor("step", True))

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            cfg.output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Show the tqdm bar in test, and during training only on an interactive terminal —
    # a non-tty (e.g. SLURM log files) would get spammed by bar refreshes.
    show_progress_bar = cfg.mode == "test" or sys.stdout.isatty()
    if show_progress_bar:
        callbacks.append(_ProgressBar())

    # Prepare the checkpoint for loading.
    if cfg.checkpointing.resume:
        if not os.path.exists(cfg.output_dir / 'checkpoints'):
            checkpoint_path = None
        else:
            checkpoint_path = find_latest_ckpt(cfg.output_dir / 'checkpoints')
            # Pass to Lightning via ckpt_path — it restores weights, optimizer, scheduler, and step.
            # Do not also set pretrained_model; that would double-load the weights.
            print(f'resume from {checkpoint_path}')
    else:
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    strategy = cfg.meta_trainer.get_dist_strategy(cfg.scene_trainer)

    if cfg_dict.profiling.mode == "basic":
        profiler = "simple"
    elif cfg_dict.profiling.mode == "advanced":
        profiler = "advanced"
    elif cfg_dict.profiling.mode == "pytorch":
        # wall clock time not representative of true wall clock time
        profiler = PyTorchProfiler(filename="profile-logs")  # saves separate reports per rank when distributed training
    else:
        profiler = None

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        logger=logger,
        devices=torch.cuda.device_count() if torch.cuda.is_available() else "auto",
        strategy=strategy,
        callbacks=callbacks,
        val_check_interval=cfg.meta_trainer.val_check_interval,
        enable_progress_bar=show_progress_bar,
        gradient_clip_val=cfg.meta_trainer.gradient_clip_val if not cfg.scene_trainer.use_fsdp else 0.,
        # clip by norm is not supported by fsdp
        max_steps=cfg.meta_trainer.max_steps,
        num_sanity_val_steps=cfg.meta_trainer.num_sanity_val_steps,
        num_nodes=cfg.meta_trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
        limit_test_batches=cfg.meta_trainer.limit_test_batches,
        limit_train_batches=cfg.meta_trainer.limit_train_batches,
        inference_mode=False,  # never use inference mode to allow autograd graph construction
        profiler=profiler,
    )

    seed = cfg.seed + trainer.global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create the model (MetaTrainer wraps SceneTrainer)
    meta_trainer = MetaTrainer(
        cfg=cfg,
        losses=get_losses(cfg.loss),
        step_tracker=step_tracker,
        eval_data_cfg=(None if eval_cfg is None else eval_cfg.dataset),
    )

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))
    else:
        print("test:", len(data_module.test_dataloader()))

    strict_load = not cfg.checkpointing.no_strict_load

    if cfg.mode == "train":
        assert cfg.scene_trainer.train_scene_opt, \
            "Scene optimizer is frozen. Nothing to train."
        load_model_weights(cfg, meta_trainer.scene_trainer, strict_load, mode="train")
        trainer.fit(meta_trainer, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        load_model_weights(cfg, meta_trainer.scene_trainer, strict_load, mode="test")
        trainer.test(
            meta_trainer,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


def print_important_cfg_flags(cfg):
    def kv(param_name):
        """Return (param_name, value) for a param known to exist."""
        return param_name, eval(param_name, {"cfg": cfg})

    def maybe(param_name):
        """Return (param_name, value), or None if the attribute is absent."""
        try:
            return kv(param_name)
        except AttributeError:
            return None

    def present(*rows):
        """Drop rows that `maybe` resolved to None."""
        return [r for r in rows if r is not None]

    if cfg.scene_trainer.scene_optimizer is None:
        optimizer_rows = [("cfg.scene_trainer.scene_optimizer", "None")]
    else:
        optimizer_rows = present(
            maybe("cfg.scene_trainer.scene_optimizer.name"),
            maybe("cfg.scene_trainer.scene_optimizer.init_state_wo_features"),
            maybe("cfg.scene_trainer.scene_optimizer.init_state_scale"),
            maybe("cfg.scene_trainer.scene_optimizer.init_state_type"),
            maybe("cfg.scene_trainer.scene_optimizer.use_fused_attn"),
            maybe("cfg.scene_trainer.scene_optimizer.knn_idx_update_every"),
            maybe("cfg.scene_trainer.scene_optimizer.update_only_nonzero_grad"),
        )

    sections = {
        "Output dir": [kv("cfg.output_dir"), kv("cfg.mode")],
        "Scene trainer": [
            kv("cfg.scene_trainer.opt_batch_size"),
            kv("cfg.scene_trainer.opt_batch_strategy"),
        ],
        "Checkpoints": [
            kv("cfg.checkpointing.pretrained_model"),
            kv("cfg.checkpointing.pretrained_optimizer"),
            kv("cfg.checkpointing.pretrained_initializer"),
            kv("cfg.checkpointing.no_strict_load"),
        ],
        "Optimizer": optimizer_rows,
        "Initialization": present(
            kv("cfg.scene_trainer.scene_initializer.name"),
            maybe("cfg.scene_trainer.scene_initializer.path"),
            maybe("cfg.scene_trainer.scene_initializer.dl3dv_settings"),
            maybe("cfg.scene_trainer.scene_initializer.eval_fixed_gaussians_num"),
            maybe("cfg.scene_trainer.scene_initializer.filter_zero_rgb"),
        ),
        "Dataset": present(
            kv("cfg.dataset.name"),
            maybe("cfg.dataset.test_start_idx"),
            maybe("cfg.dataset.num_scenes"),
            maybe("cfg.dataset.view_sampler.name"),  # absent for dataset=mixed (per-source sampler)
            maybe("cfg.dataset.view_sampler.num_context_views"),
            maybe("cfg.dataset.view_sampler.index_path"),
            maybe("cfg.dataset.image_shape"),
            maybe("cfg.dataset.ori_image_shape"),
        ),
        "Training": present(maybe("cfg.loss")),
    }
    config_table(sections, title="Important config params")


def main():
    """Console entry point. Equivalent to `python -m learn2splat.main`."""
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    if not torch.cuda.is_available():
        warn("CUDA is not available, running on CPU.")

    banner(
        "learn2splat",
        [
            f"host          {os.uname().nodename}",
            f"slurm job id  {os.environ.get('SLURM_JOB_ID', 'N/A')}",
            f"slurm gpus    {os.environ.get('SLURM_STEP_GPUS', 'N/A')}",
            f"working dir   {Path.cwd()}",
        ],
    )

    run()


if __name__ == "__main__":
    main()
