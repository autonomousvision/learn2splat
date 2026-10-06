"""Outer training loop of the two-level pipeline.

`MetaTrainer` is the PyTorch Lightning module that drives meta-learning: it iterates over scenes,
delegates the per-scene initialize -> optimize -> render work to `SceneTrainer`, computes the losses on
the rendered novel views, and meta-optimizes the SceneTrainer's parameters. Also handles the few-shot
ckpt buffer, wandb logging, evaluation (PSNR/SSIM/LPIPS, depth), and test-time output (videos, depth,
saved Gaussians). The inner per-scene loop lives in `scene_trainer.py`.
"""

import json
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, runtime_checkable, Protocol

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import wandb
from einops import rearrange, repeat, pack
from jaxtyping import Float
from lightning_fabric.utilities import rank_zero_only
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import optim, Tensor, nn
from tqdm import tqdm

from learn2splat.config import RootCfg, MetaTrainerCfg
from learn2splat.dataset import DatasetCfg
from learn2splat.dataset.data_types import BatchedExample, BatchedViews
from learn2splat.evaluation.depth_metrics import compute_depth_errors
from learn2splat.evaluation.metrics import compute_psnr, compute_rgb_metrics
from learn2splat.loss import Loss
from learn2splat.loss.loss_stability import LossStability
from learn2splat.meta_trainer.ckpt_buffer import GaussianEpisodeEntry
from learn2splat.misc.LocalLogger import LocalLogger, LOG_PATH
from learn2splat.misc.batchify import batched_select
from learn2splat.misc.benchmarker import Benchmarker
from learn2splat.misc.console import rule, warn
from learn2splat.misc.general_utils import SkipBatchException
from learn2splat.misc.image_io import prep_image, save_video, save_image
from learn2splat.misc.io import CustomPath
from learn2splat.misc.stabilize_camera import render_stabilization_path
from learn2splat.misc.step_tracker import StepTracker
from learn2splat.model.decoder import get_decoder
from learn2splat.model.decoder.decoder import DecoderOutput, DepthRenderingMode
from learn2splat.model.ply_export import save_gaussian_ply
from learn2splat.paths import DEBUG
from learn2splat.scene_trainer.initializer.initializer import InitializerOutput, Initializer
from learn2splat.scene_trainer.optimizer.optimizer import OptimizerPreviousOutput, OptimizerOutput, Optimizer
from learn2splat.scene_trainer.postprocessing import PostProcessing3DGS
from learn2splat.scene_trainer.scene_trainer import SceneTrainer  # Use existing SceneTrainer
from learn2splat.scene_trainer.scene_trainer_cfg import SceneTrainerCfg
from learn2splat.meta_trainer.meta_trainer_cfg import MetaOptimizerCfg, TestCfg, TrainCfg
from learn2splat.visualization.annotation import add_label
from learn2splat.visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from learn2splat.visualization.camera_trajectory.orbit import generate_orbit_path
from learn2splat.visualization.camera_trajectory.wobble import generate_wobble, generate_wobble_transformation
from learn2splat.visualization.color_map import apply_color_map_to_image
from learn2splat.visualization.layout import hcat, vcat, add_border
from learn2splat.visualization.vis_depth import viz_depth_tensor

try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    pass

try:
    import moviepy.editor as mpy
except ImportError:
    import moviepy as mpy


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
            self,
            t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


slurm_id_logged = False


class _SkipStepException(Exception):
    """Raised inside meta_training_step to signal that this step should be
    skipped.  Caught in training_step, which then does a single all_reduce so
    every rank skips together — preventing NCCL hangs."""
    pass


def _check_finite_loss(loss: Tensor, name: str) -> None:
    """Guard a single loss component. If it is NaN/Inf, raise SkipBatchException naming the
    offending loss (so the cause is attributable) — training_step then skips the batch rather than
    letting a bad value reach backprop. To crash instead of skipping, raise a hard error here."""
    if not torch.isfinite(loss).all():
        raise SkipBatchException(f"non-finite loss '{name}' = {loss.detach()}")


class MetaTrainer(LightningModule):
    """
    Meta-level trainer that handles the outer loop of meta-learning.

    This class focuses on:
    - Meta-level training loop and ckpt buffer management
    - Delegating scene-level optimization to the existing SceneTrainer
    - Meta-optimization of the SceneTrainer's parameters
    """

    meta_optimizer_cfg: MetaOptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    logger: Optional[WandbLogger]
    scene_trainer_cfg: SceneTrainerCfg
    losses: nn.ModuleList
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]
    meta_trainer_cfg: MetaTrainerCfg

    def __init__(
            self,
            cfg: RootCfg,
            losses: list[Loss],
            step_tracker: StepTracker | None,
            eval_data_cfg: Optional[DatasetCfg] = None,
    ) -> None:
        # All sub-configs are read from cfg.
        super().__init__()
        self.meta_optimizer_cfg = cfg.meta_trainer.meta_optimizer
        self.test_cfg = cfg.meta_trainer.test
        self.train_cfg = cfg.meta_trainer.train
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.scene_trainer_cfg = cfg.scene_trainer
        self.meta_trainer_cfg = cfg.meta_trainer

        # Single benchmarker shared with the SceneTrainer so all timings accumulate in one place.
        self.benchmarker = Benchmarker()

        # The first optimized scene of a test pass pays a one-time GPU warm-up (kernel JIT,
        # cuDNN autotune, CUDA context). This flag (reset in on_test_epoch_start) lets
        # _optimize_and_measure_scene zero that scene's first iteration so reported timings are steady-state.
        self._timing_warmup_done = False

        # The SceneTrainer holds all the scene-level logic: the initializer, the optimizer,
        # the decoder, and the per-scene optimize_scene loop.
        self.scene_trainer = SceneTrainer(
            test_cfg=self.test_cfg,
            train_cfg=self.train_cfg,
            scene_trainer_cfg=self.scene_trainer_cfg,
            decoder=get_decoder(cfg.scene_trainer.decoder, cfg.dataset),
            step_tracker=step_tracker,
            benchmarker=self.benchmarker,
            eval_data_cfg=eval_data_cfg,
        )

        self.losses = nn.ModuleList(losses)
        self._depth_mode: DepthRenderingMode = 'depth'

        # Testing utilities
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs_target = defaultdict(list)
            self.test_step_outputs_context = defaultdict(list)

        if cfg.mode == "train" and self.train_cfg.use_ckpt_buffer and self.scene_trainer_cfg.num_update_steps > 0:
            assert self.scene_optimizer is not None
            assert self.scene_optimizer.strategy == "learned"

            if getattr(self.scene_optimizer.cfg, 'concat_init_state', False):
                raise NotImplementedError("Ckpt buffer with concat_init_state is not supported")
            if getattr(self.scene_optimizer.cfg, 'replace_init_state', False):
                raise NotImplementedError("Ckpt buffer with replace_init_state is not supported")
            from learn2splat.meta_trainer.ckpt_buffer import EpisodeCkptBuffer
            self.buffer = EpisodeCkptBuffer(self.train_cfg.ckpt_buffer_cfg)
        else:
            self.buffer = None

        self._use_dataloader_batch = True  # default
        self._new_scenes_cnt = -1
        self.gaussian_timestep_list = []
        self._run_train_comparison = False

        self.promoting_buffer_sample = False

    # ==================== Lightning Hooks ====================

    def on_before_batch_transfer(self, batch: BatchedExample, dataloader_idx: int) -> BatchedExample:
        """Decide before device transfer whether this step should draw from the ckpt buffer or the dataloader."""
        # Decide whether we'll use the buffer
        if self.training and self.buffer is not None and self.buffer.should_sample():
            self._use_dataloader_batch = False
        else:
            self._use_dataloader_batch = True
        return batch

    def on_after_batch_transfer(self, batch: BatchedExample, dataloader_idx: int) -> BatchedExample:
        """Convert raw context/target dicts into typed BatchedViews after the batch lands on the device."""
        batch["context"] = BatchedViews.from_dict(batch["context"])
        batch["target"] = BatchedViews.from_dict(batch["target"])
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Skip the device transfer when the upcoming training step will draw from the ckpt buffer.

        `on_before_batch_transfer` sets `_use_dataloader_batch=False` whenever the buffer wins the
        coin flip; in that case we ignore the dataloader batch entirely and the buffer-sample path
        moves its own tensors. Validation/test always transfer.
        """
        if self.training:
            should_move = self._use_dataloader_batch  # buffer-sample path moves the batch itself
        else:
            should_move = True  # always move during validation and testing

        if should_move:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            return batch  # Don't move — we're going to ignore this batch anyway

    def on_save_checkpoint(self, checkpoint):
        # Remove the monodepth_model weights from the checkpoint
        if 'state_dict' in checkpoint:
            keys_to_remove = [k for k in checkpoint['state_dict'] if k.startswith('pretrained_monodepth')]
            for k in keys_to_remove:
                del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        # Override scheduler total_steps to match current max_steps so LR doesn't
        # hit 0 early when resuming for extended training.
        for scheduler in checkpoint.get("lr_schedulers", []):
            saved_steps = scheduler.get("total_steps")
            if saved_steps is not None and saved_steps != self.trainer.max_steps:
                print(
                    f"Resuming with extended training: scheduler total_steps "
                    f"{saved_steps} → {self.trainer.max_steps}. "
                    f"LR schedule will be stretched, not restarted from scratch."
                )
                scheduler["total_steps"] = self.trainer.max_steps

    def on_train_epoch_start(self):
        """Handle epoch start for scene-based training."""
        if hasattr(self.scene_trainer, 'on_train_epoch_start'):
            return self.scene_trainer.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        if self.global_rank == 0:
            if self.buffer is not None:
                print(f"Buffer size: {len(self.buffer)}")

            if self.logger is not None and isinstance(self.logger, WandbLogger):
                wandb.log({"ckpt_buffer/gaussian_timestep_histogram": wandb.Histogram(self.gaussian_timestep_list)})

        if self.buffer is not None:
            self.buffer.clear()
        self.gaussian_timestep_list = []

    def on_validation_start(self) -> None:
        if self.global_rank == 0:
            self._run_train_comparison = True

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    def on_test_epoch_start(self):
        """Handle test epoch start."""
        self._timing_warmup_done = False
        if hasattr(self.scene_trainer, 'on_test_epoch_start'):
            return self.scene_trainer.on_test_epoch_start()

    def on_test_epoch_end(self):
        """Handle test epoch end."""
        if hasattr(self.scene_trainer, 'on_test_epoch_end'):
            return self.scene_trainer.on_test_epoch_end()

    def on_test_end(self) -> None:
        out_dir = self.test_cfg.output_path

        # saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for output_dict, input_str in zip([self.test_step_outputs_context, self.test_step_outputs_target],
                                              ["context", "target"]):
                for metric_name, metric_scores in output_dict.items():
                    if len(metric_scores) == 0 or max(len(row) for row in metric_scores) == 0:
                        continue
                    matrix, per_step_mean, scenes_per_step = self._reduce_partial_metric(metric_scores)
                    print(input_str, metric_name, per_step_mean, "scenes_per_step:", scenes_per_step)
                    with (out_dir / "metrics" / f"{input_str}_{metric_name}.json").open("w") as f:
                        json.dump(matrix, f)

            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "metrics" / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "metrics" / "peak_memory.json")
            self.benchmarker.summarize()

    @staticmethod
    def _reduce_partial_metric(
        metric_scores: list[list[float | int]],
    ) -> tuple[list[list[float]], list[float], list[int]]:
        """Average a metric's per-scene rows across scenes, tolerating partial scenes.

        Rows hold floats (psnr/ssim/...) or ints (iteration numbers).

        A scene that stopped early has fewer steps, so the rows can be ragged; they are NaN-padded to
        the longest. Returns (padded matrix [scenes, steps], per-step mean, per-step scene count). The
        mean is NaN at any step not every scene reached -- flagging it as not comparable rather than
        averaging the survivors; the count records how many scenes reached each step.
        """
        max_steps = max(len(row) for row in metric_scores)
        padded = [row + [float("nan")] * (max_steps - len(row)) for row in metric_scores]
        mat = torch.tensor(padded).float()  # [scenes, steps]
        per_step_mean = mat.mean(dim=0).tolist()
        scenes_per_step = (~torch.isnan(mat)).sum(dim=0).tolist()
        return mat.tolist(), per_step_mean, scenes_per_step

    # ==================== Training ====================

    def training_step(self, batch, batch_idx):
        """
        This is a meta trainer class. Each training step and test step corresponds to training on one scene.
        We delegate the actual training to meta_training_step and meta_test_step.
        The loop over inner training steps (training within a specific scene) is performed in
        self.optimize_scene.
        Each inner iteration is done by calling the forward call of the optimizer.
        """
        # DDP-safe skip: all ranks must call all_reduce together, so we do it
        # here in training_step (which is always called on every rank) rather
        # than inside meta_training_step (which may return early on only one rank).
        #
        # Each rank sets skip_flag=1 if it wants to skip, 0 otherwise.
        # After MAX all_reduce, every rank sees 1 if *any* rank wants to skip,
        # and all return zero loss together — keeping NCCL collectives in sync.
        is_dist = dist.is_available() and dist.is_initialized()
        wants_skip = torch.zeros(1, device=self.device)

        try:
            loss = self.meta_training_step(batch, batch_idx)
        except _SkipStepException:
            wants_skip.fill_(1)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        if is_dist:
            dist.all_reduce(wants_skip, op=dist.ReduceOp.MAX)
            if wants_skip.item() > 0:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def meta_training_step(self, scene_batch, batch_idx):
        """One meta-training step: initialize Gaussians, run optimizer refinement, compute loss, optionally push to ckpt buffer."""
        optimizer_output: OptimizerOutput | None = None

        # Prepare input (from dataloader or ckpt buffer)
        if self._use_dataloader_batch:
            # Use new batch from dataloader
            scene_batch: BatchedExample = self.scene_trainer.apply_initializer_data_shim(scene_batch)

            # Get initialization Gaussians + renders. Context render is needed only when the
            # optimizer step loss reads from position 0 of the render lists. Target render is
            # always produced so the splice in optimize_scene can align gaussian_list
            # and target_render_list. Grads on init render are only needed when training the
            # initializer; same condition gates the depth render.
            try:
                init_output = self.init_gaussians_and_render(
                    scene_batch,
                    visualization_dump={},
                    render_context=self.scene_trainer_cfg.train_scene_opt,
                    render_target=True,
                    grad_enabled=False,  # the initializer is frozen, so initialization needs no graph
                )
            except SkipBatchException as e:
                self.log("skip_zero_gaussians_batch", 1, prog_bar=True)
                if self.global_rank == 0:
                    warn(f"Skipping batch {batch_idx} due to {e}. t meta {self.global_step}")
                raise _SkipStepException(f"SkipBatch(init): {e}")

            prev_output = init_output
            curr_inner_iter = 0
            self._new_scenes_cnt += 1
        else:
            # Resample from ckpt buffer intermediate optimized Gaussians (only when training the optimizer)
            assert self.scene_trainer_cfg.train_scene_opt

            # Sample from buffer
            gaussian_episode_entry: GaussianEpisodeEntry = self.buffer.sample(device=self.device)

            scene_batch = gaussian_episode_entry.batch
            curr_inner_iter = gaussian_episode_entry.t

            # Ckpt-buffer path: resume optimization from a buffered intermediate state instead of a
            # fresh initialization (no initializer model runs). prev_output
            # holds the gaussians + optimizer state to resume from; we pre-render its views so
            # optimize_scene can place them at iteration 0 of the optimizer output.
            prev_output = OptimizerPreviousOutput(
                gaussians=gaussian_episode_entry.gaussians,
                state=gaussian_episode_entry.state,
            )
            self.scene_trainer.render_init_views(
                scene_batch, prev_output,
                render_context=True, render_target=True, grad_enabled=False,
            )

            # Resuming from the buffer means there is no separate initializer output, so reuse
            # prev_output as init_output: it already holds the rendered resume views, which
            # _log_init_metrics reads to log the step-0 PSNR. The initializer loss is not computed
            # when only the optimizer is trained, so nothing else is taken from it.
            init_output = prev_output

        # Log the current timestep for analysis
        self.gaussian_timestep_list.append(curr_inner_iter)

        # Optimize the gaussians
        if self.scene_trainer.optimizer is not None and self.scene_trainer_cfg.train_scene_opt:
            # During optimization, we render the context and target images for:
            # 1. error/gradients calculation
            # 2. loss calculation
            # The init render at position 0 of the optimizer_output lists comes from the
            # splice in optimize_scene (see SceneTrainer._insert_init_into_output).

            try:
                optimizer_output: OptimizerOutput = self.optimize_scene(scene_batch, prev_output,
                                                                                 curr_iter=curr_inner_iter,
                                                                                 depth_mode=self._depth_mode)
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                self.log("skip_oom_batch", 1, prog_bar=True)
                print(
                    f"[rank {self.global_rank}]  skipping batch {batch_idx} t meta {self.global_step} t inner {curr_inner_iter}: {e}")
                torch.cuda.empty_cache()
                raise _SkipStepException("OOM")
            except SkipBatchException as e:
                self.log("skip_nan_batch", 1, prog_bar=True)
                if self.global_rank == 0:
                    warn(f"Skipping batch {batch_idx} due to {e}. "
                         f"t meta {self.global_step} t inner {curr_inner_iter}")
                raise _SkipStepException(f"SkipBatch(opt): {e}")
            curr_inner_iter = optimizer_output.t

            if optimizer_output.last_prev_output.state.state is not None:
                state_norm = optimizer_output.last_prev_output.state.state.norm(dim=1).mean()
                self.log("info/state_norm", state_norm)

        # Compute and log loss.
        init_gaussians = init_output.gaussians

        try:
            total_loss = self._calc_total_loss(scene_batch, optimizer_output, init_output)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            self.log("skip_oom_batch", 1, prog_bar=True)
            print(
                f"[rank {self.global_rank}] OOM: {e}, skipping batch {batch_idx} t meta {self.global_step} t inner {curr_inner_iter} num of inner {len(optimizer_output.gaussian_list)}")
            torch.cuda.empty_cache()
            raise _SkipStepException("OOM")
        except SkipBatchException as e:
            # A per-loss guard (_check_finite_loss) flagged a NaN/Inf component — skip this batch.
            self.log("skip_nan_batch", 1, prog_bar=True)
            if self.global_rank == 0:
                warn(f"Skipping batch {batch_idx} due to {e}. "
                     f"t meta {self.global_step} t inner {curr_inner_iter}")
            raise _SkipStepException(f"SkipBatch(loss): {e}")

        # More logging
        if optimizer_output is not None:
            last_gaussians = optimizer_output.gaussian_list[-1]
        else:
            last_gaussians = init_gaussians
        self.train_logging(scene_batch, optimizer_output, last_gaussians, total_loss)

        # Check for NaN loss
        # Skipping pushing to the ckpt buffer
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.log("skip_nan_batch", 1, prog_bar=True)
            if self.global_rank == 0:
                warn(f"Skipping batch {batch_idx} due to NaN loss. "
                     f"t meta {self.global_step} t inner {optimizer_output.t}")
            raise _SkipStepException("NaN/Inf loss")

        if self._use_dataloader_batch:
            self._log_train_comparison(scene_batch, init_output, last_gaussians)

        if self.buffer is not None and self.buffer.should_push(new_sample=self._use_dataloader_batch,
                                                               t=curr_inner_iter):
            self._maybe_push_to_ckpt_buffer(scene_batch, batch_idx, optimizer_output, last_gaussians)

        return total_loss

    def _maybe_push_to_ckpt_buffer(self, scene_batch, batch_idx, optimizer_output, last_gaussians):
        """Optionally roll out the optimizer, then push the current sample to the ckpt buffer.

        Called only when `self.buffer.should_push(...)` already returned True. The rollout
        branch runs extra optimizer iterations under torch.no_grad to produce a "more mature"
        sample to push (controlled by ckpt_buffer_cfg.rollout*); it may reassign
        `optimizer_output` and `last_gaussians` locally for the push.
        """
        push = True
        steps = None
        if self.train_cfg.ckpt_buffer_cfg.rollout:
            steps = self._sample_rollout_steps()
            with torch.no_grad():
                # Set eval mode
                self.eval()
                self.scene_optimizer.save_every.set_all_tags(False)
                self.promoting_buffer_sample = True

                try:
                    optimizer_output = self.optimize_scene(scene_batch, optimizer_output.last_prev_output,
                                                                    curr_iter=optimizer_output.t,
                                                                    num_update_steps=steps,
                                                                    depth_mode=None,
                                                                    disable_tqdm=True)
                    last_gaussians = optimizer_output.last_prev_output.gaussians
                # catching multiple errors
                except (ValueError, SkipBatchException) as e:
                    warn(f"Skipping pushing batch {batch_idx} to buffer due to {e}.")
                    push = False
                self.train()
                self.scene_optimizer.save_every.set_all_tags(True)
                self.promoting_buffer_sample = False

        if optimizer_output.last_prev_output.state.state is not None:
            with torch.no_grad():
                state_norm = optimizer_output.last_prev_output.state.state.norm(dim=1).mean()
            if state_norm > 500:
                warnings.warn(f"Pushing sample norm state {state_norm} {optimizer_output.t} {self.global_step}")

        if push:
            self.buffer.push(GaussianEpisodeEntry(t=optimizer_output.t,
                                                  batch=scene_batch,
                                                  gaussians=last_gaussians,
                                                  state=optimizer_output.last_prev_output.state,
                                                  id=self._new_scenes_cnt), to_cpu=True)

            self.log("ckpt_buffer/size", len(self.buffer.buffer))
            if self.train_cfg.ckpt_buffer_cfg.rollout:
                self.log("ckpt_buffer/rollout", steps)
            self.log("ckpt_buffer/stored_step", optimizer_output.t)

    def _sample_rollout_steps(self) -> int:
        """Sample the number of extra optimizer iterations to run before pushing to the buffer.

        Range is [rollout_min_steps, rollout_max_steps]; when rollout_grow > 0
        the upper bound grows linearly from min_steps up to max_steps over `rollout_grow`
        meta-steps.
        """
        cfg = self.train_cfg.ckpt_buffer_cfg
        min_steps = cfg.rollout_min_steps
        cfg_max_steps = cfg.rollout_max_steps

        if cfg.rollout_grow > 0:
            t_meta = self.global_step
            t_grow = cfg.rollout_grow
            max_steps = int(min_steps + (cfg_max_steps - min_steps) * min(1.0, t_meta / t_grow))
        else:
            max_steps = cfg_max_steps

        if min_steps == max_steps:
            return min_steps
        return int(np.random.randint(low=min_steps, high=max_steps + 1))

    def train_logging(self, batch, optimizer_output, gaussians, total_loss):
        self.log("loss/total", total_loss, prog_bar=True)  # also shown live in the tqdm bar
        if (
                self.global_rank == 0
                and (self.global_step % self.train_cfg.print_log_every_n_steps == 0 or total_loss > 5)
        ):
            print(
                f"train step {self.global_step}; "
                f"scene_name = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}; "
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # log gaussians scales
        if self.scene_trainer_cfg.num_update_steps > 0 and "deltas" in optimizer_output.info:
            delta_means = [deltas["means"] for deltas in optimizer_output.info["deltas"]]
            delta_scales = [deltas["scales"] for deltas in optimizer_output.info["deltas"]]

            for i in range(len(delta_means)):
                self.log(f"update{i}/delta_means_min", delta_means[i].abs().min().item())
                self.log(f"update{i}/delta_means_mean", delta_means[i].abs().mean().item())
                self.log(f"update{i}/delta_means_max", delta_means[i].abs().max().item())

            for i in range(len(delta_scales)):
                self.log(f"update{i}/delta_scales_min", delta_scales[i].abs().min().item())
                self.log(f"update{i}/delta_scales_mean", delta_scales[i].abs().mean().item())
                self.log(f"update{i}/delta_scales_max", delta_scales[i].abs().max().item())

        self.log("info/gaussian_scale_min", gaussians.scales.min().item())
        self.log("info/gaussian_scale_max", gaussians.scales.max().item())
        self.log("info/gaussian_scale_mean", gaussians.scales.mean().item())

        # log gaussians opacities
        self.log("info/gaussian_opacity_min", gaussians.opacities.min().item())
        self.log("info/gaussian_opacity_max", gaussians.opacities.max().item())
        self.log("info/gaussian_opacity_mean", gaussians.opacities.mean().item())

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")
        global slurm_id_logged
        if self.global_rank == 0 and not slurm_id_logged:
            print('slurm id:', os.environ.get('SLURM_JOB_ID'))
            slurm_id_logged = True

    def compute_losses(self, gaussians, i, num_output, render_output, curr_gt_rgb, valid_depth_mask,
                       error_idx=None, all_gt_rgb=None, tag="target",
                       curr_gt_depth=None, depth_mask=None, near=None, far=None):
        """Compute weighted sum of the view-dependent losses at one optimizer step.

        curr_gt_rgb [B, V_rendered] is pre-indexed to the views the optimizer rendered;
        all_gt_rgb [B, V_all] is the full GT passed only to losses that need every view (e.g. LossSh0).
        curr_gt_depth [B, V_rendered, H, W], its dataset-defined depth_mask, and near/far
        [B, V_rendered, 1, 1] are the GT depth for the rendered views, used by LossRenderDepth;
        None when the dataset has no depth.
        View-independent losses (e.g. deltas) are handled once per step in _calc_view_independent_losses.
        intermediate_loss_weight discounts earlier refinement steps.
        """
        # curr_gt_rgb: [B, V_rendered, C, H, W] — already narrowed to the views the optimizer rendered
        # all_gt_rgb:  [B, V_all, C, H, W]      — full GT; passed to losses that need all views (e.g. LossSh0)
        if all_gt_rgb is None:
            all_gt_rgb = curr_gt_rgb
        total_loss = 0
        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (num_output - 1 - i)

        gt_rgb, pred_rgb, valid_depth_mask = Loss.extract_pred_gt(
            curr_gt_rgb, render_output, error_idx, valid_depth_mask
        )

        for loss_fn in self.losses:
            # View-independent losses (e.g. deltas) are added once per step, not per tag.
            # A weight-0 loss is inert, so skip it rather than compute a term scaled to nothing.
            if not loss_fn.use_in_inner_steps or not loss_fn.view_dependent or loss_fn.is_disabled:
                continue
            loss = loss_fn(
                render_output,
                gaussians,
                self.global_step,
                gt_rgb=gt_rgb,
                pred_rgb=pred_rgb,
                gt_image=all_gt_rgb,
                valid_depth_mask=valid_depth_mask,
                gt_depth=curr_gt_depth,
                depth_mask=depth_mask,
                near=near,
                far=far,
            )

            loss_tag = f"{tag}_" + loss_fn.name
            loss_tag += f"_{i + 1}" if i > 0 else ""
            _check_finite_loss(loss, loss_tag)
            # Components go to wandb only; the training bar shows just loss/total (kept short).
            self.log(f"loss/{loss_tag}", loss, prog_bar=False)

            total_loss += curr_loss_weight * loss

        return total_loss

    def _calc_view_independent_losses(self, gaussians, i, num_output):
        """Sum the per-step losses that don't depend on the rendered views (e.g. deltas, which
        regularizes the optimizer's predicted updates). Added once per step — not once per
        target/context view set — so a view-independent quantity isn't double-counted in the total."""
        total_loss = 0
        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (num_output - 1 - i)
        for loss_fn in self.losses:
            if not loss_fn.use_in_inner_steps or loss_fn.view_dependent or loss_fn.is_disabled:
                continue
            loss = loss_fn(None, gaussians, self.global_step)
            loss_tag = loss_fn.name + (f"_{i + 1}" if i > 0 else "")
            _check_finite_loss(loss, loss_tag)
            self.log(f"loss/{loss_tag}", loss, prog_bar=False)  # wandb only; bar shows loss/total
            total_loss += curr_loss_weight * loss
        return total_loss

    @staticmethod
    def _curr_gt_depth(views, train_idx):
        """GT depth, its dataset-defined validity mask, and near/far for the views rendered this step
        (for LossRenderDepth), indexed to match the optimizer's rendered subset. depth_mask is cached
        next to depth by the dataset (None if it provides none); near/far bound the rendered depth.
        Returns all-None if the dataset has no depth."""
        gt_depth = views.get("depth")
        if gt_depth is None:
            return None, None, None, None
        depth_mask = views.get("depth_mask")
        near, far = views["near"], views["far"]
        if train_idx is not None:
            gt_depth = batched_select(gt_depth, train_idx)
            if depth_mask is not None:
                depth_mask = batched_select(depth_mask, train_idx)
            near = batched_select(near, train_idx)
            far = batched_select(far, train_idx)
        return gt_depth, depth_mask, near[..., None, None], far[..., None, None]

    def _calc_total_loss(self, batch, optimizer_output: OptimizerOutput | None, init_output):
        """Accumulate total training loss from the optimizer steps (RGB + Gaussians + render-depth).

        Every loss term is per-step and view-dependent, applied inside _calc_opt_loss to both the
        target and the context rendered views; it also logs the initialization metrics.
        """
        total_loss = 0
        valid_depth_mask = None  # It is always None, but possible, but depending on the dataset, it can be re-activated.
        t = optimizer_output.t if optimizer_output is not None else 0

        # Init loss (RGB + initializer-depth) and step-0 logging
        self._log_init_metrics(batch, init_output)

        # Log and calculate loss of intermediate outputs of the optimizer steps
        if self.scene_trainer_cfg.train_scene_opt:
            total_loss += self._calc_opt_loss(batch, optimizer_output, t, valid_depth_mask)

        return total_loss

    def _calc_opt_loss(self, batch, optimizer_output, t, valid_depth_mask):
        """Compute loss over all optimizer steps for both target and context views."""
        opt_loss = 0
        assert optimizer_output is not None
        step_num = len(optimizer_output.context_render_list) - 1  # first render is initialization

        # (tag, loss_enabled, loss_num)  — render/index lists accessed via optimizer_output methods
        view_loss_cfg = [
            ("target", self.train_cfg.loss_on_target_views, self.train_cfg.loss_on_target_views_num),
            ("context", self.train_cfg.loss_on_input_views, self.train_cfg.loss_on_input_views_num),
        ]

        # Compute loss of each optimizer step separately
        for i in range(step_num):
            gaussians_i = optimizer_output.gaussian_list[i + 1]
            # View-independent per-step losses (e.g. deltas): added once, not per target/context tag.
            opt_loss += self._calc_view_independent_losses(gaussians_i, i, step_num)
            for tag, loss_enabled, loss_num in view_loss_cfg:
                render_list = optimizer_output.get_render_list(tag)
                index_list = optimizer_output.get_index_list(tag)
                # all_gt_rgb: full GT for all views in the batch [B, V_all, C, H, W]
                all_gt_rgb = batch[tag]["image"]

                if index_list:
                    # index_list[0] is the init render index; steps start at [1].
                    # opt_batch_size < V_all: optimizer rendered a subset of views this step
                    train_idx = index_list[i + 1]  # [B, V_rendered] — from scene_trainer.opt_batch_size
                    curr_gt_rgb = batched_select(all_gt_rgb, train_idx)  # [B, V_rendered, C, H, W]
                else:
                    train_idx = None
                    curr_gt_rgb = all_gt_rgb
                self._log_train_metrics(i + 1, render_list[i + 1].color, curr_gt_rgb, tag=tag, t=t)

                if loss_enabled:
                    b, actual_v = curr_gt_rgb.shape[:2]
                    num_loss = actual_v if loss_num < 0 else loss_num
                    # error_idx: subsample rendered views down to loss_num for the loss computation
                    error_idx = torch.randperm(actual_v, device=curr_gt_rgb.device)[:num_loss]
                    error_idx = error_idx.unsqueeze(0).expand(b, -1)
                    # GT depth + its dataset validity mask for the same rendered views (LossRenderDepth);
                    # None if the dataset has no depth.
                    curr_gt_depth, curr_depth_mask, near, far = self._curr_gt_depth(batch[tag], train_idx)
                    opt_loss += self.compute_losses(gaussians_i, i, step_num,
                                                    render_list[i + 1], curr_gt_rgb, valid_depth_mask,
                                                    error_idx=error_idx, all_gt_rgb=all_gt_rgb, tag=tag,
                                                    curr_gt_depth=curr_gt_depth, depth_mask=curr_depth_mask,
                                                    near=near, far=far)

        # Compute a stability loss over all optimizer steps
        stability_loss_fn = next((l for l in self.losses if isinstance(l, LossStability)), None)
        if stability_loss_fn is not None and not stability_loss_fn.is_disabled:
            stability_loss = stability_loss_fn(optimizer_output, batch)
            _check_finite_loss(stability_loss, "stability")
            opt_loss += stability_loss
            self.log(f"loss/stability", stability_loss, prog_bar=False)  # wandb only; bar shows loss/total

        return opt_loss

    def _log_init_metrics(self, batch, init_output):
        """Log step-0 PSNR for the starting Gaussians, before any optimizer step runs.

        The starting Gaussians are either a fresh initialization or a state resumed from the ckpt
        buffer. PSNR is logged for whichever views were rendered into init_output: target only when
        training the initializer, both context and target when training the optimizer. When only a
        subset of views was rendered, its view index selects the matching GT views to compare against.
        """
        for tag in ("context", "target"):
            render = init_output.get_render(tag)
            if render is None:
                continue
            index = init_output.get_render_index(tag)
            all_gt_rgb = batch[tag]["image"]
            curr_gt_rgb = batched_select(all_gt_rgb, index) if index is not None else all_gt_rgb
            self._log_train_metrics(0, render.color, curr_gt_rgb, tag=tag)

    def _log_train_metrics(self, i, pred, gt, tag, t=-1):
        psnr = compute_psnr(
            rearrange(gt, "b v c h w -> (b v) c h w"),
            rearrange(pred, "b v c h w -> (b v) c h w"),
        )
        self.log(f"train/{tag}_psnr_{i}", psnr.mean().item())

    # ==================== Meta Optimizer Configuration ====================

    def _split_params(self, filter_key: str) -> tuple[list, list]:
        """Split parameters into (matched, rest) based on whether name contains filter_key."""
        matched, rest = [], []
        for name, param in self.named_parameters():
            (matched if filter_key in name else rest).append(param)
        return matched, rest

    def _build_adamw(self, params_or_groups, weight_decay: float, **kwargs):
        """Instantiate AdamW or AdamW8bit depending on config."""
        cls = AdamW8bit if self.meta_optimizer_cfg.adamw_8bit else optim.AdamW
        return cls(params_or_groups, weight_decay=weight_decay, **kwargs)

    def configure_optimizers(self):
        # This is the *meta* optimizer — it optimizes the parameters of the learned optimizer itself
        # (i.e. the KnnBasedOptimizer weights), not individual scene Gaussians.
        # Only called by Lightning during fit(); skipped entirely in test mode.
        cfg = self.meta_optimizer_cfg

        if cfg.lr_depth > 0:
            pretrained, rest = self._split_params("depth_predictor")
            param_groups = [{"params": pretrained, "lr": cfg.lr_depth}, {"params": rest, "lr": cfg.lr}]
            scheduler_lrs = [cfg.lr_monodepth, cfg.lr]
        elif cfg.lr_monodepth > 0:
            pretrained, rest = self._split_params("pretrained")
            param_groups = [{"params": pretrained, "lr": cfg.lr_monodepth}, {"params": rest, "lr": cfg.lr}]
            scheduler_lrs = [cfg.lr_monodepth, cfg.lr]
        else:
            param_groups = self.parameters()
            scheduler_lrs = cfg.lr

        meta_optimizer = self._build_adamw(param_groups, cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            meta_optimizer,
            scheduler_lrs,
            self.trainer.max_steps + 10,
            pct_start=cfg.warm_up_ratio,
            cycle_momentum=False,
            anneal_strategy="cos",
        )

        return {
            "optimizer": meta_optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ==================== Testing ====================

    @torch.no_grad()
    def test_step(self, scene_batch: BatchedExample, batch_idx: int):
        """
        This is a meta trainer class. Each training step and test step corresponds to training on one scene.
        We delegate the actual training/testing to meta_training_step and meta_test_step.
        The loop over inner training steps (training within a specific scene) is performed in
        self.optimize_scene.
        Each inner iteration is done by calling the forward call of the optimizer.
        """
        return self.meta_test_step(scene_batch, batch_idx)

    @torch.no_grad()
    def meta_test_step(self, scene_batch: BatchedExample, batch_idx: int):
        """Run the full test pipeline for one scene: initialize, optimize, post-process, then save+eval each phase."""
        if self._should_skip_scene(scene_batch):
            return

        output_path = self.test_cfg.output_path
        rule(f"Testing scene {batch_idx}: {scene_batch['scene'][0]}")

        # Prepare batch. Two distinct steps, kept separate on purpose: the data shim is the
        # universal patch-crop (applied on every path, training included), while
        # eval_preprocessing is eval-only depth-range/scale setup (training omits it).
        batch: BatchedExample = self.scene_trainer.apply_initializer_data_shim(scene_batch)
        self.scene_initializer.eval_preprocessing(batch, self.train_cfg)

        # Cameras
        if self.test_cfg.save_cameras_json:
            self.test_save_cameras_json(batch, output_path)
        if self.test_cfg.save_cameras_npz:
            self.test_save_cameras_npz(batch, output_path)

        # Init phase
        init_output = self._run_init_phase(batch, batch_idx, output_path)

        # Optimizer phase
        optimizer_output = None
        if self.scene_trainer.optimizer is not None:
            try:
                optimizer_output, scene_timing_metrics = self._optimize_and_measure_scene(
                    batch, init_output, allow_partial_on_skip=True,
                )
            # An out-of-memory error is recoverable: free the cache and skip just this scene so the
            # rest of the test set still runs. Any other RuntimeError -- a real bug, or a fatal CUDA
            # error like an illegal memory access that leaves the GPU context unusable -- is left to
            # propagate, since skipping the scene would only hide it.
            except torch.OutOfMemoryError as e:
                warn(f'ran out of memory before optimization started, skipping scene: {e}')
                torch.cuda.empty_cache()
                return
            except SkipBatchException as e:
                warn(f'skipping scene due to SkipBatch before optimization started: {e}')
                return

            # Init is already spliced into position 0 of optimizer_output lists by
            # SceneTrainer.optimize_scene (see _insert_init_into_output).
            self._eval_and_save(
                self.scene_trainer.optimizer,
                batch,
                batch_idx,
                optimizer_output,
                output_path,
                extra_scene_metrics=scene_timing_metrics,
            )

        # Post-processing phase
        self._run_postprocess_phase(batch, batch_idx, optimizer_output, init_output, output_path)

    def _should_skip_scene(self, scene_batch: BatchedExample) -> bool:
        """True if scenes_filter excludes this scene, or skip_if_outputs_exist and metric JSONs already exist."""
        scene_name = scene_batch['scene'][0]
        if self.test_cfg.scenes_filter is not None and scene_name not in self.test_cfg.scenes_filter:
            print(f"Scenes filter: {self.test_cfg.scenes_filter}")
            print(f"Skipping scene {scene_name} (not in scenes_filter)")
            return True

        output_path = self.test_cfg.output_path
        if output_path is None or not self.test_cfg.skip_if_outputs_exist:
            return False

        optimizer_name = (
            self.scene_trainer.optimizer.__class__.__name__.lower()
            if self.scene_trainer.optimizer is not None else "no_optimizer"
        )
        target_metric_path = output_path / optimizer_name / "metrics" / scene_name / f"target_{optimizer_name}.json"
        context_metric_path = output_path / optimizer_name / "metrics" / scene_name / f"context_{optimizer_name}.json"

        # Target views are always evaluated; context only when eval_context_views is set.
        skip_target = target_metric_path.exists()
        skip_context = context_metric_path.exists() or not self.test_cfg.eval_context_views

        if skip_target and skip_context:
            print(
                f"Metrics for scene {scene_name} already exist at {target_metric_path} and {context_metric_path}. Skipping..."
            )
            return True
        return False

    def _run_init_phase(self, batch: BatchedExample, batch_idx: int, output_path: Path) -> InitializerOutput:
        """Run the initializer (with full-V context + target rendering) and optionally eval+save it."""
        init_output: InitializerOutput = self.init_gaussians_and_render(
            batch,
            visualization_dump={},
            render_context=True,
            render_target=True,
            grad_enabled=False,
            depth_mode=self._depth_mode,
        )
        if self.test_cfg.eval_initialization:
            print("\nEvaluating initialization...")
            self._eval_and_save(self.scene_initializer, batch, batch_idx, init_output, output_path)
        return init_output

    @staticmethod
    def _zero_first_iter_warmup(timer: Benchmarker) -> None:
        """Zero the first optimization iteration's timing in `timer` (one-time GPU warm-up:
        kernel JIT, cuDNN autotune, CUDA context). Zeros both the total ("iter") and render
        ("decoder") entries so the derived update time (iter - decoder) stays consistent."""
        times = timer.execution_times  # flushes pending CUDA events
        for tag in ("iter", "decoder"):
            if times.get(tag):
                times[tag][0] = 0.0

    def _optimize_and_measure_scene(
            self,
            batch: BatchedExample,
            init_output: InitializerOutput,
            *,
            allow_partial_on_skip: bool = False,
    ) -> tuple[OptimizerOutput, dict]:
        """Run the optimizer for one scene; record per-scene CUDA timings + peak VRAM.

        Shared by the test path and the train-eval pipeline. `allow_partial_on_skip` keeps and
        scores the steps a scene completed before failing mid-optimization; it defaults off, which
        is the train-eval behavior (drop a failed scene whole). Saving renders/metrics to disk is
        done afterward by the caller (`_eval_and_save`), not here.

        Raises torch.OutOfMemoryError/RuntimeError on OOM and SkipBatchException on a
        deliberate skip — the caller catches both and drops the scene.
        Returns (optimizer_output, scene_timing_metrics) where the dict is keyed by
        peak_vram_mb / decoder_ms / optimizer_ms / optimizer_net_ms / scene_start_ms and
        will be written into target_*.json / context_*.json by `_eval_and_save`.
        optimizer_net = on_scene_start + all iteration steps; excludes save-every renders
        (which happen after iter_end.record() and are therefore not in iter_time_log).
        The first optimized scene of the pass has its first iteration zeroed as GPU warm-up.
        """
        scene_name = batch["scene"][0]

        torch.cuda.reset_peak_memory_stats()
        optimizer_output = self.optimize_scene(
            batch,
            init_output,
            allow_partial_on_skip=allow_partial_on_skip,
            depth_mode=self._depth_mode,
        )

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.benchmarker.record("peak_vram_mb", peak_vram_mb)

        scene_timing_metrics = self._record_optimizer_scene_timing()
        scene_timing_metrics["peak_vram_mb"] = peak_vram_mb
        print(
            f"[timing] scene={scene_name} "
            f"scene_start={scene_timing_metrics['scene_start_ms']:.0f}ms "
            f"decoder={scene_timing_metrics['decoder_ms']:.0f}ms "
            f"optimizer={scene_timing_metrics['optimizer_ms']:.0f}ms "
            f"optimizer_net={scene_timing_metrics['optimizer_net_ms']:.0f}ms "
            f"peak_vram={peak_vram_mb:.0f}MB"
        )
        return optimizer_output, scene_timing_metrics

    def _record_optimizer_scene_timing(self) -> dict:
        """Read the just-finished scene's timings from the optimizer's benchmarker, record one
        per-scene value for each into the shared benchmarker, and return them (ms).

        decoder = time rendering during the update steps; optimizer = the update steps themselves;
        optimizer_net = scene setup + decoder + optimizer. The first optimized scene of the pass has
        its first update step zeroed as a one-time GPU warm-up (kernel compilation, autotuning); the
        per-step time curve reads the same values, so it stays consistent.
        """
        opt = self.scene_trainer.optimizer
        if not self._timing_warmup_done:
            self._zero_first_iter_warmup(opt.benchmarker)
            self._timing_warmup_done = True
        decoder_ms = sum(opt.decoder_time_log)
        optimizer_ms = sum(opt.optimizer_time_log)
        optimizer_net_ms = opt.scene_start_ms + decoder_ms + optimizer_ms
        self.benchmarker.record("decoder", decoder_ms)
        self.benchmarker.record("optimizer", optimizer_ms)
        self.benchmarker.record("optimizer_net", optimizer_net_ms)
        return {
            "decoder_ms": decoder_ms,
            "optimizer_ms": optimizer_ms,
            "optimizer_net_ms": optimizer_net_ms,
            "scene_start_ms": opt.scene_start_ms,
        }

    def _run_postprocess_phase(
            self,
            batch: BatchedExample,
            batch_idx: int,
            optimizer_output: OptimizerOutput | None,
            init_output: InitializerOutput,
            output_path: Path,
    ) -> None:
        """Run optional post-processing on the final Gaussians and eval+save it."""
        gaussians = optimizer_output.gaussian_list[-1] if optimizer_output is not None else init_output.gaussians
        postprocessed_output = self.test_postprocess_gaussians(batch, gaussians=gaussians, visualization_dump={})
        if postprocessed_output is not None:
            self._eval_and_save(self.scene_trainer.postprocess, batch, batch_idx, postprocessed_output, output_path)

    def _run_eval_pipeline(
            self,
            batch: BatchedExample,
    ) -> tuple[BatchedExample, InitializerOutput, DecoderOutput, OptimizerOutput | None]:
        """Run the shared single-scene eval pipeline: data-shim + preprocessing, initialize,
        render the init target view, then optionally optimize.

        Returns (batch, init_output, final_target_render, optimizer_output). `batch` is the
        shimmed batch (callers must rebind). The init target render is attached to `init_output`,
        so when an optimizer runs, optimize_scene places it first in
        `optimizer_output.target_render_list` (position 0), followed by the per-step renders --
        the same layout the train and test paths use. When no optimizer runs (num_update_steps == 0,
        which is exactly when the optimizer is None) the init render is read back from
        `init_output.target_render`. `final_target_render` is the
        init render when no optimizer ran, else `optimizer_output.target_render_list[-1]`.
        The initializer and init-render decode are always timed into the shared `self.benchmarker`
        (under "initializer"/"init_decoder"); the per-scene optimizer timings are read separately
        from the optimizer's own logs by the caller. Raises SkipBatchException if the initializer or
        optimizer signals a skip.
        """
        batch = self.scene_trainer.apply_initializer_data_shim(batch)
        self.scene_initializer.eval_preprocessing(batch, self.train_cfg)
        assert batch["target"]["image"].shape[0] == 1

        depth_mode = self._depth_mode
        # Initialize and render the target view. Kept on the GPU (to_cpu=False) since it feeds the
        # optimizer below and is placed first in target_render_list.
        init_output = self.init_gaussians_and_render(
            batch,
            visualization_dump={},
            render_context=False,
            render_target=True,
            grad_enabled=False,
            depth_mode=depth_mode,
            to_cpu=False,
        )

        optimizer_output = None
        final_target_render = init_output.get_render("target")
        if self.scene_optimizer is not None:
            # Refine the Gaussians over the optimization steps; each step renders the target views,
            # so the final renders are the last entry of target_render_list. Same measured optimize
            # step as the test path (records per-scene timing + peak VRAM into the benchmarker).
            optimizer_output, _ = self._optimize_and_measure_scene(batch, init_output)
            final_target_render = optimizer_output.target_render_list[-1]

        return batch, init_output, final_target_render, optimizer_output

    @torch.no_grad()
    @rank_zero_only
    def validation_step(self, scene_batch: BatchedExample, batch_idx: int):
        if self.global_rank == 0:
            print(
                f"Validation step {self.global_step}; "
                f"scene_name = {[a[:20] for a in scene_batch['scene']]}; "
                f"context = {scene_batch['context']['index'].tolist()}; "
                f"target = {scene_batch['target']['index'].tolist()}"
            )

        # Running evaluation
        try:
            scene_batch, initializer_output, last_output, optimizer_output = self._run_eval_pipeline(scene_batch)
        except SkipBatchException as e:
            scene_short = scene_batch['scene'][0][:20]
            warn(f"Skipping validation for scene {scene_short}: {e}")
            return


        # RGB metrics.
        last_rgb = last_output.color[0]
        last_rgb = last_rgb.to(scene_batch["target"]["image"].device)
        rgb_gt = scene_batch["target"]["image"][0]
        scores = self._score_renders([last_rgb], rgb_gt, metrics=["psnr", "ssim"])[0]
        self.log("val/psnr", scores["psnr"])
        self.log("val/ssim", scores["ssim"])

        # Depth
        self._log_validation_depth_viz(scene_batch, initializer_output, last_output)


        # Summary image — render 2 context and 2 target views at init and final step
        init_gaussians = initializer_output.gaussians  # already on GPU (to_cpu=False)
        final_gaussians = (
            optimizer_output.last_prev_output.gaussians  # final-step Gaussians, on GPU
            if optimizer_output is not None
            else init_gaussians
        )
        init_ctx, final_ctx = self._render_init_final_ctx(init_gaussians, final_gaussians, scene_batch)

        comparison = self._build_comparison_image(
            ctx_gt=scene_batch["context"]["image"][0][:2],
            init_ctx=init_ctx,
            final_ctx=final_ctx,
            tgt_gt=scene_batch["target"]["image"][0][:2],
            init_tgt=initializer_output.get_render("target"),
            final_tgt=last_output,
        )

        self.logger.log_image(
            "Media/eval_vis",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=scene_batch["scene"],
        )

        # Run video validation step.
        if not self.train_cfg.no_viz_video:
            self.render_video_interpolation(scene_batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(scene_batch)

    def _log_validation_depth_viz(
            self,
            scene_batch: BatchedExample,
            initializer_output: InitializerOutput,
            last_output: DecoderOutput,
    ) -> None:
        """Log the standalone validation depth images: the initializer's predicted context
        depth ("depth", when viz_depth_separate) and the final target-render depth
        ("render_depth", when eval_render_depth/viz_render_depth)."""
        # viz depth
        if initializer_output.depths is not None and self.train_cfg.viz_depth_separate:
            # only visualize predicted depth
            pred_depths = initializer_output.depths[0]  # [V, H, W]

            # gaussian downsample
            # downsample image to depth resolution
            if pred_depths.shape[1:] != scene_batch["context"]["image"].shape[-2:]:
                input_images = F.interpolate(
                    scene_batch["context"]["image"][0],
                    size=pred_depths.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            else:
                input_images = scene_batch["context"]["image"][0]  # [N, 3, H, W]

            concat = self._make_depth_viz(1.0 / pred_depths, input_images)

            # reshape when the number of input images is too large
            # otherwise the image will be too wide
            num_inputs = input_images.shape[0]
            width = input_images.shape[-1]
            if num_inputs > 8:
                rows = 4
                assert num_inputs % rows == 0
                stride = num_inputs // rows
                out = []
                for i in range(rows):
                    out.append(concat[:, :, width * stride * i: width * stride * (i + 1)])

                concat = torch.cat(out, dim=1)  # [3, H*2*R, W*N/R]

                # resize to half resolution to save space
                concat = F.interpolate(concat.unsqueeze(0), scale_factor=0.5, mode='bilinear',
                                       align_corners=True).squeeze(0)

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=scene_batch["scene"],
            )

        # viz rendered depth
        if self.train_cfg.eval_render_depth or self.train_cfg.viz_render_depth:
            render_depth = last_output.depth[0]  # [V, H, W]
            input_images = scene_batch["target"]["image"][0]  # [N, 3, H, W]
            concat = self._make_depth_viz(1.0 / render_depth.clamp(min=0.01, max=1000.), input_images)

            self.logger.log_image(
                "render_depth",
                [concat],
                step=self.global_step,
                caption=scene_batch["scene"],
            )

    def _log_train_comparison(self, scene_batch, init_output, last_gaussians) -> None:
        """Log context+target init/final comparison image during training.
        Fires once per validation round (same cadence as eval_vis)."""
        if not (self.global_rank == 0 and self._run_train_comparison):
            return
        self._run_train_comparison = False
        init_gaussians = init_output.gaussians
        with torch.no_grad():
            init_ctx, final_ctx = self._render_init_final_ctx(init_gaussians, last_gaussians, scene_batch)
            final_tgt = self.scene_trainer.decoder.forward_batch_subset(
                last_gaussians, scene_batch["target"], start=0, end=2, depth_mode=self._depth_mode,
            )
        comparison = self._build_comparison_image(
            ctx_gt=scene_batch["context"]["image"][0][:2],
            init_ctx=init_ctx,
            final_ctx=final_ctx,
            tgt_gt=scene_batch["target"]["image"][0][:2],
            init_tgt=init_output.get_render("target"),
            final_tgt=final_tgt,
        )
        self.logger.log_image(
            "Media/train_vis",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=scene_batch["scene"],
        )

    def _render_init_final_ctx(self, init_gaussians, final_gaussians, scene_batch):
        """Render 2 context views at init and final Gaussians. no_grad is the caller's responsibility."""
        init_ctx = self.scene_trainer.decoder.forward_batch_subset(
            init_gaussians, scene_batch["context"], start=0, end=2, depth_mode=self._depth_mode,
        )
        final_ctx = (
            init_ctx if final_gaussians is init_gaussians else
            self.scene_trainer.decoder.forward_batch_subset(
                final_gaussians, scene_batch["context"], start=0, end=2, depth_mode=self._depth_mode,
            )
        )
        return init_ctx, final_ctx

    @staticmethod
    def _inv_depth_col(depth: Tensor, ref_hw: tuple) -> list:
        """Colorize [V,H,W] depth into a list of [3,H,W] inverse-depth frames matching ref_hw."""
        device = depth.device
        inv = 1.0 / depth.clamp(min=0.01, max=1000.)
        concat = rearrange(inv, "v h w -> (v h) w").detach().cpu()
        viz = viz_depth_tensor(concat, as_uint8=False).to(device)
        viz = rearrange(viz, "c (v h) w -> v c h w", v=depth.shape[0])
        if viz.shape[-2:] != ref_hw:
            viz = F.interpolate(viz, size=ref_hw, mode='bilinear', align_corners=True)
        return list(viz)

    def _build_comparison_image(
            self,
            ctx_gt: Tensor,
            init_ctx: DecoderOutput,
            final_ctx: DecoderOutput,
            tgt_gt: Tensor,
            init_tgt: DecoderOutput,
            final_tgt: DecoderOutput,
    ) -> Tensor:
        """Build side-by-side comparison: context and target views at init and final step."""
        device = ctx_gt.device
        ref_hw = tuple(ctx_gt.shape[-2:])

        def rgb_col(images: Tensor, label: str) -> Tensor:
            return add_label(vcat(*images.to(device)[:2]), label)

        def depth_col(dec: DecoderOutput, label: str) -> Tensor | None:
            if dec.depth is None:
                return None
            return add_label(vcat(*self._inv_depth_col(dec.depth[0].to(device)[:2], ref_hw)), label)

        cols = [
            rgb_col(ctx_gt, "GT Ctx"),
            rgb_col(init_ctx.color[0], "Init Ctx"),
            depth_col(init_ctx, "Init Ctx Depth"),
            rgb_col(final_ctx.color[0], "Final Ctx"),
            depth_col(final_ctx, "Final Ctx Depth"),
            rgb_col(tgt_gt, "GT Tgt"),
            rgb_col(init_tgt.color[0], "Init Tgt"),
            depth_col(init_tgt, "Init Tgt Depth"),
            rgb_col(final_tgt.color[0], "Final Tgt"),
            depth_col(final_tgt, "Final Tgt Depth"),
        ]
        return hcat(*[c for c in cols if c is not None])

    @staticmethod
    def _make_depth_viz(inverse_depth: Tensor, images: Tensor) -> Tensor:
        """Combine inverse-depth colormap with RGB images for logging. Returns [3, H*2, W*V]."""
        depth_viz = viz_depth_tensor(torch.cat(list(inverse_depth), dim=1).cpu().detach())  # [3, H, W*V]
        concat_img = torch.cat(list(images), dim=-1).cpu().detach() * 255  # [3, H, W*V]
        return torch.cat((concat_img, depth_viz), dim=1)  # [3, H*2, W*V]

    def on_fit_start(self):
        run = self.logger.experiment
        if run is not None:
            run.define_metric("inner_iteration")
            run.define_metric("test/psnr/*", step_metric="inner_iteration")

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        """Run evaluation on the full test set during training (rank-zero only).

        Iterates the test dataloader, accumulates per-iteration RGB metrics (psnr/ssim/lpips)
        and optional depth metrics (init-depth + render-depth) into scores_dict, then logs:
          (1) a per-meta-step inner-iteration PSNR series for the wandb dashboard,
          (2) the averaged metrics under test/<metric>,
          (3) the averaged initializer/decoder/optimizer runtime in ms (the first optimized
              scene's first update step is excluded as warm-up), and
          (4) a per-inner-step PSNR summary line + total wall-clock.
        """
        print(
            f"Validation step at global step {self.global_step}. "
            f"Running evaluation on {self.train_cfg.eval_data_length} test sets..."
        )
        start_t = time.time()

        full_testsets = self.trainer.datamodule.test_dataloader(dataset_cfg=self.eval_data_cfg)
        scores_dict = defaultdict(list)
        self.benchmarker.clear_history()
        self._timing_warmup_done = False

        for batch_idx, batch in tqdm(
                enumerate(full_testsets),
                total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break
            self._eval_one_test_batch(batch, batch_idx, scores_dict)

        self._log_inner_iteration_table(scores_dict)
        self._summarize_test_scores(scores_dict, start_t)

    def _eval_one_test_batch(
            self,
            batch,
            batch_idx: int,
            scores_dict: dict,
    ) -> None:
        """Evaluate one test-set batch and append metrics into `scores_dict` in-place.

        Skips the batch on SkipBatchException from initialization or optimization.
        """
        batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
        batch = self.on_after_batch_transfer(batch, dataloader_idx=batch_idx)

        try:
            batch, init_output, final_render, optimizer_output = self._run_eval_pipeline(batch)
        except SkipBatchException as e:
            warn(f'Skipping batch due to SkipBatch: {e}')
            return

        # RGBs
        # target_render_list holds the init render first (position 0), then one render per
        # optimizer step. When no optimizer runs (num_update_steps == 0) there is no list, so
        # score the init render alone at step 0.
        if optimizer_output is not None:
            rgbs = [render.color[0] for render in optimizer_output.target_render_list]
            steps = self.scene_optimizer.save_every.get_iterations(len(rgbs))
        else:
            rgbs = [init_output.target_render.color[0]]
            steps = [0]

        rgb_gt = batch["target"]["image"][0]
        self._accumulate_rgb_metrics(rgbs, rgb_gt, steps, scores_dict)
        
        # Depths
        # Two distinct depth metrics: init-depth scores the initializer's context-view
        # prediction (pred_depths); render-depth scores the final target render's depth
        # (init's target render, or the optimizer's last render).
        pred_depths = init_output.depths
        depth_gt = batch["context"].get("depth")
        if pred_depths is not None and depth_gt is not None and depth_gt.max() > 0:
            self._accumulate_init_depth_metrics(pred_depths, depth_gt, batch, scores_dict)
        target_depth_gt = batch["target"].get("depth")
        if self.train_cfg.eval_render_depth and target_depth_gt is not None and target_depth_gt.max() > 0:
            self._accumulate_render_depth_metrics(final_render, batch, scores_dict)

    @staticmethod
    def _score_renders(
            renders: list[Tensor],
            rgb_gt: Tensor,
            metrics: list[str],
            iter_batch_size: int = -1,
    ) -> list[dict[str, float]]:
        """Compute RGB metrics for each render in `renders` against `rgb_gt`.

        Returns a list parallel to `renders`; each entry maps metric name to a Python float.
        `lpips` is unpacked into separate `alex_lpips`/`vgg_lpips` entries so callers can
        treat all metrics uniformly.
        """
        per_render: list[dict[str, float]] = []
        for rgb in renders:
            raw = compute_rgb_metrics(rgb, rgb_gt, metrics=metrics, iter_batch_size=iter_batch_size)
            flat: dict[str, float] = {}
            for name, score in raw.items():
                if name == "lpips":
                    alex, vgg = score
                    flat["alex_lpips"] = alex.item()
                    flat["vgg_lpips"] = vgg.item()
                else:
                    flat[name] = score.item()
            per_render.append(flat)
        return per_render

    def _accumulate_rgb_metrics(self, rgbs: list, rgb_gt, steps: list[int], scores_dict: dict) -> None:
        """For each refinement checkpoint k, compute psnr/ssim/lpips(alex,vgg) and append into
        scores_dict[f"{metric}_{steps[k]}"]. Also mirror the last step's value under the
        unsuffixed `metric` key for between-run comparison."""
        per_render = self._score_renders(rgbs, rgb_gt, metrics=["psnr", "ssim", "lpips"])
        for i, scores in enumerate(per_render):
            is_last = (i == len(per_render) - 1)
            for name, val in scores.items():
                scores_dict[f"{name}_{steps[i]}"].append(val)
                if is_last:
                    scores_dict[name].append(val)

    @staticmethod
    def _compute_init_depth_metrics(pred_depths, depth_gt, batch) -> dict[str, float]:
        """abs_rel / rmse / a1 between the initializer's context-view depth prediction and GT.
        Depth is upsampled to context-image resolution to match GT if the initializer downsampled."""
        pred_depths = pred_depths[0]  # [V, H, W]
        if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
            pred_depths = F.interpolate(
                pred_depths.unsqueeze(1),
                size=batch["context"]["image"].shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)
        depth_gt = depth_gt[0]  # [V, H, W]
        near = batch["context"]["near"][..., None, None][0]  # [V, 1, 1]
        far = batch["context"]["far"][..., None, None][0]
        valid = (depth_gt >= near) & (depth_gt <= far)

        all_metrics = compute_depth_errors(
            depth_gt[valid].detach().cpu().numpy(),
            pred_depths[valid].detach().cpu().numpy(),
        )
        return {"abs_rel": all_metrics[0], "rmse": all_metrics[2], "a1": all_metrics[4]}

    def _accumulate_init_depth_metrics(self, pred_depths, depth_gt, batch, scores_dict: dict) -> None:
        """Append init-depth metrics (abs_rel/rmse/a1) for one scene into scores_dict."""
        for name, val in self._compute_init_depth_metrics(pred_depths, depth_gt, batch).items():
            scores_dict[name].append(val)

    @staticmethod
    def _accumulate_render_depth_metrics(target_render, batch, scores_dict: dict) -> None:
        """render_abs_rel / render_rmse / render_a1 between the final target-render depth and GT.

        The render depth comes from whichever module produced the final target render: the
        initializer's target render, or the optimizer's last step if an optimizer ran.
        
        (`pred_depths` is the initializer's context-view prediction)
        """
        render_depth = target_render.depth[0]  # [V, H, W]
        depth_gt = batch["target"]["depth"][0]  # [V, H, W]
        near = batch["target"]["near"][..., None, None][0]  # [V, 1, 1]
        far = batch["target"]["far"][..., None, None][0]
        valid = (depth_gt >= near) & (depth_gt <= far)

        all_metrics = compute_depth_errors(
            depth_gt[valid].detach().cpu().numpy(),
            render_depth[valid].detach().cpu().numpy(),
        )
        scores_dict["render_abs_rel"].append(all_metrics[0])
        scores_dict["render_rmse"].append(all_metrics[2])
        scores_dict["render_a1"].append(all_metrics[4])

    def _log_inner_iteration_table(self, scores_dict: dict) -> None:
        """Log a per-meta-step PSNR-vs-inner-iteration series under test/psnr/meta_<global_step>,
        so wandb can render the optimization trajectory for the current meta step.

        Picks the step-suffixed psnr_<k> entries out of scores_dict and logs (inner_step, mean psnr).
        """
        if not (hasattr(self.logger, 'experiment') and self.logger.experiment is not None):
            return

        # Step-suffixed metrics look like "psnr_0", "psnr_1", ...; collect (inner_step, mean psnr).
        series = []
        for score_tag, scores in scores_dict.items():
            if '_' not in score_tag or not score_tag.split('_')[-1].isdigit():
                continue
            metric_name, step_str = score_tag.rsplit('_', 1)
            if metric_name != "psnr":
                continue
            if scores:
                series.append((int(step_str), sum(scores) / len(scores)))

        if len(series) <= 1:
            return

        try:
            run = self.logger.experiment
            run.define_metric("inner_iteration")
            run.define_metric(f"test/psnr/meta_{self.global_step}", step_metric="inner_iteration")
            for inner_step, value in series:
                run.log({
                    "inner_iteration": inner_step,
                    f"test/psnr/meta_{self.global_step}": value,
                })
        except Exception as e:
            warn(f"Could not create automatic charts: {e}")

    def _summarize_test_scores(self, scores_dict: dict, start_t: float) -> None:
        """Average each metric across batches, log under test/<metric>, log per-tag avg runtimes,
        and print the per-step PSNR summary + total wall-clock."""
        for score_tag, cur_scores in scores_dict.items():
            if len(cur_scores) > 0:
                self.log(f"test/{score_tag}", sum(cur_scores) / len(cur_scores))

        for tag, times in self.benchmarker.execution_times.items():
            if len(times) == 0:
                continue
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times):.1f} ms per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        # Keep the step index i tied to its psnr_<i> label; skip steps that recorded no scores.
        psnr_str = ", ".join(
            f"psnr_{i}: {sum(scores) / len(scores):.3f}"
            for i in range(self.scene_trainer_cfg.num_update_steps + 1)
            if (scores := scores_dict[f"psnr_{i}"])
        )
        example_num = len(scores_dict['psnr_0'])
        print(f"Eval total time cost: {overall_eval_time:.3f}s, {psnr_str}, example_num: {example_num} ")
        self.log("test/runtime_all", overall_eval_time)

    @staticmethod
    def _get_renders_list(
            output: OptimizerOutput | InitializerOutput,
            input_str: str,
            module: Initializer | Optimizer | PostProcessing3DGS,
    ) -> tuple[list, list[int]]:
        """Get render list and corresponding iteration indices for a given view tag ('context' or 'target')."""
        if isinstance(output, OptimizerOutput):
            renders_list = output.get_render_list(input_str)
        elif isinstance(output, InitializerOutput):
            if input_str == "context":
                assert output.context_render is not None, "InitializerOutput must contain context_render"
                renders_list = [output.context_render]
            elif input_str == "target":
                assert output.target_render is not None, "InitializerOutput must contain target_render"
                renders_list = [output.target_render]
            else:
                raise ValueError(f"Unknown input_str: {input_str}")
        else:
            raise ValueError(f"Unknown output type: {type(output)}")
        iterations = [0] if isinstance(module, Initializer) else module.save_every.get_iterations(len(renders_list))
        return renders_list, iterations

    @staticmethod
    def _compute_depth_range(
            output: OptimizerOutput | InitializerOutput,
            input_strs: list[str],
            module: Initializer | Optimizer | PostProcessing3DGS,
    ) -> tuple[float, float]:
        """Scan all renders to find the global depth min/max for consistent visualization."""
        depth_vmin, depth_vmax = np.inf, -np.inf
        have_depths = True
        for input_str in input_strs:
            renders_list, _ = MetaTrainer._get_renders_list(output, input_str, module)
            assert renders_list is not None, f"No renders found for {input_str}"
            for iter_renders in renders_list:
                iter_depths = iter_renders.depth  # (1, V, H, W)
                if iter_depths is None:
                    have_depths = False
                    continue
                iter_depths = iter_depths[0]  # (V, H, W)
                depth_vmin = min(depth_vmin, iter_depths.min().item())
                depth_vmax = max(depth_vmax, iter_depths.max().item())
        if not have_depths:
            depth_vmin, depth_vmax = 0.0, 1.0
        return depth_vmin, depth_vmax

    def _compute_error_vmax(
            self,
            output: OptimizerOutput | InitializerOutput,
            input_strs: list[str],
            module: Initializer | Optimizer | PostProcessing3DGS,
            batch: BatchedExample,
    ) -> float:
        # Per-scene error range for error-map visualization. The 99th percentile
        # (rather than the raw max) keeps a single outlier pixel from flattening
        # the magma color scale across the whole scene.
        if not self.test_cfg.save_error_image:
            return 1.0
        error_values = []
        for input_str in input_strs:
            renders_list, _ = self._get_renders_list(output, input_str, module)
            if renders_list is None:
                continue
            rgb_gt = batch[input_str]["image"][0]  # (V, 3, H, W)
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                err = (iter_rgbs - rgb_gt.to(iter_rgbs)).abs().mean(1)  # (V, H, W)
                error_values.append(err.flatten().cpu().numpy())
                del iter_rgbs, err
        if not error_values:
            return 1.0
        return float(np.percentile(np.concatenate(error_values), 99))

    def _compute_and_save_scores(
            self,
            module: Initializer | Optimizer | PostProcessing3DGS,
            output: OptimizerOutput | InitializerOutput,
            renders_list: list,
            rgb_gt: Tensor,
            iterations: list[int],
            input_str: str,
            module_name: str,
            out_dir: Path,
            extra_scene_metrics: dict | None,
    ) -> None:
        """Compute RGB metrics per iteration, accumulate into output_dict, and save per-scene JSON."""
        # Collect per-step stats logs from the module
        if isinstance(module, Initializer):
            nr_gaussians_log = [output.gaussians.means.shape[1]]
            iter_time_log = [0.0]
            nr_nonzero_grads_log = [0.0]
        elif isinstance(module, (Optimizer, PostProcessing3DGS)):
            nr_gaussians_log = module.nr_gaussians_log
            nr_nonzero_grads_log = module.nr_nonzero_grad_log
            # The one-time GPU warm-up is already zeroed at its source (the first optimized
            # scene's first iteration, in _optimize_and_measure_scene), so the timer is read as-is here.
            iter_time_log = module.iter_time_log
        else:
            raise ValueError(f"Unknown module type: {type(module)}")

        self.init_output_dict_for_new_scene(input_str=input_str, tag=module_name)
        output_dict = self.test_step_outputs_context if input_str == "context" else self.test_step_outputs_target
        out_dir.mkdir(parents=True, exist_ok=True)

        renders = [renders_list[i].color[0] for i in range(len(iterations))]  # each (V, 3, H, W)
        per_render = self._score_renders(
            renders, rgb_gt,
            metrics=self.test_cfg.compute_scores_metrics,
            iter_batch_size=self.test_cfg.metrics_batch_size,
        )

        # Per-step logs hold one entry per optimizer step (0-indexed), recorded at the state
        # that step starts from; a render at `nr_iter` steps is exactly that state, so its
        # gaussians / nonzero_grads are log[nr_iter] (the init, nr_iter=0, reads the gradient
        # at the init state). The final render (state N) has no further logged step, so clamp.
        # Cumulative wall-clock to reach `nr_iter` steps is cum_time[nr_iter] (init -> 0).
        cum_time = np.concatenate([[0.0], np.cumsum(iter_time_log)]) if iter_time_log else None
        for i, nr_iter in tqdm(enumerate(iterations), desc=f"Evaluating {input_str}", total=len(iterations)):
            scores = dict(per_render[i])
            if nr_gaussians_log:
                scores["gaussians"] = nr_gaussians_log[min(nr_iter, len(nr_gaussians_log) - 1)]
            if nr_nonzero_grads_log:
                scores["nonzero_grads"] = nr_nonzero_grads_log[min(nr_iter, len(nr_nonzero_grads_log) - 1)]
            if cum_time is not None:
                scores["time"] = float(cum_time[nr_iter])

            for name, val in scores.items():
                output_dict[f"{module_name}_{name}"][-1].append(val)
            output_dict[f"{module_name}_iterations"][-1].append(nr_iter)

        # Save per-scene metrics to JSON
        last_scene_metrics = {key: vals[-1] for key, vals in output_dict.items()}
        if extra_scene_metrics:
            last_scene_metrics.update(extra_scene_metrics)
        metrics_save_path = out_dir / f"{input_str}_{module_name}.json"
        with metrics_save_path.open("w") as f:
            print(f"Saving metrics to {metrics_save_path}")
            json.dump(last_scene_metrics, f, indent=4)

    @torch.no_grad()
    def _eval_and_save(
            self,
            module: Initializer | Optimizer | PostProcessing3DGS,
            batch: BatchedExample,
            batch_idx: int,
            output: OptimizerOutput | InitializerOutput,
            output_path: Path,
            extra_scene_metrics: dict | None = None,
    ) -> None:
        """Evaluate and save renders/depths/scores for one module's output (init / optimizer / postprocess)."""
        module_name = module.__class__.__name__.lower()

        output_path = CustomPath(output_path / module_name)
        output_path.mkdir(parents=True, exist_ok=True)

        target_shape = batch["target"]["image"].shape  # [B, V, 3, H, W]
        context_shape = batch["context"]["image"].shape  # [B, V, 3, H, W]
        assert target_shape[-3:] == context_shape[-3:], f"{target_shape}, {context_shape}"
        b, v, _, h, w = target_shape
        assert b == 1, "Evaluation only supports scene batch size 1."
        scene_name = batch["scene"][0]

        # Save poses
        if self.test_cfg.save_poses:
            poses_data = {
                "context": {"shape": context_shape[-2:]},
                "target": {"shape": target_shape[-2:]},
            }
            for key in ["extrinsics", "intrinsics", "near", "far"]:
                poses_data["context"][key] = batch["context"][key][0].cpu().numpy().tolist()
                poses_data["target"][key] = batch["target"][key][0].cpu().numpy().tolist()
            save_path = output_path / 'poses' / f"{scene_name}_poses.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving poses to {save_path.parent}")
            with open(save_path, 'w') as f:
                json.dump(poses_data, f, indent=4)

        # Save gaussians
        if self.test_cfg.save_gaussian:
            if isinstance(output, InitializerOutput):
                save_path = output_path / 'gaussians' / scene_name / 'init.ply'
                save_gaussian_ply(output.gaussians, save_path)
            elif isinstance(output, OptimizerOutput):
                iterations = module.save_every.get_iterations(len(output.gaussian_list))
                gaussian_list = output.gaussian_list
                if self.test_cfg.save_gaussian_last_only:
                    # keep the scene-start (init) gaussians and the final step
                    keep = [0, -1] if len(gaussian_list) > 1 else [0]
                    iterations = [iterations[i] for i in keep]
                    gaussian_list = [gaussian_list[i] for i in keep]
                for step, iter_gaussians in zip(iterations, gaussian_list):
                    save_path = output_path / 'gaussians' / scene_name / f'step{step}.ply'
                    save_gaussian_ply(iter_gaussians, save_path)
            else:
                raise ValueError(f"Unknown output type: {type(output)}")

        # Score the initializer's predicted context-view depth against the GT depth (initialization
        # quality, one value per scene).
        if isinstance(module, Initializer) and self.test_cfg.compute_scores:
            pred_depths = output.depths
            depth_gt = batch["context"].get("depth")
            if pred_depths is not None and depth_gt is not None and depth_gt.max() > 0:
                for name, val in self._compute_init_depth_metrics(pred_depths, depth_gt, batch).items():
                    # one row per scene (length-1, like the init's other per-scene metrics)
                    self.test_step_outputs_context[f"{module_name}_{name}"].append([float(val)])

        input_strs = ["target"]
        if self.test_cfg.eval_context_views:
            input_strs.insert(0, "context")

        depth_vmin, depth_vmax = self._compute_depth_range(output, input_strs, module)
        error_vmax = self._compute_error_vmax(output, input_strs, module, batch)

        # Save the initializer's predicted input-view (context) depth — distinct from the rendered
        # depth saved in the loop below. Only the initializer carries a predicted depth.
        if (self.test_cfg.save_init_pred_depth and isinstance(output, InitializerOutput)
                and output.depths is not None):
            ctx_indices = batch["context"]["index"][0]
            self.test_save_pred_depth(output.depths[0], ctx_indices, output_path, scene_name, "context",
                                      vmin=depth_vmin, vmax=depth_vmax)

        for input_str in input_strs:
            # Full-V frame indices for filename labelling. Safe to use even when
            # opt_batch_size < V because Optimizer._save_post_update_renders always
            # renders the full V views at test time.
            indices = batch[input_str]["index"][0]  # (V,)
            renders_list, iterations = self._get_renders_list(output, input_str, module)
            if renders_list is None:
                continue

            rgb_gt = batch[input_str]["image"][0].cpu()  # (V, 3, H, W)
            depth_gt = batch[input_str].get("depth", None)

            # save pred rgbs
            if self.test_cfg.save_render_image:
                if self.test_cfg.save_render_image_last_only:
                    self.test_save_last_rendered_images(renders_list, indices, output_path, scene_name, input_str)
                else:
                    self.test_save_rendered_images(renders_list, indices, output_path, scene_name, input_str)

            # save gt rgbs
            if self.test_cfg.save_gt_image and rgb_gt is not None:
                self.test_save_gt_images(rgb_gt, indices, output_path, scene_name, input_str)

            # save rgb error maps
            if self.test_cfg.save_error_image and rgb_gt is not None:
                self.test_save_rendered_errors(
                    renders_list, rgb_gt, indices, output_path, scene_name, input_str,
                    vmin=0.0,
                    vmax=error_vmax,
                )

            # save depths
            if self.test_cfg.save_render_depth:
                self.test_save_rendered_depth(renders_list, indices, output_path, scene_name, input_str,
                                              vmin=depth_vmin, vmax=depth_vmax)

            # save gt depths
            if self.test_cfg.save_gt_depth and depth_gt is not None:
                self.test_save_gt_depth(depth_gt, indices, output_path, scene_name, input_str,
                                        vmin=depth_vmin, vmax=depth_vmax)
            # Compute scores
            if self.test_cfg.compute_scores:
                print("\nComputing scores...")
                self._compute_and_save_scores(
                    module, output, renders_list, rgb_gt, iterations, input_str, module_name,
                    out_dir=output_path / "metrics" / scene_name,
                    extra_scene_metrics=extra_scene_metrics,
                )

        # Save the optimization videos for the target view (one mp4 per enabled mode).
        if module is not None and self.test_cfg.save_video and isinstance(output, OptimizerOutput):
            self._save_optimization_videos(batch, output, module, h, v, w, scene_name, output_path)

    # region ==================== Save Results Methods =======================
    @staticmethod
    def test_save_cameras_json(batch: BatchedExample, output_path: Path) -> None:
        """Save raw extrinsics/intrinsics (cam-to-world, normalized) for context + target as JSON."""
        scene_name = batch["scene"][0]
        cameras_dir = output_path / "cameras"
        cameras_dir.mkdir(parents=True, exist_ok=True)
        relevant_keys = ["extrinsics", "intrinsics"]
        cameras_data = {
            "scene": scene_name,
            "context": {key: batch["context"][key][0].cpu().tolist() for key in relevant_keys},
            "target": {key: batch["target"][key][0].cpu().tolist() for key in relevant_keys},
            "resolution": list(batch["context"]["image"].shape[-2:]),
        }
        cameras_path = cameras_dir / f"{scene_name}_cameras.json"
        with open(cameras_path, "w") as f:
            json.dump(cameras_data, f, indent=4)
        print(f"Saved cameras JSON to {cameras_path}")

    @staticmethod
    def test_save_cameras_npz(batch: BatchedExample, output_path: Path) -> None:
        """Save cameras in renderer-ready form: viewmats=inverse(extrinsics) (world-to-cam),
        Ks=intrinsics * diag(W, H, 1) (pixel-space). Mirrors
        GSplatDecoderSplattingCUDA.forward (gsplat_decoder_splatting_cuda.py:137-140).
        """
        scene_name = batch["scene"][0]
        cameras_dir = output_path / "cameras"
        cameras_dir.mkdir(parents=True, exist_ok=True)
        npz_data = {"scene": scene_name}
        for input_str in ("context", "target"):
            view = batch[input_str]
            extrinsics = view["extrinsics"][0]  # [V, 4, 4] cam-to-world
            intrinsics = view["intrinsics"][0]  # [V, 3, 3] normalized
            h, w = view["image"].shape[-2:]
            viewmats = extrinsics.inverse()  # [V, 4, 4] world-to-cam
            scale = intrinsics.new_tensor([[w], [h], [1]])
            Ks = intrinsics * scale  # [V, 3, 3] pixel-space
            npz_data[f"{input_str}_viewmats"] = viewmats.cpu().numpy()
            npz_data[f"{input_str}_Ks"] = Ks.cpu().numpy()
            npz_data[f"{input_str}_image_shape"] = np.array([h, w], dtype=np.int64)
        cameras_npz_path = cameras_dir / f"{scene_name}_cameras.npz"
        np.savez(cameras_npz_path, **npz_data)
        print(f"Saved renderer-ready cameras NPZ to {cameras_npz_path}")

    @staticmethod
    def test_save_rendered_images(renders_list: list, indices, output_path, scene_name, input_str):
        """Save the per-view optimization trajectory as <output>/images/<scene>/color_<input_str>/<index>.png,
        with iterations concatenated along width. Also dumps the last iteration separately."""
        out_dir = output_path / "images" / scene_name / f"color_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} images"):
            color = []
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                color.append(iter_rgbs[i])
            color = torch.cat(color, dim=-1)  # concat along width
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs
            del color
        # save last image separately too
        MetaTrainer.test_save_last_rendered_images(renders_list, indices, output_path, scene_name, input_str)

    @staticmethod
    def test_save_last_rendered_images(renders_list: list, indices, output_path, scene_name, input_str):
        """Save only the final-iteration render per view at <output>/images/<scene>/last/color_<input_str>/<index>.png."""
        out_dir = output_path / "images" / scene_name / "last" / f"color_{input_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} last images"):
            iter_renders = renders_list[-1]
            iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
            color = iter_rgbs[i]
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs
            del color

    @staticmethod
    def test_save_gt_images(rgb_gt, indices, output_path, scene_name, input_str):
        """Save GT images alongside the last render at <output>/images/<scene>/last/color_<input_str>/<index>_gt.png."""
        out_dir = output_path / "images" / scene_name / "last" / f"color_{input_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, gt in tqdm(zip(indices, rgb_gt), desc=f"Saving {input_str} GT images"):
            save_image(gt, out_dir / f"{index:06d}_gt.png")

    @staticmethod
    def test_save_rendered_depth(renders_list: list, indices, output_path, scene_name, input_str,
                                 vmin: float = 0.0,
                                 vmax: float = 1.0):
        """Save the rendered-depth trajectory (depth rasterized from the Gaussians) as colormapped PNGs at
        <output>/images/<scene>/rendered_depth_<input_str>/<index>.png, iterations concatenated along width.
        This is the rendered depth, not the initializer's predicted depth. vmin/vmax control the per-scene
        depth normalization."""
        out_dir = output_path / "images" / scene_name / f"rendered_depth_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} rendered depths"):
            depth = []
            for iter_renders in renders_list:
                iter_depths = iter_renders.depth  # (1, V, H, W)
                assert iter_depths is not None, "Depths not found in renders."
                iter_depths = iter_depths[0]  # (V, H, W)
                depth.append(iter_depths[i])  # (H, W)
            depth = torch.cat(depth, dim=-1)  # concat along width
            color = viz_depth_tensor(depth, return_numpy=False, as_uint8=False, vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_depths
            del color

    @staticmethod
    def test_save_pred_depth(pred_depths, indices, output_path, scene_name, input_str,
                             vmin: float = 0.0, vmax: float = 1.0):
        """Save the initializer's predicted input-view depth (the encoder/monodepth estimate used to place
        Gaussians) as colormapped PNGs at <output>/images/<scene>/pred_depth_<input_str>/<index>.png.
        Distinct from the rendered depth above; uses the same vmin/vmax for comparability."""
        out_dir = output_path / "images" / scene_name / f"pred_depth_{input_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} pred depths"):
            color = viz_depth_tensor(pred_depths[i], return_numpy=False, as_uint8=False, vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}.png")
            del color

    @staticmethod
    def test_save_gt_depth(depth_gt, indices, output_path, scene_name, input_str, vmin: float = 0.0,
                           vmax: float = 1.0):
        """Save GT depth colormapped under <output>/images/<scene>/rendered_depth_<input_str>/<index>_gt.png
        using the same vmin/vmax as the rendered depths."""
        out_dir = output_path / "images" / scene_name / f"rendered_depth_{input_str}"
        depth_gt = depth_gt[0]  # (B, V, H, W) -> (V, H, W); iterate views, not the batch dim
        for index, gt in tqdm(zip(indices, depth_gt), desc=f"Saving {input_str} GT depths"):
            color = viz_depth_tensor(gt, return_numpy=False, as_uint8=False, vmin=vmin, vmax=vmax)  # gt is (H, W)
            save_image(color, out_dir / f"{index:06d}_gt.png")

    @staticmethod
    def test_save_rendered_errors(renders_list: list, rgb_gt, indices, output_path, scene_name, input_str,
                                  vmin: float = 0.0,
                                  vmax: float = 1.0):
        """Save per-view RGB-error magma maps at <output>/images/<scene>/error_<input_str>/<index>.png,
        iterations concatenated along width. vmin/vmax come from the per-scene 99th-percentile error
        computed in `_compute_error_vmax` for consistent visualization."""
        # rgb_gt is (V, 3, H, W). Looping per view (outer) and per iteration (inner)
        # keeps only one view's iterations in memory at a time.
        out_dir = output_path / "images" / scene_name / f"error_{input_str}"
        for i, index in tqdm(enumerate(indices), desc=f"Saving {input_str} errors"):
            error_maps = []
            for iter_renders in renders_list:
                iter_rgbs = iter_renders.color[0]  # (V, 3, H, W)
                error_maps.append((iter_rgbs[i] - rgb_gt[i].to(iter_rgbs)).abs().mean(0))  # (H, W)
            error_map = torch.cat(error_maps, dim=-1)  # concat along width
            color = viz_depth_tensor(error_map, return_numpy=False, as_uint8=False, colormap='magma',
                                     vmin=vmin, vmax=vmax)
            save_image(color, out_dir / f"{index:06d}.png")
            del iter_rgbs, error_maps, error_map

    def init_output_dict_for_new_scene(self, input_str, tag=None):
        """Append an empty per-iteration sublist to test_step_outputs_<input_str> for every metric we
        will collect (psnr, ssim, lpips×2, iterations, time, gaussians, nonzero_grads). Called once at
        the start of each scene's evaluation; `tag` (e.g. module class name) prefixes each key so
        init / optimizer / postprocess buckets don't collide."""
        tag = "" if tag is None else f"{tag}_"

        if input_str == "target":
            output_dict = self.test_step_outputs_target
        elif input_str == "context":
            output_dict = self.test_step_outputs_context
        else:
            raise ValueError(f"Unknown input_str={input_str}")

        for metric in ["psnr", "ssim", "lpips", "iterations", "time", "gaussians", "nonzero_grads"]:
            if metric == "lpips":
                # alex
                key = f"{tag}alex_lpips"
                output_dict[key].append([])
                # vgg
                key = f"{tag}vgg_lpips"
                output_dict[key].append([])
            else:
                key = f"{tag}{metric}"
                output_dict[key].append([])

    # endregion

    # region ==================== Video Rendering Methods ====================
    def _save_optimization_videos(
            self, batch, output, module, h, v, w, scene_name, output_path,
    ) -> None:
        """Render and save the enabled optimization videos for the target view.

        Each enabled mode writes one mp4 (plus a matching per-frame iteration-label JSON) to
        videos/<scene>:
        - save_video_optim: the chosen view (save_video_view_index) rendered at every save checkpoint.
        - save_video_orbit: at each step in save_video_orbit_steps, a full camera orbit around the
          scene, optionally preceded by the per-checkpoint frames (save_video_orbit_with_optim).
        - save_video_optim_orbit: the per-checkpoint frames with a short orbit spliced in at
          save_video_optim_orbit_steps (spanning view -> view + save_video_orbit_span).
        """
        _, iterations = self._get_renders_list(output, "target", module)
        gaussian_list = output.gaussian_list

        if self.test_cfg.save_video_optim:
            self.render_optim_video(
                batch, h, v, w, "target", iterations, gaussian_list, scene_name, output_path,
            )
        if self.test_cfg.save_video_orbit:
            for t in self.test_cfg.save_video_orbit_steps:
                self.render_orbit_video(
                    batch, h, v, w, "target", iterations, gaussian_list, scene_name, output_path,
                    orbit_step=t,
                    with_optim=self.test_cfg.save_video_orbit_with_optim,
                )
        if self.test_cfg.save_video_optim_orbit:
            self.render_optim_orbit_video(
                batch, h, v, w, "target", iterations, gaussian_list, scene_name, output_path,
                orbit_steps=self.test_cfg.save_video_optim_orbit_steps,
                orbit_span=self.test_cfg.save_video_orbit_span,
            )

    def render_optim_video(
            self, batch, h, v, w, input_str, all_iterations, gaussian_list, scene_name, output_path,
    ) -> None:
        """For each save-checkpoint, render only `save_video_view_index` and append it
        (duplicated `save_video_frame_repeat` times so the optimization trajectory at
        that single view stands out)."""
        view = self.test_cfg.save_video_view_index
        duplicate = self.test_cfg.save_video_frame_repeat

        all_frames, combined_iterations = [], []
        for i, t in enumerate(all_iterations):
            decoder_output = self.test_render_videos_views(
                batch, gaussian_list[i], h, v, w, input_str, start=view, end=view + 1,
            )
            frame = decoder_output.color[0].detach().cpu()  # (1, 3, H, W)
            assert frame.shape[0] == 1, f"{frame.shape}"
            all_frames += [frame[0]] * duplicate  # (3, H, W)
            combined_iterations.extend([t] * duplicate)

        self._save_video_file(input_str, "optim", all_frames, combined_iterations,
                              scene_name, output_path)

    def render_orbit_video(
            self, batch, h, v, w, input_str, all_iterations, gaussian_list, scene_name, output_path,
            orbit_step: int, with_optim: bool,
    ) -> None:
        """At one chosen iteration, orbit the camera around the scene; optionally also append
        the per-iteration fixed-view frames leading up to it."""
        view = self.test_cfg.save_video_view_index
        duplicate = self.test_cfg.save_video_frame_repeat

        path = self._fitted_orbit_path(batch, input_str)

        all_frames, combined_iterations = [], []
        for i, t in enumerate(all_iterations):
            if with_optim:
                decoder_output = self.test_render_videos_views(
                    batch, gaussian_list[i], h, v, w, input_str, start=view, end=view + 1,
                )
                frame = decoder_output.color[0].detach().cpu()
                all_frames += [frame[0]] * duplicate
                combined_iterations.extend([t] * duplicate)

            if t == orbit_step:
                decoder_output = self.test_render_videos_views(
                    batch, gaussian_list[i], h, v, w, input_str, poses=path,
                    start=None, end=None,
                )
                self._collect_orbit_frames(decoder_output, all_frames, combined_iterations)
                if not with_optim:
                    break  # nothing more to render

        self._save_video_file(input_str, f"orbit_{orbit_step}",
                              all_frames, combined_iterations, scene_name, output_path)

    def render_optim_orbit_video(
            self, batch, h, v, w, input_str, all_iterations, gaussian_list, scene_name, output_path,
            orbit_steps, orbit_span: int,
    ) -> None:
        """Render the per-iteration fixed-view trajectory; at the given iterations also splice
        in a short orbit (rendered from view → view+orbit_span)."""
        view = self.test_cfg.save_video_view_index
        duplicate = self.test_cfg.save_video_frame_repeat
        path = self._fitted_orbit_path(batch, input_str)
        # A fitted path is rendered whole; the span selects a slice of the views instead.
        orbit_start = view if orbit_span > 0 and path is None else None
        orbit_end = view + orbit_span if orbit_span > 0 and path is None else None

        all_frames, combined_iterations = [], []
        for i, t in enumerate(all_iterations):
            decoder_output = self.test_render_videos_views(
                batch, gaussian_list[i], h, v, w, input_str, start=view, end=view + 1,
            )
            frame = decoder_output.color[0].detach().cpu()
            all_frames += [frame[0]] * duplicate
            combined_iterations.extend([t] * duplicate)

            if t in orbit_steps:
                decoder_output = self.test_render_videos_views(
                    batch, gaussian_list[i], h, v, w, input_str, poses=path,
                    start=orbit_start, end=orbit_end,
                )
                self._collect_orbit_frames(decoder_output, all_frames, combined_iterations)

        self._save_video_file(input_str, "optim_orbit", all_frames, combined_iterations,
                              scene_name, output_path)

    def _fitted_orbit_path(self, batch, input_str) -> Float[Tensor, "frame 4 4"] | None:
        """The camera path the spliced-in sweeps follow, or None to sweep the views themselves.

        The arc is anchored at the view the rest of the video sits at, so it leaves and returns
        to it, and is fitted against every camera the scene has, context and target alike, so a
        held-out target subset is still placed against the full camera distribution.
        """
        if not self.test_cfg.save_video_orbit_fitted_path:
            return None
        extrinsics = torch.cat([batch["context"]["extrinsics"][0],
                                batch["target"]["extrinsics"][0]], dim=0)
        anchor = batch[input_str]["extrinsics"][0, self.test_cfg.save_video_view_index]
        return generate_orbit_path(extrinsics, anchor,
                                   self.test_cfg.save_video_orbit_fitted_path_frames,
                                   self.test_cfg.save_video_orbit_fitted_path_span_deg)

    def _collect_orbit_frames(self, decoder_output, all_frames: list, combined_iterations: list) -> None:
        """Append forward+backward sweeps of the rendered camera path, each frame held for
        save_video_orbit_frame_repeat frames so the sweep plays at a watchable speed."""
        frames_t = decoder_output.color[0].detach().cpu()  # (num_frames, 3, H, W)
        sweep = list(frames_t) + list(frames_t.flip(0))
        frames = [f for f in sweep
                  for _ in range(self.test_cfg.save_video_orbit_frame_repeat)]
        frames = frames * self.test_cfg.save_video_orbit_sweeps
        all_frames += frames
        combined_iterations.extend(['orbit'] * len(frames))

    @staticmethod
    def _save_video_file(input_str: str, save_str: str, all_frames: list, combined_iterations: list,
                         scene_name: str, output_path: Path) -> None:
        """Write the mp4 and the matching per-frame iteration-label JSON to videos/<scene>."""
        out_dir = output_path / "videos" / scene_name
        out_dir.mkdir(parents=True, exist_ok=True)
        save_video(all_frames, out_dir / f"{input_str}_{save_str}.mp4")
        with open(out_dir / f"{input_str}_{save_str}_iterations.json", 'w') as f:
            json.dump(combined_iterations, f, indent=4)

    def test_render_videos_views(self, batch, gaussians, h, v, w, input_str="target", poses=None, start=None, end=None):
        """Render the Gaussians along a camera path and return the decoder output.

        The path is the input_str views' poses (optionally camera-stabilized) or the explicit `poses`;
        rendering is split into chunks of test_cfg.render_chunk_size when that is set.
        """
        gaussians = gaussians.to(batch["target"]["image"].device)
        with self.benchmarker.time("decoder", num_calls=v):
            if poses is None:
                camera_poses = batch[input_str]["extrinsics"]

                if self.test_cfg.stabilize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)
            else:
                camera_poses = poses.unsqueeze(0)

            if self.test_cfg.render_chunk_size is not None:
                assert start is None
                assert end is None
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)
                    curr_output = self.scene_decoder.forward_batch(gaussians, batch, (h, w),
                                                                   input_str=input_str,
                                                                   start=start, end=end, camera_poses=camera_poses)

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat((output.color, curr_output.color), dim=1)

            else:
                output = self.scene_decoder.forward_batch(gaussians.to(batch["target"]["image"].device),
                                                          batch, (h, w),
                                                          input_str=input_str,
                                                          camera_poses=camera_poses,
                                                          start=start,
                                                          end=end)
        return output

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        """Log a video interpolating the camera between the two context views (or context->target when there is one context view)."""
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        """Log a context-view interpolation video with an added exaggerated wobble transform along the path (needs 2 context views)."""
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
            self,
            batch: BatchedExample,
            trajectory_fn: TrajectoryFn,
            name: str,
            num_frames: int = 30,
            smooth: bool = True,
            loop_reverse: bool = True,
    ) -> None:
        """Render num_frames of the init Gaussians along the trajectory_fn camera path and log an
        RGB+depth video to wandb under video/{name}."""
        if self.train_cfg.no_log_video:
            return
        gaussians = self.get_init_gaussians(batch, is_training=False).gaussians

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.scene_decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        rgb_pred = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], depth_map(output.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(rgb_pred, rgb_pred)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    # endregion

    # region ==================== Delegation Methods =========================

    def optimize_scene(self, *args, **kwargs):
        """Delegate to SceneTrainer's optimize_scene."""
        return self.scene_trainer.optimize_scene(*args, **kwargs)

    def get_init_gaussians(self, *args, **kwargs):
        """Delegate to SceneTrainer's get_init_gaussians."""
        return self.scene_trainer.get_init_gaussians(*args, **kwargs)

    def init_gaussians_and_render(self, *args, **kwargs):
        """Delegate to SceneTrainer's init_gaussians_and_render."""
        return self.scene_trainer.init_gaussians_and_render(*args, **kwargs)

    def test_postprocess_gaussians(self, *args, **kwargs):
        """Delegate to SceneTrainer's test_postprocess_gaussians."""
        return self.scene_trainer.test_postprocess_gaussians(*args, **kwargs)

    @property
    def scene_initializer(self):
        """Delegate to SceneTrainer's initializer."""
        return self.scene_trainer.initializer

    @property
    def scene_optimizer(self):
        """Delegate to SceneTrainer's optimizer."""
        return self.scene_trainer.optimizer

    @property
    def scene_decoder(self):
        """Delegate to SceneTrainer's decoder."""
        return self.scene_trainer.decoder

    # endregion
