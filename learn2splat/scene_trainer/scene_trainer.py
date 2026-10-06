"""Inner per-scene loop of the two-level pipeline.

`SceneTrainer` composes the three pipeline stages for a single scene: an Initializer produces the
starting Gaussians from the input views, an Optimizer iteratively refines them, and a Decoder renders
novel views. Each stage is chosen by name from its registry (see the package `__init__.py` files). The
outer meta-loop that drives many scenes and computes losses lives in `meta_trainer.py`.
"""

import math
import random
from typing import Optional, Mapping, Any
from warnings import warn

import torch
from einops import rearrange
from lightning_fabric.utilities import move_data_to_device
from torch import Tensor, nn
from tqdm import tqdm

from learn2splat.dataset import DatasetCfg
from learn2splat.dataset.data_types import BatchedExample
from learn2splat.dataset.view_sampler.view_sampler_bounded_v2 import farthest_point_sample
from learn2splat.misc.benchmarker import Benchmarker
from learn2splat.misc.general_utils import SkipBatchException
from learn2splat.misc.io import FrequencyScheduler
from learn2splat.misc.step_tracker import StepTracker
from learn2splat.model.decoder.decoder import Decoder, DepthRenderingMode
from learn2splat.model.types import Gaussians
from learn2splat.paths import DEBUG
from learn2splat.scene_trainer.initializer import get_scene_initializer
from learn2splat.scene_trainer.initializer.initializer import Initializer, InitializerOutput
from learn2splat.scene_trainer.optimizer import get_scene_optimizer
from learn2splat.scene_trainer.optimizer.optimizer import OptimizerInput, Optimizer, OptimizerOutput, OptimizerPreviousOutput
from learn2splat.scene_trainer.postprocessing import PostProcessing3DGS
from learn2splat.scene_trainer.scene_trainer_cfg import SceneTrainerCfg
from learn2splat.meta_trainer.meta_trainer_cfg import TestCfg, TrainCfg


class SceneTrainer(nn.Module):
    test_cfg: TestCfg
    train_cfg: TrainCfg
    scene_trainer_cfg: SceneTrainerCfg
    decoder: Decoder
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
            self,
            test_cfg: TestCfg,
            train_cfg: TrainCfg,
            scene_trainer_cfg: SceneTrainerCfg,
            decoder: Decoder,
            step_tracker: StepTracker | None,
            benchmarker: Benchmarker,
            eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.scene_trainer_cfg = scene_trainer_cfg

        # Set up the model
        self.initializer = get_scene_initializer(scene_trainer_cfg.scene_initializer)
        # Per-source initializer overrides for mixed datasets, keyed by source key; empty for single
        # datasets / uniform mixes -> every scene uses self.initializer. A ModuleDict so the sub-inits
        # register as submodules (device/dtype moves with the trainer).
        self.source_initializers = nn.ModuleDict({
            key: get_scene_initializer(c)
            for key, c in (scene_trainer_cfg.scene_initializers or {}).items()
        })

        # Scene trainer performs updates
        if self.scene_trainer_cfg.num_update_steps > 0:
            optimizer_save_every = FrequencyScheduler(
                frequencies=self.test_cfg.save_every_freq,
                steps=self.test_cfg.save_every_steps,
                iters=self.test_cfg.save_at_iters,
                last_step=self.scene_trainer_cfg.num_update_steps,
                enable_context=self.test_cfg.eval_context_views,
            )
            self.optimizer: Optimizer | None = get_scene_optimizer(scene_trainer_cfg.scene_optimizer)
            self.optimizer.save_every = optimizer_save_every
            # The renders stored per saved step keep their depth map only for the outputs that
            # read it: the depth PNGs (whose color scale is shared with the initializer's
            # predicted depth), the videos, and the train-time render-depth metrics.
            self.optimizer.keep_render_depth = (
                self.test_cfg.save_render_depth or self.test_cfg.save_gt_depth
                or self.test_cfg.save_init_pred_depth or self.test_cfg.save_video
                or self.train_cfg.eval_render_depth or self.train_cfg.viz_render_depth
            )
        else:
            self.optimizer = None

        self.decoder = decoder

        # Shared with the MetaTrainer so init/decoder timings land in one place.
        self.benchmarker = benchmarker

        if self.test_cfg.postprocessing is not None and self.test_cfg.postprocessing.is_active:
            self.postprocess_save_every = FrequencyScheduler(
                frequencies=self.test_cfg.save_every_freq,
                steps=self.test_cfg.save_every_steps,
                iters=self.test_cfg.save_at_iters,
                last_step=self.test_cfg.postprocessing.steps,
                enable_context=self.test_cfg.eval_context_views,
            )
            self.postprocess = PostProcessing3DGS(
                cfg=self.test_cfg.postprocessing,
                save_every=self.postprocess_save_every
            )
        else:
            self.postprocess = None

    @property
    def device(self):
        # Use try/except to catch StopIteration explicitly rather than letting it
        # propagate, which silently terminates PL's generator-based test loop.
        try:
            return next(self.parameters()).device
        except StopIteration:
            pass
        try:
            return next(self.buffers()).device
        except StopIteration:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Load weights into initializer and optimizer, skipping non-learned strategies."""
        # Remove scene_trainer prefix from state dict keys if it exists
        state_dict = {k.replace("scene_trainer.", ""): v for k, v in state_dict.items()}

        prefixes = {s.split(".")[0] for s in state_dict.keys()}
        assert all([p in ["initializer", "optimizer"] for p in
                    prefixes]), f"State dict keys must start with 'initializer.' or 'optimizer.', got {prefixes}"

        if self.initializer.strategy == "learned":
            initializer_state_dict = {k[len("initializer."):]: v for k, v in state_dict.items() if
                                      k.startswith("initializer.")}
            self.initializer.load_state_dict(initializer_state_dict, strict=strict)

        if self.optimizer is not None:
            if self.optimizer.strategy == "learned":
                optimizer_state_dict = {k[len("optimizer."):]: v for k, v in state_dict.items() if
                                        k.startswith("optimizer.")}
                self.optimizer.load_state_dict(optimizer_state_dict, strict=strict)

    def optimize_scene(
            self,
            batch: BatchedExample,
            prev_output: InitializerOutput | OptimizerPreviousOutput,
            curr_iter=0,
            debug_dict=None,
            num_update_steps=None,
            disable_tqdm=False,
            allow_partial_on_skip=False,
            depth_mode: DepthRenderingMode | None = None,
            **kwargs,
    ) -> OptimizerOutput:
        """Optimize the Gaussians for a single scene, in training or at test time.

        `prev_output` is an InitializerOutput for a new scene (the optimizer's on_scene_start
        converts it) or an OptimizerPreviousOutput when resuming from the ckpt buffer; `curr_iter`
        is 0 for a new scene. `num_update_steps` defaults to the config value.

        With `allow_partial_on_skip`, a SkipBatchException or CUDA OOM during an update step stops
        the loop and returns the steps completed so far instead of propagating, so the test path
        still reports metrics for a scene that fails mid-optimization.

        Returns the optimized Gaussians and the renders of the intermediate steps.
        """

        assert self.optimizer is not None, "Optimizer is not initialized."

        if num_update_steps is None:
            num_update_steps = self.get_num_update_steps()

        optimizer_input = OptimizerInput(
            context=batch["context"],  # this is full context, not iter batch
            target=batch["target"],
            prev_output=prev_output,
            renderer=self.decoder,
            iter_batch_size=self.scene_trainer_cfg.iter_batch_size,  # For rendering in batches
            debug_dict=debug_dict,
            depth_mode=depth_mode,
        )

        # Handles both new scenes (InitializerOutput) and ckpt buffer continuations (OptimizerPreviousOutput).
        self.optimizer.validate_input(optimizer_input)
        self.optimizer.scene_start_event_start.record()
        self.optimizer.on_scene_start(optimizer_input)
        self.optimizer.scene_start_event_end.record()
        assert isinstance(optimizer_input.prev_output, OptimizerPreviousOutput), \
            f"Should be OptimizerPreviousOutput after on_scene_start, got {type(optimizer_input.prev_output)}"

        # Initialize empty output to store intermediate and final results
        optimizer_output: OptimizerOutput = OptimizerOutput.empty(t=curr_iter)
        optimizer_output.T = num_update_steps

        # Insert the initialization into position 0 of the output lists so downstream
        # consumers (evaluation, plotting, ckpt buffer) can treat init uniformly with
        # the optimizer steps. Handles both InitializerOutput and OptimizerPreviousOutput,
        # and skips insertion during the rollout that saves no per-step outputs.
        self._insert_init_into_output(optimizer_output, prev_output)

        # SH degree scheduling (inspired by gsplat simple_trainer):
        # sh_degree_to_use = min(step // sh_degree_interval, max_sh_degree)
        sh_degree_interval = self.scene_trainer_cfg.sh_degree_interval
        if sh_degree_interval > 0:
            max_sh_degree = int(math.sqrt(optimizer_input.prev_output.gaussians.harmonics.shape[-1])) - 1

        # The following lists are truncated to align them to the failed step when a scene is
        # failed in the middle at test time (allow_partial_on_skip).
        if allow_partial_on_skip:
            aligned_lists = [
                optimizer_output.target_render_list, optimizer_output.context_render_list,
                optimizer_output.gaussian_list,
                optimizer_output.target_index_list, optimizer_output.context_index_list,
            ]

        # Loop over update steps
        pbar = tqdm(range(num_update_steps),
                    desc="Scene-Opt",
                    disable=(self.training or num_update_steps < 20 or DEBUG) or disable_tqdm,
                    total=num_update_steps)
        for step in pbar:

            # Snapshot the aligned lists' lengths so a step that fails partway can be rolled back
            # to the steps that fully completed (lengths grow per step, so this is per-step).
            if allow_partial_on_skip:
                pre_step_lens = [len(lst) for lst in aligned_lists]

            # Sample minibatch of context/target views and move to device
            optimizer_input.context, batch_idx = self.batchify_views(batch, "context", self.device)
            if batch_idx is not None:
                optimizer_output.context_index_list.append(batch_idx)
            optimizer_input.target, batch_idx = self.batchify_views(batch, "target", self.device)
            if batch_idx is not None:
                optimizer_output.target_index_list.append(batch_idx)

            # Build per-step kwargs, adding SH degree if scheduler is active
            step_kwargs = dict(kwargs)
            if sh_degree_interval > 0:
                step_kwargs["sh_degree"] = min(step // sh_degree_interval, max_sh_degree)

            # Single optimization step
            # Optimizer output is updated in place, but we return it for clarity
            try:
                optimizer_output = self.optimizer(
                    step,
                    optimizer_input,
                    optimizer_output,
                    full_context=batch["context"],
                    full_target=batch["target"],
                    **step_kwargs
                )
            # torch.cuda.OutOfMemoryError is an alias of torch.OutOfMemoryError; one covers both.
            # Kept specific to the two expected failures so a generic RuntimeError (a real bug)
            # still propagates and drops the scene rather than being half-recovered.
            except (SkipBatchException, torch.OutOfMemoryError) as e:
                if not allow_partial_on_skip:
                    raise
                torch.cuda.empty_cache()
                # No step completed -> nothing to score; re-raise so the caller drops the scene.
                if optimizer_output.t == curr_iter:
                    raise
                # >=1 step completed: roll back the failed step's partial appends so the output lists
                # stay aligned to the fully-completed steps, then stop early and score those. (At test
                # time SkipBatch fires before any append, so the rollback only matters for an OOM
                # mid-step.) The per-iteration module logs may keep one extra (unread) entry; downstream
                # indexes them by completed-step number, so that is harmless.
                warn(f"Optimization stopped early at step {step}/{num_update_steps} ({type(e).__name__}: {e}); "
                     f"scoring the {step} completed step(s).")
                for lst, n in zip(aligned_lists, pre_step_lens):
                    del lst[n:]
                break
            optimizer_output.t += 1
            # Show the optimizer's inner loss on the bar every few steps (when shown; absent for
            # optimizers that don't expose one). float() here, not per-step in the optimizer, so the
            # GPU->CPU sync only happens while the bar is actually being displayed.
            if not pbar.disable and step % 20 == 0 and \
                    (inner := getattr(self.optimizer, "last_inner_loss", None)) is not None:
                postfix = {"loss": f"{float(inner):.4f}"}
                components = getattr(self.optimizer, "last_inner_loss_components", None)
                if components:
                    postfix.update({k: f"{float(v):.4f}" for k, v in components.items()})
                pbar.set_postfix(postfix, refresh=False)

        # Sync GPU before reading scene_start elapsed time (events were recorded before the loop).
        torch.cuda.synchronize()
        self.optimizer.benchmarker.record(
            "scene_start",
            self.optimizer.scene_start_event_start.elapsed_time(self.optimizer.scene_start_event_end),
        )

        self.optimizer.on_scene_end()

        # Extract the last output (for ckpt buffer)
        optimizer_output.last_prev_output = optimizer_input.prev_output

        return optimizer_output

    def batchify_views(self, scene_batch, input_str, device, batch_size=None):
        """
        Sample a subset of views from the batch for the current optimization step.

        Args:
            scene_batch: Full batch containing context/target views
            input_str: "context" or "target"
            device: Target device for the subset
            batch_size: Override batch size. If None, uses config-based batch size.

        Returns:
            Tuple of (subset_batch, indices) where indices is None if no subsampling
        """
        scene_batch_split = scene_batch[input_str]

        # Determine batch size (may be randomized during training)
        if batch_size is None:
            batch_size = self._get_batch_size()
        v_all = scene_batch_split["image"].shape[1]
        if batch_size <= 0 or batch_size >= v_all:
            return scene_batch_split, None

        strategy = self.scene_trainer_cfg.opt_batch_strategy
        views_idxs = self._sample_indices(scene_batch_split, batch_size, strategy)  # [scene_batch, views_batch]
        views_batch = scene_batch_split.batchify_views(views_idxs)
        views_batch = move_data_to_device(views_batch, device)
        return views_batch, views_idxs

    def _get_batch_size(self) -> int:
        """Determine the batch size, potentially randomized during training."""
        batch_size = self.scene_trainer_cfg.opt_batch_size

        # Randomize batch size if configured (training or promoting buffer)
        if self.scene_trainer_cfg.opt_batch_size_max > 0:
            if self.training or self.promoting_buffer_sample:
                batch_size = random.randint(
                    self.scene_trainer_cfg.opt_batch_size_min,
                    self.scene_trainer_cfg.opt_batch_size_max
                )
        return batch_size

    def _sample_indices(self, batch_split, views_batch_size: int, strategy: str) -> torch.Tensor:
        """Sample a minibatch of view indices using the configured strategy.
        Uses viewpoint_stack to cycle through all views before reshuffling."""

        # Initialize or reset viewpoint stack for new epoch
        batch_split.reset_viewpoint_stack_if_needed(strategy, views_batch_size)
        viewpoint_stack = batch_split.viewpoint_stack  # [B, V]
        scene_batch, v = viewpoint_stack.shape

        views_batch_size = min(views_batch_size, v)

        if strategy in ["random", "sequential"]:
            # Take views from the front of the stack (shuffled if random)
            batch_idxs = viewpoint_stack[:, :views_batch_size]
            idx_to_remove = batch_idxs

        elif strategy == "neighbors":
            # Use first view in stack as center, select its neighbors
            extrinsics = batch_split["extrinsics"]
            if extrinsics.ndim == 4:  # [B, V, 4, 4]
                assert extrinsics.shape[0] == 1, "Batch size must be 1 for neighbor sampling"
                extrinsics = extrinsics[0]

            center_idx = viewpoint_stack[0, 0]
            batch_idxs = self._get_neighbor_indices(extrinsics, center_idx, views_batch_size)
            idx_to_remove = torch.tensor([[center_idx]])  # Only remove center from stack
        elif strategy == "fps":
            # FPS on camera positions of the remaining views in the stack
            extrinsics = batch_split["extrinsics"]  # [B, V_total, 4, 4]
            B = extrinsics.shape[0]
            batch_arange = torch.arange(B, device=self.device)[:, None]
            stack_positions = extrinsics[batch_arange, viewpoint_stack][:, :, :3, 3]  # [B, V_stack, 3]
            fps_local_idxs = farthest_point_sample(stack_positions, views_batch_size,
                                                   first_idx_strategy="random")  # [B, K]
            batch_idxs = viewpoint_stack[batch_arange, fps_local_idxs]  # [B, K]
            idx_to_remove = batch_idxs
        else:
            raise ValueError(f"Unknown opt_batch_strategy: {strategy}")

        # Remove used indices from the stack, preserving order between the views separately for each batch.
        remove_mask = (viewpoint_stack.unsqueeze(-1) == idx_to_remove.unsqueeze(1)).any(-1)  # [B, V]
        batch_split.viewpoint_stack = viewpoint_stack[~remove_mask].view(scene_batch, -1)  # [B, V_used]

        return batch_idxs

    def _get_neighbor_indices(self, extrinsics, center_idx, batch_size: int) -> torch.Tensor:
        """Get indices of nearest neighbor views based on camera pose distance."""
        combined_metric = self.calc_extrinsics_dist(center_idx, extrinsics)
        return torch.argsort(combined_metric)[:batch_size].unsqueeze(0)  # [1, K]

    @staticmethod
    def calc_extrinsics_dist(center_idx, extrinsics):
        """Combined position + rotation distance from a center view to all views. Returns [V]."""
        rotations = extrinsics[:, :3, :3]  # [V, 3, 3]
        # Calculate camera center as -R^T * t
        translation = extrinsics[:, :3, [3]]  # [V, 3, 1]
        poses = -rotations.transpose(1, 2) @ translation  # [V, 3, 1]
        center_pose = poses[center_idx]  # [3, 1]
        # Calculate Euclidean distances to the center view
        dists = torch.norm(poses - center_pose.unsqueeze(0), dim=1)[0]  # [V]
        # Calculate angular differences to the center view
        center_rot = extrinsics[center_idx, :3, :3]  # [3, 3]
        # Compute rotation difference
        rot_diffs = torch.matmul(rotations, center_rot.transpose(0, 1))  # [V, 3, 3]
        # Compute angles from rotation matrices
        cos_angles = (rot_diffs[:, 0, 0] + rot_diffs[:, 1, 1] + rot_diffs[:, 2, 2] - 1) / 2  # [V]
        cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Numerical stability
        angles = torch.acos(cos_angles)  # [V]
        # Combine distance and angle into a single metric
        combined_metric = dists + angles  # [V]
        return combined_metric

    def get_num_update_steps(self) -> int:
        """Return number of optimizer steps, randomly sampled during training if train_max_refine is set."""
        if self.training and self.scene_trainer_cfg.train_max_refine > 0:
            num_updates = random.randint(
                self.scene_trainer_cfg.train_min_refine,
                self.scene_trainer_cfg.train_max_refine
            )
        else:
            num_updates = self.scene_trainer_cfg.num_update_steps
        return num_updates

    def _initializer_for(self, batch) -> Initializer:
        """Initializer for this scene's source: the per-source override when the mixed dataset tagged
        a source and one is configured, else the default `self.initializer`. Single datasets, the
        single-source eval path (get_eval_cfg carries no source tag), and un-overridden sources all
        take the default."""
        source = batch.get("source") if self.source_initializers else None
        if not source:
            return self.initializer
        key = source[0]  # batch_size=1 for mixes -> one source per batch
        # nn.ModuleDict has no .get(), so branch on __contains__/__getitem__.
        return self.source_initializers[key] if key in self.source_initializers else self.initializer

    def apply_initializer_data_shim(self, batch):
        """Apply the data-shim of the initializer selected for this batch's source: the per-source
        override for mixed datasets, else the default initializer (single datasets / eval path).
        Dispatched per batch rather than pre-extracted so a per-source initializer that needs its own
        shim (e.g. resplat's patch-alignment) gets it on its own scenes instead of the default's."""
        from learn2splat.dataset.data_module import get_data_shim
        return get_data_shim(self._initializer_for(batch))(batch)

    def get_init_gaussians(self, batch, is_training: bool, **kwargs) -> InitializerOutput:
        """Run the initializer to produce Gaussians from context views, with optional sliding window.

        Gradients are disabled when not training so the init model is frozen during refine-only runs.
        """
        window_size = self.train_cfg.train_window_size if is_training else self.test_cfg.inference_window_size
        with torch.set_grad_enabled(is_training):
            if window_size is not None:
                initializer_output = self.init_gaussians_with_window(batch, window_size, **kwargs)
            else:
                # In some cases we might want to pass the target as well
                # (e.g., to manipulate the poses in colmap dataset)
                initializer = self._initializer_for(batch)
                initializer_output = initializer(batch["context"], scene=batch["scene"],
                                                 target=batch["target"], device=self.device, **kwargs)
            return initializer_output

    def init_gaussians_with_window(self, batch, window, **kwargs) -> InitializerOutput:
        """Run the initializer in a sliding window over views, then combine the per-window Gaussians."""
        assert not self.source_initializers, (
            "Sliding-window initialization uses the default initializer only; it does not dispatch the "
            "per-source initializer for mixed datasets. Disable train/inference_window_size for mixed runs."
        )
        assert self.initializer.cfg.per_view, "Sliding window initialization only supports per-pixel initialization."
        b, v, _, h, w = batch["context"]["image"].shape
        assert window > 0

        window_indices = sliding_window_indices(v, window, 0)
        all_gaussians = []
        all_states = []
        all_pred_depths = []
        for indices in window_indices:

            start, end = indices
            view_indices = torch.arange(start, end, device=batch["context"]["image"].device).unsqueeze(0).expand(b, -1)
            curr_window_input = batch["context"].batchify_views(view_indices)

            initializer_output = self.initializer(curr_window_input, **kwargs)

            curr_gaussians = initializer_output.gaussians  # Gaussians object with tensors shape [B, G, D1, ...]
            curr_features = initializer_output.features  # [BV, C, H, W]

            all_gaussians.append(curr_gaussians)
            all_states.append(curr_features)
            if initializer_output.depths is not None:
                all_pred_depths.append(initializer_output.depths)

        # merge all gaussians
        def combine_gaussians_attribute(attr_name):
            all_attr = [getattr(g, attr_name) for g in all_gaussians[:-1]]
            last_g = all_gaussians[-1]
            last_g_attr = getattr(last_g, attr_name)
            # handle the overlapping in the last window
            if v % window != 0:
                x = v % window
                b, vhw, *d = last_g_attr.shape
                if self.initializer.cfg.per_pixel:
                    # per-pixel initialization
                    h_gaussians = h // self.initializer.cfg.latent_downsample
                    w_gaussians = w // self.initializer.cfg.latent_downsample
                else:
                    raise NotImplementedError
                last_g_attr = last_g_attr.view(b, window, h_gaussians, w_gaussians, *d)  # [B, V, H, W, ...]
                last_g_attr = last_g_attr[:, -x:, ...]  # [B, x, H, W, ...]
                last_g_attr = last_g_attr.view(b, -1, *d)  # [B, x*H*W, ...]
            all_attr.append(last_g_attr)
            return torch.cat(all_attr, dim=1)

        gaussians = Gaussians(
            means=combine_gaussians_attribute('means'),
            covariances=combine_gaussians_attribute('covariances'),
            harmonics=combine_gaussians_attribute('harmonics'),
            opacities=combine_gaussians_attribute('opacities'),
            scales=combine_gaussians_attribute('scales'),
            rotations=combine_gaussians_attribute('rotations'),
            rotations_unnorm=combine_gaussians_attribute('rotations_unnorm'),
        )

        # Collect condition features for the optimizer (only needed if optimizer is active)
        if self.scene_trainer_cfg.num_update_steps > 0:
            out = []
            is_ori_feature = True  # set by first window; True = [BV,C,H,W], False = [BVHW,C]
            for i in range(len(all_states)):
                # Assuming no overlap between windows
                curr = all_states[i]
                if curr.dim() == 4:
                    # [BV, C, H, W]
                    curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                    is_ori_feature = True
                elif curr.dim() == 2:
                    # [BVHW, C]
                    curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b,
                                     h=h // self.initializer.cfg.latent_downsample,
                                     w=w // self.initializer.cfg.latent_downsample,
                                     )
                    is_ori_feature = False
                else:
                    raise NotImplementedError

                # Only need to handle the overlaping in the last window
                if i == len(all_states) - 1 and v % window != 0:
                    # last window with overlap
                    x = v % window
                    curr = curr[:, -x:, ...]
                out.append(curr)

            # concat
            if is_ori_feature:
                concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                concat = rearrange(concat, "b v c h w -> (b v) c h w")
            else:
                concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                concat = rearrange(concat, "b v h w c -> (b v) c h w")

            condition_features = concat
        else:
            condition_features = None

        return InitializerOutput(gaussians=gaussians,
                                 features=condition_features,
                                 depths=all_pred_depths)

    def _insert_init_into_output(
            self,
            optimizer_output: OptimizerOutput,
            prev_output: InitializerOutput | OptimizerPreviousOutput,
    ) -> None:
        """Insert the init/resumed render + gaussians at position 0 of the optimizer_output lists,
        so every list is indexed by step: [0] is the init and [1..N] are the update steps.

        `prev_output` is an InitializerOutput for a fresh scene or an OptimizerPreviousOutput for a
        ckpt-buffer resume. Returns early for the ckpt-buffer rollout, which saves no per-step
        renders and so has nothing for position 0 to align with.
        """
        # Per-step renders are saved every step in training, and at the save_every iterations at test
        # time; the rollout turns both off, so no per-step entry exists for position 0 to sit before.
        saves_per_step = self.training or any(
            self.optimizer.save_every.enabled_tags[tag] for tag in ("target", "context")
        )
        if not saves_per_step:
            return

        if prev_output.context_render is None and prev_output.target_render is None:
            raise ValueError(
                "prev_output has no render attached, so the init cannot be placed at position 0 of "
                "the optimizer output. Render the init/resume views before optimizing."
            )

        # Insert the init gaussians together with the init render so that position k of gaussian_list
        # and of the render lists always refer to the same step.
        optimizer_output.gaussian_list.insert(0, prev_output.gaussians)
        for tag in ("context", "target"):
            render = prev_output.get_render(tag)
            if render is None:
                continue
            optimizer_output.get_render_list(tag).insert(0, render, detach_and_cpu=not self.training)
            index = prev_output.get_render_index(tag)
            if index is not None:
                optimizer_output.get_index_list(tag).insert(0, index)

    def init_gaussians_and_render(
            self, batch, visualization_dump,
            render_context: bool, render_target: bool, grad_enabled: bool,
            depth_mode: DepthRenderingMode | None = None,
            to_cpu: bool | None = None,
            **kwargs,
    ) -> InitializerOutput:
        """Run the initializer and optionally render its output to context/target views.

        Single entry point for "initialize then render" across training and the test/validation
        eval pipelines.

        By default renders go off the GPU when grads are disabled (test/eval, to save memory) and
        stay on the GPU with grads enabled (training, so the init-loss term can backward through
        them). Pass `to_cpu` to override -- the eval pipeline keeps the render on the GPU even with
        grads disabled because it feeds straight into the optimizer.

        Both the "initializer" timer (the init model) and the init render's "init_decoder" timer
        always record; "init_decoder" is kept separate from the "decoder" tag so that one stays the
        optimizer-phase decode time.
        """
        with self.benchmarker.time("initializer"):
            init_output: InitializerOutput = self.get_init_gaussians(batch, is_training=grad_enabled, **kwargs)

        self.render_init_views(batch, init_output, render_context, render_target, grad_enabled,
                               depth_mode, to_cpu=to_cpu)
        return init_output

    def render_init_views(
            self,
            batch,
            init_output: InitializerOutput | OptimizerPreviousOutput,
            render_context: bool,
            render_target: bool,
            grad_enabled: bool,
            depth_mode: DepthRenderingMode | None = None,
            to_cpu: bool | None = None,
    ) -> None:
        """Render context/target views into any output that exposes get_render/set_render.

        Mutates `init_output` in place; returns None. Skips a view if it's already rendered.
        Accepts either InitializerOutput (dataloader path) or OptimizerPreviousOutput
        (ckpt-buffer path, where renders are needed to populate the splice but no
        initializer model runs).
        """
        # to_cpu freezes the render off the GPU for evaluation/saving; with grads enabled we
        # keep it on GPU so the init-loss term can backward through it. The eval pipeline overrides
        # it (to_cpu=False) since the render feeds straight into the optimizer.
        if to_cpu is None:
            to_cpu = not grad_enabled

        # In optimizer-training mode (self.training=True, grad_enabled=False), render only a
        # random subset of views — matching what the optimizer does per step — to avoid the
        # cost of rendering all views (which may not fit in one batch).
        use_subset = self.training and not grad_enabled

        with torch.set_grad_enabled(grad_enabled):
            for input_str, should_render in (
                ("context", render_context),
                ("target", render_target),
            ):
                if not should_render or init_output.get_render(input_str) is not None:
                    continue
                if use_subset:
                    subset_views, index = self.batchify_views(batch, input_str, self.device)
                    rendered = self.decoder.forward_batch_subset(
                        init_output.gaussians.to(self.device),
                        subset_views,
                        iter_batch_size=self.scene_trainer_cfg.iter_batch_size,
                        depth_mode=depth_mode,
                    )
                    init_output.set_render_index(input_str, index)
                else:
                    views = batch[input_str]
                    h, w = views["image"].shape[-2:]
                    n_views = views["image"].shape[1]
                    # The initializer produces a single render per view set, so it is stored as one
                    # DecoderOutput per tag; the optimizer keeps one render per step in a list. The
                    # init render becomes position 0 of that list (see _insert_init_into_output).
                    with self.benchmarker.time("init_decoder", num_calls=n_views):
                        rendered = self.decoder.forward_batch(
                            init_output.gaussians.to(batch["target"]["image"].device),
                            batch, (h, w),
                            input_str=input_str,
                            to_cpu=to_cpu,
                            iter_batch_size=self.scene_trainer_cfg.iter_batch_size,
                            depth_mode=depth_mode,
                        )
                init_output.set_render(input_str, rendered)

    def test_postprocess_gaussians(self, batch, gaussians, visualization_dump) -> OptimizerOutput | None:
        """Run optional post-processing on the final Gaussians. Returns None if disabled."""
        postprocess_output = None
        if self.postprocess is not None:
            postprocess_output = self.postprocess.apply(
                batch,
                gaussians=gaussians,
                decoder=self.decoder,
                visualization_dump=visualization_dump,
                iter_batch_size=self.scene_trainer_cfg.iter_batch_size,
                batchify_fn=lambda b, input_str: self.batchify_views(
                    b, input_str, self.device,
                ),
            )

        return postprocess_output


def sliding_window_indices(N, x, y):
    """Return [start, end] pairs for a sliding window of size x with overlap y over N views.
    The last window is always [N-x, N] to cover any remainder."""
    indices = []
    start = 0
    while start + x < N:  # Ensure the last window is not processed here
        end = min(start + x, N)
        indices.append([start, end])
        start += (x - y)  # Move the start by the window size minus overlap

    # Append the last window [N-x, N]
    indices.append([N - x, N])

    return indices
