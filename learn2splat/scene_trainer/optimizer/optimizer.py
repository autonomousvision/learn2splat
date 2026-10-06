"""Base classes and shared types for the scene optimizers.

Defines the optimizer interface that `SceneTrainer` calls: `Optimizer` (the abstract base, with the
per-iteration timing/benchmarking and the ADC calls) and `LearnedOptimizer` (adds the network-based
machinery). Also defines the pipeline data types passed in and out each iteration: `OptimizerInput`,
`OptimizerOutput`, `OptimizerState`, `OptimizerPreviousOutput`, and the `OptimizerCfg` base config. The
learned optimizer used in the paper is `KnnBasedOptimizer` (optimizer_knn_based.py); the 3DGS Adam
baseline is `AdamOptimizer` (optimizer_adam.py).
"""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, Generic, Optional, TYPE_CHECKING, Any
import torch
from torch import nn
from torch import Tensor
from learn2splat.misc.benchmarker import Benchmarker
from learn2splat.misc.io import FrequencyScheduler
from learn2splat.dataset.data_types import BatchedViews
from learn2splat.model.decoder import Decoder
from learn2splat.model.decoder.decoder import DecoderOutput, DepthRenderingMode
from learn2splat.model.types import Gaussians
# Per-strategy configs imported from their leaf submodules (not the adc package) to avoid the
# adc/__init__ <-> optimizer import cycle that the lazy imports below guard against.
from learn2splat.scene_trainer.adc.vanilla import VanillaStrategyCfg
from learn2splat.scene_trainer.adc.mcmc import McmcStrategyCfg
from learn2splat.scene_trainer.adc.fastgs import FastGSStrategyCfg

# Discriminated by `name`; mirrors adc.StrategyCfg (kept inline here to dodge the package import).
StrategyCfg = VanillaStrategyCfg | McmcStrategyCfg | FastGSStrategyCfg
from learn2splat.scene_trainer.initializer.initializer import InitializerOutput
from learn2splat.scene_trainer.initializer import InitializerCfg
from learn2splat.misc.detaching_cpu_list import DetachingCPUList
from learn2splat.scene_trainer.optimizer.lr_scheduler import LrSchedulerCfgType, get_scheduler

if TYPE_CHECKING:
    from learn2splat.scene_trainer.adc.vanilla import VanillaStrategyState
    from learn2splat.scene_trainer.adc.mcmc import McmcStrategyState


@dataclass
class OptimizerState:
    adc_state: Any = None  # VanillaStrategyState | McmcStrategyState | None


@dataclass
class OptimizerPreviousOutput:
    gaussians: Gaussians
    state: OptimizerState | None = None
    # Optional pre-computed renders of `gaussians`, used by the splice in
    # SceneTrainer.optimize_scene to put init/resumed renders at position 0
    # of optimizer_output's render lists. Mirrors the same fields on InitializerOutput
    # so the splice can treat both prev_output types uniformly.
    target_render: DecoderOutput | None = None
    context_render: DecoderOutput | None = None

    # View indices used when rendering a subset (training); None means all views were rendered.
    target_render_index: "torch.Tensor | None" = None
    context_render_index: "torch.Tensor | None" = None

    def get_render(self, which: str) -> "DecoderOutput | None":
        if which == "target":
            return self.target_render
        elif which == "context":
            return self.context_render
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render(self, which: str, value: "DecoderOutput") -> None:
        if which == "target":
            self.target_render = value
        elif which == "context":
            self.context_render = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def get_render_index(self, which: str) -> "torch.Tensor | None":
        if which == "target":
            return self.target_render_index
        elif which == "context":
            return self.context_render_index
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render_index(self, which: str, value: "torch.Tensor | None") -> None:
        if which == "target":
            self.target_render_index = value
        elif which == "context":
            self.context_render_index = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")


@dataclass
class OptimizerInput:
    context: BatchedViews
    renderer: Decoder
    prev_output: InitializerOutput | OptimizerPreviousOutput
    iter_batch_size: int | None
    target: BatchedViews | None = None
    context_remain: dict | None = None
    debug_dict: dict | None = None
    depth_mode: DepthRenderingMode | None = None

    @property
    def device(self) -> torch.device:
        return self.context["image"].device


@dataclass
class OptimizerOutput:
    gaussian_list: DetachingCPUList[Gaussians]
    t: int | None = None
    T: int | None = None
    last_prev_output: OptimizerPreviousOutput | None = None
    target_render_list: DetachingCPUList[DecoderOutput] | None = None
    context_render_list: DetachingCPUList[DecoderOutput] | None = None
    info: dict | None = None
    context_index_list: list[int] = field(default_factory=list)
    target_index_list: list[int] = field(default_factory=list)

    def get_render_list(self, which: str) -> DetachingCPUList[DecoderOutput] | None:
        if which == "target":
            return self.target_render_list
        elif which == "context":
            return self.context_render_list
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def get_index_list(self, which: str):
        if which == "target":
            return self.target_index_list
        elif which == "context":
            return self.context_index_list
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    @classmethod
    def empty(cls, t=None) -> "OptimizerOutput":
        new = cls(gaussian_list=DetachingCPUList(), t=t)
        new.target_render_list = DetachingCPUList()
        new.context_render_list = DetachingCPUList()
        # info is a dict of lists of dicts, should all be stored in cpu
        new.info: dict[str, list[dict[str, Tensor]]] = {}
        return new


@dataclass
class OptimizerCfg:
    
    # subset optimization flags
    freeze_mean: bool
    freeze_scale: bool
    freeze_rotation: bool
    freeze_opacity: bool
    freeze_sh0: bool
    freeze_shN: bool

    # lr scheduler
    lr_scheduler: LrSchedulerCfgType
    
    refiner: StrategyCfg

    # gradients
    input_gradients_chunk_size: int | None  # if None, use full image

    # Photometric loss driving the gradients: L1 alone, or L1 + 0.2 SSIM when true.
    input_gradient_with_ssim_loss: bool
    # Use squared error (MSE) instead of L1 for the photometric term.
    input_gradient_l2_loss: bool

    # L1 opacity regularization from 3DGS-MCMC (arXiv:2404.09591); 0.0 to disable
    opacity_reg_lambda: float

    def update(self, initializer_cfg: InitializerCfg, source_initializer_cfgs: "dict | None" = None):
        # source_initializer_cfgs (mixed-dataset per-source overrides) is accepted for a uniform call
        # signature; optimizers that don't specialize per initializer ignore it.
        pass

    @property
    def any_adc(self) -> bool:
        return self.refiner.do_densify or self.refiner.do_prune or self.refiner.do_opacity_reset

    @property
    def need_2d_grads(self) -> bool:
        return self.refiner.do_densify

    @property
    def optimize_all(self):
        # All the freeze_* are False
        return not any([
            self.freeze_mean,
            self.freeze_scale,
            self.freeze_rotation,
            self.freeze_opacity,
            self.freeze_sh0,
            self.freeze_shN,
        ])


T = TypeVar("T")


class Optimizer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T, save_every: Optional[FrequencyScheduler] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_every = save_every
        # Whether the renders stored at eval time keep their depth map. SceneTrainer sets it
        # from the depth/video output flags; see _save_post_update_renders.
        self.keep_render_depth = True

        # Per-iteration timing. The optimizer brackets each iteration with benchmarker.time("iter") and
        # the render inside it with benchmarker.time("decoder"); these use CUDA events with a deferred
        # sync, so the loop is not stalled every step. iter/decoder/optimizer_time_log read back from it.
        self.benchmarker = Benchmarker()
        # scene_start_event_start/end bracket optimizer.on_scene_start() (KNN, Adam init).
        # Read after the post-loop cuda.synchronize() in scene_trainer.optimize_scene.
        self.scene_start_event_start = torch.cuda.Event(enable_timing=True)
        self.scene_start_event_end = torch.cuda.Event(enable_timing=True)

        # LR scheduler
        self.scheduler = get_scheduler(self.cfg.lr_scheduler)

    def forward(self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, **kwargs) -> OptimizerOutput:
        return self._forward_impl(i, optimizer_input, optimizer_output, **kwargs)

    @property
    def iter_time_log(self) -> list[float]:
        """Total ms per optimization iteration (from benchmarker.time("iter"))."""
        return self.benchmarker.execution_times["iter"]

    @property
    def decoder_time_log(self) -> list[float]:
        """Ms spent rendering-for-gradients per iteration (from benchmarker.time("decoder"))."""
        return self.benchmarker.execution_times["decoder"]

    @property
    def optimizer_time_log(self) -> list[float]:
        """Ms spent in the update step per iteration (iter_time - decoder_time)."""
        return [it - dec for it, dec in zip(self.iter_time_log, self.decoder_time_log)]

    # Per-iteration stats, stored in `self.benchmarker` next to the timings (one entry per step).
    # Counts are ints, max-radii/grads are floats, so the element type is float | int.
    @property
    def nr_gaussians_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["gaussians"]

    @property
    def nr_nonzero_grad_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["nonzero_grads"]

    @property
    def nr_cloned_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["cloned"]

    @property
    def nr_splitted_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["splitted"]

    @property
    def nr_pruned_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["pruned"]

    @property
    def radii_max_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["radii_max"]

    @property
    def grads_max_log(self) -> list[float | int]:
        return self.benchmarker.execution_times["grads_max"]

    @property
    def scene_start_ms(self) -> float:
        """Ms for on_scene_start (Adam/state init, preprocessing); one value per scene."""
        times = self.benchmarker.execution_times["scene_start"]
        return times[-1] if times else 0.0

    @staticmethod
    def _append_info(optimizer_output: "OptimizerOutput", gaussians, **info_lists) -> None:
        """Append per-iteration debug info at test time (shared by all optimizers).

        Saves `gaussians`, then appends each keyword value (e.g. deltas / grads /
        normalized_grads / states_norms / learning_rates) to optimizer_output.info[key].
        Each optimizer prepares its own already-CPU values before calling.
        """
        optimizer_output.gaussian_list.append(gaussians, detach_and_cpu=True)
        assert optimizer_output.info is not None
        for key, value in info_lists.items():
            optimizer_output.info.setdefault(key, []).append(value)

    def on_scene_start(self, optimizer_input: OptimizerInput) -> None:
        self.benchmarker.clear_history()  # isolate this scene's per-iteration timing
        self._on_scene_start_impl(optimizer_input)

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        init_output = optimizer_input.prev_output
        assert isinstance(init_output, InitializerOutput), \
            (f"base Optimizer class on_scene_start just convert the InitializerOutput to OptimizerPreviousOutput, "
             f"without handling the state. "
             f"It also initialize a new state for density control."
             f"Got type {type(init_output)}")

        # Converting the initializer output to optimizer previous output
        optimizer_prev_output = OptimizerPreviousOutput(
            gaussians=init_output.gaussians.clone(),
            state=None,
        )
        optimizer_input.prev_output = optimizer_prev_output

        if self.cfg.any_adc:
            self.reset_logs()
            optimizer_prev_output.state = OptimizerState()  # init to empty state
            self.initialize_adc_state(self.cfg, optimizer_input)

    def on_scene_end(self) -> None:
        pass

    def reset_logs(self):
        # All per-scene logs (timings, stats, scene_start) live in self.benchmarker; one clear resets them.
        self.benchmarker.clear_history()

    @staticmethod
    def initialize_adc_state(cfg: OptimizerCfg, optimizer_input: OptimizerInput) -> None:
        # Lazy import to avoid circular dependency
        from learn2splat.scene_trainer.adc import init_strategy_state

        # get number of points
        init_gaussians = optimizer_input.prev_output.gaussians
        nr_points = init_gaussians.means.shape[1]
        # get scene extent
        context = optimizer_input.context
        target = optimizer_input.target
        assert (
                context["extrinsics"].shape[0] == context["intrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported yet..."

        scene_scale = context["scene_scale"][0].item()
        # Initialize ADC state
        optimizer_input.prev_output.state.adc_state = init_strategy_state(
            cfg=cfg.refiner,
            nr_points=nr_points,
            device=init_gaussians.means.device,
            scene_extent=scene_scale
        )
        print("Initialized ADC state with", nr_points, "points and scene extent", scene_scale)
        
    def _forward_impl(self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, **kwargs) -> OptimizerOutput:
        raise NotImplementedError()

    def validate_input(self, optimizer_input: OptimizerInput) -> None:
        pass

    def _save_post_update_renders(
            self,
            i: int,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput,
            updated_gaussians: Gaussians,
            full_context: BatchedViews,
            full_target: BatchedViews,
    ) -> None:
        """Render and append post-update context+target views into optimizer_output's render lists.

        When and what is rendered:
        - Training: render every iteration so per-step renders can feed the meta-loss.
                    If `opt_batch_size < V` (subset sampling), render only that per-iter subset
                    to match the views the optimization step saw.
        - Test/eval (self.training=False): render only when save_every fires for this tag,
                    and ALWAYS use the full V views — even if opt_batch_size < V drove the
                    optimization step on a subset. This is the invariant the test-time
                    save/score paths in MetaTrainer._eval_and_save rely on: render_list[k]
                    is always [1, V, ...], so it lines up 1:1 with batch[tag]["index"][0]
                    (the full-V frame indices) used for filename labelling and metric GT.
        """
        for tag, full, iter_views in (
            ("context", full_context, optimizer_input.context),
            ("target", full_target, optimizer_input.target),
        ):
            if not (self.training or self.save_every(i + 1, tag=tag)):
                continue
            index_list = optimizer_output.get_index_list(tag)
            # Subset rendering is training-only — at test time we always re-render the full V
            # views so downstream save/score paths see uniform [1, V, ...] tensors.
            subset = iter_views if (index_list and self.training) else full
            render_output = optimizer_input.renderer.forward_batch_subset(
                updated_gaussians,
                subset,
                iter_batch_size=optimizer_input.iter_batch_size,
                depth_mode=optimizer_input.depth_mode,
            )
            if not self.training:
                # At eval time one entry per saved step is held on CPU until the scene is
                # scored, so keep only what the save/score path reads: the colors, and the
                # depth map when a depth or video output is enabled. The screen-space fields
                # (means2d, radii, visibility_filter, accumulated_alpha) are read from the
                # gradient render's meta during optimization, not from this list.
                render_output = DecoderOutput(
                    render_output.color,
                    depth=render_output.depth if self.keep_render_depth else None,
                )
            optimizer_output.get_render_list(tag).append(
                render_output,
                detach_and_cpu=not self.training,
            )

    @torch.no_grad()
    def apply_adc(self, i, v, h, w, adc_state, gaussians, meta, object_dict_to_adjust=None,
                  renderer=None, full_context=None):
        """
        Apply adaptive density control (ADC) based on 2D gradient norms.
        Implements densification and pruning of Gaussians during optimization, as in vanilla 3DGS.

        Args:
            gaussians: Gaussians to be densified/pruned in place.
            h: Height of the rendered images.
            i: Current optimization iteration.
            v: Number of views.
            meta: Metadata dict from the rendering, including visibility masks and radii.
            w: Width of the rendered images.
            object_dict_to_adjust: Dict of object to adjust after pruning and densification, if needed.
            renderer: Decoder used for the run; the FastGS strategy needs its render_metric_counts.
            context: Context views (GT + cameras) the FastGS strategy scores against.
        """
        # Lazy import to avoid circular dependency
        from learn2splat.scene_trainer.adc import post_backward
        from learn2splat.scene_trainer.adc.base import is_densify_step, is_final_prune_step

        cfg = self.cfg.refiner
        visibility_mask = meta["visibility_filter"]  # [B, V, N]
        radii_2d = meta["radii"].float()  # [B, V, N, 2]
        means2d_grads = meta["means_2d_grads"]  # [B, V, N, 2] or None
        means2d_abs_grads = meta.get("means_2d_abs_grads")  # FastGS Abs-GS split signal, else None

        # FastGS multi-view scores at densification steps: render a few context views, flag
        # high-error pixels, and ask the FastGS rasterizer which Gaussians to grow/prune. Needs the
        # FastGS decoder (render_metric_counts); other backends leave the scores None -> the
        # strategy's deterministic fallback (no importance filter, full pruning).
        importance_score = pruning_score = None
        # Sample num_score_cams from the FULL scene (matching FastGS's sampling_cameras over the whole
        # train set), NOT the per-step optimization minibatch — otherwise the multi-view importance/
        # pruning filter collapses to opt_batch_size views (1 at batch size 1), starving densification.
        if cfg.name == "fastgs" and full_context is not None and hasattr(renderer, "render_metric_counts"):
            densify = is_densify_step(cfg, i)
            # Phase 1 (densify) needs both scores; phase 2 (final prune) needs only pruning_score.
            if densify or is_final_prune_step(cfg, i):
                from learn2splat.scene_trainer.adc.fastgs import render_fastgs_scores
                importance_score, pruning_score = render_fastgs_scores(
                    renderer, full_context, gaussians, (h, w),
                    num_score_cams=cfg.num_score_cams, loss_thresh=cfg.loss_thresh,
                    densify=densify,
                )

        # means lr for MCMC noise injection
        # check if optimizer has means_lr_scheduler
        if hasattr(self, "means_lr_scheduler"):
            assert self.means_lr_scheduler is not None, "means_lr_scheduler is None."
            lr = self.means_lr_scheduler(i)
        else:
            # Use fallback_means_lr from the refiner config so noise magnitude matches the
            # original paper (means_lr * noise_lr ≈ 1.6e-4 * 5e5 = 80 covariance-units).
            lr = cfg.fallback_means_lr

        # Post-backward (ADC)
        nr_cloned, nr_splitted, nr_pruned, max_radii, max_grad2d = post_backward(
            cfg=cfg,
            step=i,
            gaussians=gaussians,
            adc_state=adc_state,
            smoothers=object_dict_to_adjust,
            radii_2d=radii_2d,  # [V, N]
            means2d_grads=means2d_grads,  # [V, N, 2]
            visibility_mask=visibility_mask,  # [V, N]
            iter_batch_size=v,
            w=w,
            h=h,
            lr=lr,
            means2d_abs_grads=means2d_abs_grads,  # FastGS Abs-GS split signal (None for other backends)
            importance_score=importance_score,    # FastGS multi-view scores (None off densify steps)
            pruning_score=pruning_score,
        )
        
        self.benchmarker.record("cloned", nr_cloned)
        self.benchmarker.record("splitted", nr_splitted)
        self.benchmarker.record("pruned", nr_pruned)
        self.benchmarker.record("radii_max", max_radii if max_radii is not None else 0.0)
        self.benchmarker.record("grads_max", max_grad2d if max_grad2d is not None else 0.0)

class LearnedOptimizer(Optimizer[T], ABC):
    @property
    def strategy(self) -> str:
        return "learned"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class NonlearnedOptimizer(Optimizer[T], ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # nn.Module.__init__ sets training=True (a plain attribute, not via
        # train()); a non-learned optimizer has no trainable parameters, so pin
        # it to eval at construction.
        self.eval()

    @property
    def strategy(self) -> str:
        return "nonlearned"

    def train(self, mode: bool = True):
        # train mode is meaningless here, and `self.training` gates
        # meta-training-only code paths (e.g. _save_post_update_renders
        # retaining full-scene renders on GPU). Pin to eval, even under a
        # generic `module.train()` recursion.
        return super().train(False)
