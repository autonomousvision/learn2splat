"""Public API: use learn2splat's learned optimizer in external 3DGS codebases.

Typical inria (graphdeco-inria/gaussian-splatting) integration — replace the
hand-written training loop with three lines::

    from learn2splat.experimental.api import Learn2Splat

    gaussians = GaussianModel(sh_degree)        # set up as usual (SfM init)
    scene = Scene(dataset, gaussians)
    learn2splat = Learn2Splat(checkpoint="hf://org/repo/model.ckpt", device="cuda")
    learn2splat.initialize(scene)                      # ingest scene + build optimizer
    learn2splat.optimize(scene)                        # learned optimization, written back in place
    scene.save(iteration)                        # proceed as normal

Full-replacement semantics: ``optimize`` overwrites ``scene.gaussians`` in
place and nulls the inria Adam optimizer + densification accumulators. If you
later want to resume inria Adam, call ``gaussians.training_setup(...)`` again.

For non-inria codebases use :meth:`Learn2Splat.initialize_from_ply` /
:meth:`Learn2Splat.initialize_from_tensors` + :meth:`Learn2Splat.export_ply`.

External SfM scenes carry no learn2splat encoder features, so checkpoints trained
with ``init_state_wo_features=False`` are coerced at construction (with a
warning): the feature-conditioned ``state_proj`` weights are dropped and the
optimizer state is initialized standard-normal.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import torch

from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from learn2splat.model.types import Gaussians

__all__ = ["Learn2Splat", "Learn2SplatError"]


class Learn2Splat:
    """Facade around the learned per-scene optimizer."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | torch.device = "cuda",
        num_refine: int | None = None,
        iter_batch_size: int | None = None,
        opt_batch_size: int | None = None,
        opt_batch_strategy: str | None = None,
        background_color: Sequence[float] | None = None,
        rasterize_mode: str | None = None,
        eps2d: float | None = None,
        strict_load: bool = True,
    ) -> None:
        if not checkpoint:
            raise Learn2SplatError(
                "Learn2Splat(checkpoint=...) is required (an 'hf://org/repo/file' "
                "reference or a local checkpoint path)."
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise Learn2SplatError(
                "Learn2Splat requires a CUDA device (the learned optimizer uses "
                "CUDA/KNN kernels). Pass device='cuda'."
            )
        # float32 only — the learned optimizer's CUDA/KNN kernels and the
        # gsplat rasterizer require it (and the checkpoint trained with it).
        self.dtype = torch.float32
        self.iter_batch_size = iter_batch_size
        self.opt_batch_size = opt_batch_size
        self.opt_batch_strategy = opt_batch_strategy

        from learn2splat.experimental.api.integration.config_bridge import (
            build_decoder,
            build_optimizer,
            build_optimizer_cfg,
            get_scene_trainer_scalar,
            load_optimizer_state,
            resolve_checkpoint_and_config,
        )

        local_ckpt, cfg_path = resolve_checkpoint_and_config(
            checkpoint, what="optimizer architecture"
        )

        opt_cfg, num_update_steps = build_optimizer_cfg(cfg_path)

        if not getattr(opt_cfg, "init_state_wo_features", False):
            warnings.warn(
                "this checkpoint was trained WITH encoder features "
                "(scene_trainer.scene_optimizer.init_state_wo_features=False). "
                "External SfM/inria scenes carry no learn2splat encoder features; "
                "proceeding with init_state_wo_features=True — the "
                "feature-conditioned state_proj weights are dropped and the "
                "initial optimizer state is set to a standard-normal random "
                "vector (init_state_type='random', init_state_scale=1.0)."
            )
            opt_cfg.init_state_wo_features = True
            opt_cfg.init_state_type = "random"
            opt_cfg.init_state_scale = 1.0

        optimizer = build_optimizer(opt_cfg)  # asserts cfg.name; nn.Module
        load_optimizer_state(
            optimizer, local_ckpt, init_state_wo_features=True, strict=strict_load
        )
        self.optimizer = optimizer.to(device=self.device, dtype=self.dtype).eval()

        from types import SimpleNamespace

        bg = list(background_color) if background_color is not None else [0.0, 0.0, 0.0]
        # Build the renderer the checkpoint trained with (gsplat by default;
        # NOT a hardcoded backend — see build_decoder). rasterize_mode / eps2d,
        # when given, override the checkpoint's decoder config.
        decoder_overrides = {
            k: v
            for k, v in (("rasterize_mode", rasterize_mode), ("eps2d", eps2d))
            if v is not None
        }
        self.decoder = build_decoder(
            cfg_path, SimpleNamespace(background_color=bg), decoder_overrides
        ).to(self.device)

        resolved = num_refine if num_refine is not None else num_update_steps
        if resolved is None:
            raise Learn2SplatError(
                "num_refine could not be determined: pass Learn2Splat(num_refine=...) "
                "or use a checkpoint whose config has "
                "scene_trainer.num_update_steps."
            )
        self.num_refine = int(resolved)

        # Render-batching size: user override, else the checkpoint's
        # scene_trainer.iter_batch_size (-1 = render all views per step).
        if self.iter_batch_size is None:
            self.iter_batch_size = int(
                get_scene_trainer_scalar(cfg_path, "iter_batch_size", -1)
            )

        # Per-step view minibatch — opt_batch_size views are fed to the
        # optimizer each step (the checkpoint's scene_trainer.opt_batch_size /
        # opt_batch_strategy, i.e. the regime it was trained with). -1 = all.
        if self.opt_batch_size is None:
            self.opt_batch_size = int(
                get_scene_trainer_scalar(cfg_path, "opt_batch_size", -1)
            )
        if self.opt_batch_strategy is None:
            self.opt_batch_strategy = str(
                get_scene_trainer_scalar(cfg_path, "opt_batch_strategy", "random")
            )
        if self.opt_batch_strategy not in ("random", "sequential", "fps"):
            raise Learn2SplatError(
                f"opt_batch_strategy={self.opt_batch_strategy!r} is not supported "
                f"by the API (supported: 'random', 'sequential', 'fps'). Pass "
                f"Learn2Splat(opt_batch_strategy='random')."
            )

        self._opt_cfg = opt_cfg
        # SH degree the checkpoint's Gaussians use — derived from the optimizer
        # cfg's init_sh_d (= (sh_degree + 1) ** 2, set by opt_cfg.update from the
        # initializer cfg). API consumers build/render Gaussians with this; it is
        # dictated by the checkpoint, not a free choice.
        self.sh_degree = int(round(opt_cfg.init_sh_d ** 0.5)) - 1
        self._initialized = False
        self._scene_ref = None
        self._context = None
        self._init_output = None
        self._refined: "Gaussians | None" = None

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def initialize(self, scene: object) -> "Learn2Splat":
        """Ingest an already-initialized inria-style scene.

        This does NOT run learn2splat's learned Initializer — the scene already has
        Gaussians (e.g. from SfM / inria ``create_from_pcd``).
        """
        from learn2splat.experimental.api.integration.inria_bridge import (
            batched_views_from_cameras,
            learn2splat_gaussians_from_inria_model,
        )
        from learn2splat.experimental.api.integration.scene_protocol import (
            assert_scene_protocol,
        )
        from learn2splat.scene_trainer.initializer.initializer import InitializerOutput

        assert_scene_protocol(scene)
        g = learn2splat_gaussians_from_inria_model(
            scene.gaussians, device=self.device, dtype=self.dtype
        )
        self._init_output = InitializerOutput(gaussians=g, features=None, depths=None)
        self._context = batched_views_from_cameras(
            list(scene.getTrainCameras()),
            scene_scale=float(scene.cameras_extent),
            device=self.device,
            dtype=self.dtype,
        )
        self._scene_ref = scene
        self._initialized = True
        return self

    def initialize_from_ply(
        self,
        ply_path: str,
        cameras: Sequence[object],
        *,
        sh_degree: int,
        scene_scale: float,
    ) -> "Learn2Splat":
        """Low-level ingest for non-inria codebases (no inria ``Scene``).

        ``cameras`` is a sequence of inria-``Camera``-like objects (``R``,
        ``T``, ``FoVx``, ``FoVy``, ``image_width``, ``image_height``,
        ``original_image``).
        """
        from learn2splat.experimental.api.integration.inria_bridge import (
            batched_views_from_cameras,
            learn2splat_gaussians_from_ply,
        )
        from learn2splat.scene_trainer.initializer.initializer import InitializerOutput

        g = learn2splat_gaussians_from_ply(
            ply_path, sh_degree=sh_degree, device=self.device, dtype=self.dtype
        )
        self._init_output = InitializerOutput(gaussians=g, features=None, depths=None)
        self._context = batched_views_from_cameras(
            list(cameras), scene_scale=scene_scale, device=self.device, dtype=self.dtype
        )
        self._scene_ref = None
        self._initialized = True
        return self

    def initialize_from_tensors(self, gaussians: object, batched_views: object) -> "Learn2Splat":
        """Low-level ingest from learn2splat-native objects (power users).

        ``gaussians``: a learn2splat ``Gaussians`` (batch=1, post-activation).
        ``batched_views``: a learn2splat ``BatchedViews`` or a dict accepted by
        ``BatchedViews.from_dict``.
        """
        from learn2splat.dataset.data_types import BatchedViews
        from learn2splat.model.types import Gaussians
        from learn2splat.scene_trainer.initializer.initializer import InitializerOutput

        if not isinstance(gaussians, Gaussians):
            raise Learn2SplatError(
                "initialize_from_tensors expects a learn2splat Gaussians instance "
                "(use initialize_from_ply for raw 3DGS PLY input)."
            )
        bv = (
            batched_views
            if isinstance(batched_views, BatchedViews)
            else BatchedViews.from_dict(batched_views)
        )
        self._init_output = InitializerOutput(
            gaussians=gaussians.to(device=self.device, dtype=self.dtype),
            features=None,
            depths=None,
        )
        self._context = bv
        self._scene_ref = None
        self._initialized = True
        return self

    # ------------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------------

    def _view_minibatch(self, views):
        """Sample the next per-step view minibatch from ``views``.

        Mirrors SceneTrainer's viewpoint-stack cycling: views are drawn
        ``opt_batch_size`` at a time and the stack is refilled once exhausted,
        so every view is seen before any repeats. ``random``/``sequential`` take
        the front of the (shuffled/ordered) stack; ``fps`` picks a
        farthest-point spread over the remaining views' camera positions.
        Returns ``views`` unchanged when ``opt_batch_size`` is <= 0 or already
        covers the whole scene.
        """
        v = views.image.shape[1]
        bs = self.opt_batch_size
        if bs <= 0 or bs >= v:
            return views

        views.reset_viewpoint_stack_if_needed(self.opt_batch_strategy, bs)
        stack = views.viewpoint_stack  # [B, V_stack]

        if self.opt_batch_strategy == "fps":
            from learn2splat.dataset.view_sampler.view_sampler_bounded_v2 import (
                farthest_point_sample,
            )

            b = stack.shape[0]
            arange = torch.arange(b, device=stack.device)[:, None]
            # FPS over the camera positions of the views still in the stack.
            positions = views.extrinsics[arange, stack][:, :, :3, 3]  # [B, V_stack, 3]
            local = farthest_point_sample(positions, bs, first_idx_strategy="random")
            idx = stack[arange, local]  # [B, bs]
            keep = ~(stack.unsqueeze(-1) == idx.unsqueeze(1)).any(-1)  # [B, V_stack]
            views.viewpoint_stack = stack[keep].view(b, -1)
        else:  # random / sequential — take the front of the stack
            idx = stack[:, :bs]
            views.viewpoint_stack = stack[:, bs:]
        return views.batchify_views(idx)

    def configure_adc(self, strategy: str) -> "Learn2Splat":
        """Set the ADC (densification) strategy for subsequent ``optimize`` runs.

        ``strategy`` is a raw refiner name: ``"none"`` (the default the checkpoints
        ship with) leaves the Gaussian set fixed; ``"default"``/``"edgs"``/``"mcmc"``/…
        enable clone/split/prune with the schedule rescaled to ``self.num_refine``
        (see ``build_refiner_cfg``). Call before ``optimize``/``optimize_iter`` — the
        ADC state is created in ``on_scene_start`` and gated on the refiner's flags.
        """
        from learn2splat.experimental.api.integration.config_bridge import build_refiner_cfg

        self.optimizer.cfg.refiner = build_refiner_cfg(strategy, self.num_refine)
        return self

    @torch.no_grad()
    def optimize(self, scene: object | None = None, *, optimizer=None):
        """Run the learned optimization.

        inria path: refined Gaussians are written back into ``scene.gaussians``
        in place and ``scene.gaussians`` is returned. Low-level path: the
        refined learn2splat ``Gaussians`` is returned (use :meth:`export_ply` to
        persist).

        ``optimizer`` swaps in a different learn2splat ``Optimizer`` (e.g. an Adam
        baseline) — running the *same* per-scene pipeline (init, view minibatch,
        step budget, renderer) with another update rule, i.e. a fair
        comparison. Defaults to the checkpoint's learned optimizer.
        """
        if scene is not None and scene is not self._scene_ref:
            self.initialize(scene)
        if not self._initialized:
            raise Learn2SplatError("call initialize(scene) before optimize().")

        opt = optimizer if optimizer is not None else self.optimizer

        from learn2splat.scene_trainer.optimizer.optimizer import (
            OptimizerInput,
            OptimizerOutput,
            OptimizerPreviousOutput,
        )

        inp = OptimizerInput(
            context=self._context,
            renderer=self.decoder,
            prev_output=self._init_output,
            iter_batch_size=self.iter_batch_size,
            target=self._context,
        )
        opt.validate_input(inp)
        opt.on_scene_start(inp)  # InitializerOutput -> OptimizerPreviousOutput (+ADC)
        if not isinstance(inp.prev_output, OptimizerPreviousOutput):
            raise Learn2SplatError(
                "optimizer.on_scene_start did not produce an "
                f"OptimizerPreviousOutput (got {type(inp.prev_output)})."
            )

        out = OptimizerOutput.empty(t=0)
        out.T = self.num_refine
        steps = range(self.num_refine)
        try:
            from tqdm import tqdm

            steps = tqdm(steps, desc=f"optimize[{type(opt).__name__}]")
        except Exception:
            pass
        for step in steps:
            # Feed the optimizer a fresh view minibatch each step (the regime it
            # was trained with); full_context/full_target stay the whole scene.
            batch = self._view_minibatch(self._context)
            inp.context = batch
            inp.target = batch
            out = opt(
                step, inp, out, full_context=self._context, full_target=self._context
            )
            out.t = (out.t or 0) + 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opt.on_scene_end()

        final = inp.prev_output.gaussians
        self._refined = final

        if self._scene_ref is not None:
            from learn2splat.experimental.api.integration.inria_bridge import (
                write_back_to_inria_model,
            )

            write_back_to_inria_model(self._scene_ref.gaussians, final)
            return self._scene_ref.gaussians
        return final

    def optimize_iter(self, *, optimizer=None):
        """Generator form of :meth:`optimize`: yields ``(step, gaussians)`` after
        each optimization step.

        Lets a caller drive the learned optimization one step at a time and
        render the Gaussians in between — used by ``demo.py``'s ``--with-gui``.
        ``on_scene_end()`` runs even if the caller closes the generator early
        (e.g. a GUI Reset), via the ``finally`` block.
        """
        if not self._initialized:
            raise Learn2SplatError("call initialize(...) before optimize_iter().")

        opt = optimizer if optimizer is not None else self.optimizer

        from learn2splat.scene_trainer.optimizer.optimizer import (
            OptimizerInput,
            OptimizerOutput,
            OptimizerPreviousOutput,
        )

        # torch's grad mode is thread-local, and a generator consumer (e.g.
        # gradio) may resume this generator on a *different* worker thread each
        # step — so a single ``with torch.no_grad()`` spanning the ``yield`` does
        # not reliably hold across steps. Enter no_grad *per step* and yield
        # outside it: the Adam baseline's in-place parameter updates require
        # no_grad ambient (calc_input_gradients re-enables grad internally just
        # for the gradient computation), so a step that resumes on a grad-enabled
        # thread would otherwise raise "a leaf Variable that requires grad is
        # being used in an in-place operation" and leak the autograd graph.
        inp = OptimizerInput(
            context=self._context,
            renderer=self.decoder,
            prev_output=self._init_output,
            iter_batch_size=self.iter_batch_size,
            target=self._context,
        )
        with torch.no_grad():
            opt.validate_input(inp)
            opt.on_scene_start(inp)  # InitializerOutput -> OptimizerPreviousOutput
        if not isinstance(inp.prev_output, OptimizerPreviousOutput):
            raise Learn2SplatError(
                "optimizer.on_scene_start did not produce an "
                f"OptimizerPreviousOutput (got {type(inp.prev_output)})."
            )

        out = OptimizerOutput.empty(t=0)
        out.T = self.num_refine
        try:
            for step in range(self.num_refine):
                with torch.no_grad():
                    # Fresh view minibatch each step (the regime the optimizer
                    # was trained with); full_context/target stay the whole scene.
                    batch = self._view_minibatch(self._context)
                    inp.context = batch
                    inp.target = batch
                    out = opt(
                        step, inp, out,
                        full_context=self._context, full_target=self._context,
                    )
                    out.t = (out.t or 0) + 1
                    g = inp.prev_output.gaussians
                yield step, g
        finally:
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                opt.on_scene_end()
            self._refined = inp.prev_output.gaussians

    def export_ply(self, path: str) -> None:
        """Write the most recently refined Gaussians to a 3DGS PLY."""
        if self._refined is None:
            raise Learn2SplatError("nothing to export — call optimize() first.")
        from pathlib import Path

        from learn2splat.model.ply_export import save_gaussian_ply

        save_gaussian_ply(self._refined, save_path=Path(path))
