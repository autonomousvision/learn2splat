from dataclasses import dataclass
from functools import partial
from typing import Literal, List, Optional

import torch
from torch import Tensor

from learn2splat.dataset.data_types import BatchedViews
from learn2splat.misc.io import FrequencyScheduler
from learn2splat.model.decoder.decoder import Decoder
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.initializer import InitializerCfg
from learn2splat.scene_trainer.optimizer.layer import AdamInputSmoothing
from learn2splat.scene_trainer.optimizer.optimizer import (
    OptimizerInput,
    OptimizerOutput,
    OptimizerCfg, NonlearnedOptimizer,
)
from learn2splat.scene_trainer.optimizer.optimizer_utils import (
    calc_input_gradients,
    squeeze_grad_dict,
    smooth_grads,
)


@dataclass
class AdamOptimizerCfg(OptimizerCfg):
    name: Literal["adam"]

    # adam params
    betas: List[float | int]  # Typically a list of two floats, e.g., [0.9, 0.999]
    eps: float
    weight_decay: float

    # Per-param learning rates and the means decay schedule come from the inherited
    # lr_scheduler cfg (lr_data / apply_scheduler + the expon scheduler params).

    def update(self, initializer_cfg: InitializerCfg, source_initializer_cfgs: "dict | None" = None):
        # Per-source overrides are ignored: Adam does not specialize per initializer.
        pass


class AdamOptimizer(NonlearnedOptimizer[AdamOptimizerCfg]):
    def __init__(
            self, cfg: AdamOptimizerCfg, save_every: Optional[FrequencyScheduler] = None
    ) -> None:
        super().__init__(cfg, save_every)

        self.smoothers = None
        self.means_lr_scheduler = None
        self._meta_bufs: dict = {}  # reused across steps: radii, visibility buffers

        # NOTE: AdamOptimizer is evaluation-only (3DGS baseline); not used during meta-training.

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        super()._on_scene_start_impl(optimizer_input)

        # assert scene batch size 1
        context = optimizer_input.context
        assert (
                context["extrinsics"].shape[0] == context["intrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported yet..."

        # instantiate Adam optimizers for each parameter type
        nr_gaussians = optimizer_input.prev_output.gaussians.means.shape[1]
        device = optimizer_input.prev_output.gaussians.means.device
        smoother_cls = partial(AdamInputSmoothing, beta1=self.cfg.betas[0], beta2=self.cfg.betas[1], eps=self.cfg.eps,
                               device=device)
        means_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.means.shape[1:])
        scales_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.scales.shape[1:])
        rotations_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.rotations.shape[1:])
        opacities_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.opacities.shape[1:])
        sh0s_smoother = smoother_cls(shape=optimizer_input.prev_output.gaussians.harmonics[..., :, :1].shape[1:])

        init_gaussians = optimizer_input.prev_output.gaussians
        if init_gaussians.harmonics.shape[-1] > 1:
            shNs_smoother = smoother_cls(shape=(init_gaussians.harmonics[..., :, 1:]).shape[1:])
        else:
            shNs_smoother = None

        self.smoothers = {
            "means": means_smoother,
            "scales": scales_smoother,
            "rotations": rotations_smoother,
            "opacities": opacities_smoother,
            "sh0s": sh0s_smoother,
            "shNs": shNs_smoother,
        }

        # get scene extent
        scene_scale = optimizer_input.context["scene_scale"]
        if scene_scale is None:
            scene_scale = torch.ones(1, 1, device=device)
        scene_scale = scene_scale[0].item()

        # Means LR follows the lr_scheduler's expon decay, scaled by scene extent. The schedule is
        # linear in its endpoints, so multiplying its output by scene_scale matches scaling both
        # endpoints (the original 3DGS recipe). Exposed as means_lr_scheduler so the shared ADC
        # path (MCMC noise injection) can read the same per-step means LR.
        self.means_lr_scheduler = lambda step: self.scheduler.get_lr(step, "means") * scene_scale

    def on_scene_end(self) -> None:
        super().on_scene_end()
        self.smoothers = None
        self.means_lr_scheduler = None
        self._meta_bufs.clear()

    def _forward_impl(
            self,
            i: int,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput,
            full_context: BatchedViews,
            full_target: BatchedViews,
            **kwargs
    ) -> OptimizerOutput:

        with self.benchmarker.time("iter"):
            # Unpack
            iter_context: BatchedViews = optimizer_input.context
            target: BatchedViews = optimizer_input.target
            renderer: Decoder = optimizer_input.renderer
            b, v, _, h, w = iter_context["image"].shape
            assert b == 1, "Batch size > 1 not supported for post-processing"

            # Log number of gaussians
            self.benchmarker.record("gaussians", optimizer_input.prev_output.gaussians.means.shape[1])

            # One optimization step
            res = self._apply_step(i, optimizer_input, optimizer_output, sh_degree=kwargs.get("sh_degree", None))
            gaussians: Gaussians = res[0]
            meta_for_adc: dict = res[1]
            updates: dict[str, Tensor] = res[2]
            grads_raw: dict[str, Tensor] = res[3]
            normalized_grads: dict[str, Tensor] = res[4]
            learning_rates: dict[str, float] = res[5]

            # Densification and Pruning
            if self.cfg.any_adc:
                # Apply ADC
                self.apply_adc(
                    i=i, v=v, h=h, w=w,
                    adc_state=optimizer_input.prev_output.state.adc_state,
                    gaussians=gaussians,
                    meta=meta_for_adc,
                    object_dict_to_adjust=self.smoothers,
                    renderer=optimizer_input.renderer,
                    full_context=full_context,  # full scene views for multi-view scoring, not the minibatch
                )
                # ADC changes N → cached buffers are invalid; re-make tensors as fresh leaves.
                buf_nr_gaussians = self._meta_bufs['N']
                actual_nr_gaussians = gaussians.means.shape[1]
                if buf_nr_gaussians != actual_nr_gaussians:
                    self._meta_bufs.clear()
                    # ADC rebuilt these tensors. Re-make them as clean leaves that require grad —
                    # the invariant the i==0 setup establishes and that the next step's in-place
                    # updates and autograd.grad both assume.
                    gaussians.means = gaussians.means.detach().requires_grad_(True)
                    gaussians.scales = gaussians.scales.detach().requires_grad_(True)
                    gaussians.rotations_unnorm = gaussians.rotations_unnorm.detach().requires_grad_(True)
                    gaussians.opacities = gaussians.opacities.detach().requires_grad_(True)
                    gaussians.harmonics = gaussians.harmonics.detach().requires_grad_(True)

        # Log stats — one entry per step (consumer indexes nr_nonzero_grad_log by step index)
        if grads_raw is not None:
            G = grads_raw["means"].shape[0]
            nonzero_grads = [(g.reshape(G, -1) != 0).any(dim=-1) for g in grads_raw.values() if g is not None]
            nonzero_grads = torch.stack(nonzero_grads)  # [num_params, G]
            nonzero_grads = nonzero_grads.any(dim=0)  # [G]
            self.benchmarker.record("nonzero_grads", nonzero_grads.sum().item())

        # Save updated gaussians (for next iteration)
        optimizer_input.prev_output.gaussians = gaussians

        # Info
        if self.save_every(i + 1, tag="info"):
            self._append_info(
                optimizer_output, gaussians,
                deltas={k: v.cpu() for k, v in updates.items() if v is not None},
                grads={k: v.cpu() for k, v in grads_raw.items() if v is not None},
                normalized_grads={k: v.cpu() for k, v in normalized_grads.items() if v is not None},
                learning_rates=learning_rates,
            )

        # Post-update context + target renders
        self._save_post_update_renders(
            i, optimizer_input, optimizer_output, gaussians,
            full_context, full_target,
        )

        # Optimizer output is being changed in place, but for clarity we return it
        return optimizer_output

    def _apply_step(
            self, i, optimizer_input: OptimizerInput, optimizer_output: OptimizerOutput, sh_degree: int | None = None
    ) -> tuple[Gaussians, dict | None, dict, dict[str, Tensor], dict[str, Tensor], dict[str, float]]:

        iter_context = optimizer_input.context
        b, v, _, h, w = iter_context["image"].shape
        renderer = optimizer_input.renderer
        gaussians = optimizer_input.prev_output.gaussians

        # if first iteration
        if i == 0:
            # assert gaussians stores activated values
            assert gaussians.stores_activated, "Gaussians must store activated values."
            # deactivate values in-place (avoids allocating new tensors)
            gaussians.scales.log_()  # [B, N, 3]
            gaussians.opacities.logit_()
            gaussians.stores_activated = False
            # enable requires_grad once — .grad buffers persist across steps,
            # so backward() reuses them instead of allocating new tensors each call
            gaussians.means.requires_grad_(True)
            gaussians.scales.requires_grad_(True)
            gaussians.rotations_unnorm.requires_grad_(True)
            gaussians.opacities.requires_grad_(True)
            gaussians.harmonics.requires_grad_(True)
        else:
            # assert gaussians does not store activated values
            assert not gaussians.stores_activated, "Gaussians must not store activated values."

        # learning rates — all from the lr_scheduler cfg (lr_data per-param values, with the
        # expon decay applied only where apply_scheduler is set, i.e. means). Means is additionally
        # scaled by scene extent via means_lr_scheduler.
        assert self.means_lr_scheduler is not None, "means_lr_scheduler is not initialized"
        means_lr = self.means_lr_scheduler(i)
        scales_lr = self.scheduler.get_lr(i, "scales")
        rotations_lr = self.scheduler.get_lr(i, "rotations")
        opacities_lr = self.scheduler.get_lr(i, "opacities")
        sh0s_lr = self.scheduler.get_lr(i, "sh0")
        shNs_lr = self.scheduler.get_lr(i, "shN")

        assert (
                iter_context["extrinsics"].shape[0] == iter_context["extrinsics"].shape[0] == 1
        ), "scene batch size > 1 not supported for yet..."

        # unpack gaussians
        means = gaussians.means  # [B, N, 3]
        rotations_unnorm = gaussians.rotations_unnorm  # [B, N, 4]
        scales_raw = gaussians.scales  # [B, N, 3]
        opacities_raw = gaussians.opacities  # [B, N]
        shs = gaussians.harmonics  # [B, N, 3, sh_d]

        with self.benchmarker.time("decoder"):
            loss, grads_raw, meta_for_adc = calc_input_gradients(
                iter_context,
                means,
                scales_raw,
                rotations_unnorm,
                opacities_raw,
                shs,
                renderer,
                need_2d_grads=self.cfg.need_2d_grads,
                chunk_size=self.cfg.input_gradients_chunk_size,
                any_adc=self.cfg.any_adc,
                sh_degree=sh_degree,
                meta_bufs=self._meta_bufs,
                loss_with_ssim=self.cfg.input_gradient_with_ssim_loss,
                loss_l2=self.cfg.input_gradient_l2_loss,
                opacity_reg_lambda=self.cfg.opacity_reg_lambda,
            )

        # get updates from adam optimizer
        grads_raw = squeeze_grad_dict(grads_raw)
        assert self.smoothers is not None, "Smoothers not initialized"
        grads_adam = smooth_grads(grads_raw, self.smoothers)

        # update the gaussians parameters
        # Batch delta computation for contiguous params with _foreach_mul to reduce kernel launches.
        # no_refine flags are handled by excluding the param from the batch (delta stays None).
        _grad_lr_pairs = [
            (grads_adam["means"], -means_lr, self.cfg.freeze_mean),
            (grads_adam["scales"], -scales_lr, self.cfg.freeze_scale),
            (grads_adam["rotations"], -rotations_lr, self.cfg.freeze_rotation),
            (grads_adam["opacities"], -opacities_lr, self.cfg.freeze_opacity),
        ]
        _active_grads = [g for g, lr, skip in _grad_lr_pairs if not skip]
        _active_lrs = [lr for g, lr, skip in _grad_lr_pairs if not skip]
        _active_deltas = torch._foreach_mul(_active_grads, _active_lrs) if _active_grads else []

        _delta_iter = iter(_active_deltas)
        delta_means = next(_delta_iter) if not self.cfg.freeze_mean else None
        delta_scales_raw = next(_delta_iter) if not self.cfg.freeze_scale else None
        delta_rotations_unnorm = next(_delta_iter) if not self.cfg.freeze_rotation else None
        delta_opacities_raw = next(_delta_iter) if not self.cfg.freeze_opacity else None

        # SH deltas stay separate (non-contiguous slice views)
        delta_sh0s = None if self.cfg.freeze_sh0 else -sh0s_lr * grads_adam["sh0s"]
        delta_shNs = None
        if grads_adam["shNs"] is not None and not self.cfg.freeze_shN:
            delta_shNs = -shNs_lr * grads_adam["shNs"]

        # step — batch contiguous params with _foreach_add_ to reduce kernel launches;
        # SH slice views are non-contiguous so they stay separate
        _params = [means, scales_raw, rotations_unnorm, opacities_raw]
        _deltas = [delta_means, delta_scales_raw, delta_rotations_unnorm, delta_opacities_raw]
        _active = [(p, d) for p, d in zip(_params, _deltas) if d is not None]
        if _active:
            torch._foreach_add_([p for p, d in _active], [d for p, d in _active])
        self.safe_inplace_update(delta_sh0s, shs[..., 0:1])
        self.safe_inplace_update(delta_shNs, shs[..., 1:])

        # assign (means/scales/rotations/harmonics are the same objects; in-place ops above
        # already updated their storage. opacities_raw is a view — do NOT reassign
        # gaussians.opacities here, as that would replace the persistent leaf with a non-leaf
        # view and break retain_grad() on subsequent steps.)
        gaussians.means = means
        gaussians.scales = scales_raw
        gaussians.rotations_unnorm = rotations_unnorm
        gaussians.harmonics = shs

        # group updates
        updates = {
            "means": delta_means,
            "scales": delta_scales_raw,
            "rotations": delta_rotations_unnorm,
            "opacities": delta_opacities_raw,
            "sh0s": delta_sh0s,
            "shNs": delta_shNs,
        }

        learning_rates = {
            "means": means_lr,
            "scales": scales_lr,
            "rotations": rotations_lr,
            "opacities": opacities_lr,
            "sh0s": sh0s_lr,
            "shNs": shNs_lr,
        }

        return gaussians, meta_for_adc, updates, grads_raw, grads_adam, learning_rates

    @staticmethod
    def safe_inplace_update(delta_means: Tensor | None, means: Tensor):
        if delta_means is not None:
            means += delta_means
