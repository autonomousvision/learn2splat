"""Public API: learn2splat's learned feed-forward initializer (resplat), standalone.

Produces an initial learn2splat ``Gaussians`` from posed context views via the learned
MVS initializer — without going through Hydra. Pairs with :class:`Learn2Splat`
(the learned *optimizer* facade), mirroring the codebase's Initializer/Optimizer
split::

    from learn2splat.experimental.api import Learn2Splat, Learn2SplatInitializer

    init = Learn2SplatInitializer(checkpoint="hf://org/repo/resplat_init/.../model.ckpt")
    g = init.initialize(batched_views, num_context_views=8)   # -> Gaussians

    learn2splat = Learn2Splat(checkpoint="hf://org/repo/dense/.../model.ckpt")  # any optimizer ckpt
    learn2splat.initialize_from_tensors(g, batched_views)
    refined = learn2splat.optimize()

The initializer must come from a checkpoint trained with a learned
``scene_initializer`` (``resplat_v1`` / ``resplat_v2``) — e.g. the dedicated
``resplat_init`` checkpoint. The dense/colmap optimizer checkpoints carry no
initializer weights (``build_initializer_cfg`` raises an actionable error).

Only the ``Gaussians`` are returned, not the initializer's ``features``: the
:class:`Learn2Splat` optimizer facade runs with ``init_state_wo_features=True`` and
drops the feature-conditioned ``state_proj`` weights, so the features are never
consumed downstream — returning the bare Gaussians keeps the contract honest.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from learn2splat.model.types import Gaussians

__all__ = ["Learn2SplatInitializer"]

# Training-matched defaults for the feed-forward init, single-sourced so `initialize`
# and the `estimate_num_gaussians` predictor of its output can't drift apart.
_DEFAULT_NUM_CONTEXT_VIEWS = 8
_DEFAULT_MAX_IMAGE_SIDE = 512


def _resplat_geometry(init_cfg) -> tuple[int, int, int]:
    """``(crop_multiple, latent_downsample, gaussians_per_latent_cell)`` for a resplat cfg.

    ``crop_multiple``: the MVS input H/W must be a multiple of this, satisfying two
    constraints via their lcm — the data shim's latent-grid size
    (``shim_patch_size * downscale_factor``) and the depth predictor's window attention
    (it downsamples to ``1/lowest_feature_resolution`` and splits up to 8 ways there, so
    the input must also be divisible by ``lowest_feature_resolution * 8``; see
    ``mv_unimatch`` ``attn_splits``). Training images were sized so the shim's smaller
    multiple sufficed; arbitrary COLMAP scenes (e.g. garden at 840x1297) need the stricter one.

    The initializer then emits one latent cell per ``latent_downsample`` x
    ``latent_downsample`` image block, times ``gaussians_per_latent_cell`` Gaussians per
    cell — the basis for :meth:`Learn2SplatInitializer.estimate_num_gaussians`.
    """
    ps, df = init_cfg.shim_patch_size, init_cfg.downscale_factor
    shim_mult = (ps if isinstance(ps, int) else max(ps)) * df
    lfr = int(getattr(init_cfg, "lowest_feature_resolution", 8))
    crop_multiple = math.lcm(shim_mult, lfr * 8)
    per_cell = int(getattr(init_cfg, "gaussians_per_pixel", 1)) * int(
        getattr(init_cfg, "init_gaussian_multiple", 1)
    )
    return crop_multiple, int(init_cfg.latent_downsample), per_cell


@lru_cache(maxsize=4)
def _resplat_geometry_for_checkpoint(checkpoint: str) -> tuple[int, int, int]:
    """``_resplat_geometry`` for a checkpoint, from its *config only* (no weights
    download / model build). Memoized: the geometry is fixed per checkpoint, while
    :meth:`Learn2SplatInitializer.estimate_num_gaussians` is called live as UI settings change —
    so the (uncached) config compose runs once per checkpoint, not per keystroke.
    """
    from learn2splat.config import _find_config_for_checkpoint, resplat_official_config_path
    from learn2splat.experimental.api.integration.config_bridge import build_initializer_cfg
    from learn2splat.misc.hf_ckpt import hf_sibling_config, maybe_resolve_hf_ref

    # Prefer the sibling config (hf:// -> just config.yaml); only resolve the full
    # checkpoint (which would download the weights) if that lookup fails.
    local_ckpt = maybe_resolve_hf_ref(checkpoint)
    cfg_path = (
        hf_sibling_config(checkpoint)
        or _find_config_for_checkpoint(local_ckpt)
        or resplat_official_config_path(local_ckpt)
    )
    if cfg_path is None:
        raise Learn2SplatError(
            f"no config.yaml found for checkpoint {checkpoint!r}; cannot estimate "
            f"the Gaussian count."
        )
    return _resplat_geometry(build_initializer_cfg(cfg_path))


class Learn2SplatInitializer:
    """Facade around the learned feed-forward (resplat) initializer."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str | torch.device = "cuda",
        strict_load: bool = True,
    ) -> None:
        if not checkpoint:
            raise Learn2SplatError(
                "Learn2SplatInitializer(checkpoint=...) is required (an "
                "'hf://org/repo/file' reference or a local checkpoint path)."
            )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise Learn2SplatError(
                "Learn2SplatInitializer requires a CUDA device (the learned initializer's "
                "MVS / DINOv2 kernels and autocast paths are CUDA-bound). Pass "
                "device='cuda'."
            )
        # float32 only — same constraint as the gsplat rasterizer / the checkpoint.
        self.dtype = torch.float32

        from learn2splat.experimental.api.integration.config_bridge import (
            build_initializer,
            build_initializer_cfg,
            load_initializer_state,
            resolve_checkpoint_and_config,
        )

        local_ckpt, cfg_path = resolve_checkpoint_and_config(
            checkpoint, what="initializer architecture"
        )

        init_cfg = build_initializer_cfg(cfg_path)  # Learn2SplatError if not a learned init
        initializer = build_initializer(init_cfg)
        # Official resplat .pth files carry pt.mv_blocks.* keys that our model
        # doesn't implement (init_pt_with_mv_attn=False); skip them gracefully.
        import re as _re
        _strict = strict_load and not (
            local_ckpt
            and local_ckpt.endswith(".pth")
            and bool(_re.search(r"resplat-(small|base|large)-", local_ckpt))
        )
        load_initializer_state(initializer, local_ckpt, strict=_strict)
        self.initializer = initializer.to(device=self.device, dtype=self.dtype).eval()

        self._init_cfg = init_cfg
        # SH degree the initializer's Gaussians use — must match the Learn2Splat optimizer
        # the result is fed to (both are 3 for the shipped checkpoints).
        self.sh_degree = int(init_cfg.gaussian_adapter.sh_degree)
        # Input H/W is center-cropped to a multiple of this (see _resplat_geometry).
        self.patch_multiple, _, _ = _resplat_geometry(init_cfg)

    @torch.no_grad()
    def initialize(
        self,
        batched_views: object,
        *,
        num_context_views: int = _DEFAULT_NUM_CONTEXT_VIEWS,
        strategy: str = "fps",
        first_idx_strategy: str = "random",
        near: float = 0.01,
        far: float = 200.0,
        max_image_side: int = _DEFAULT_MAX_IMAGE_SIDE,
    ) -> "Gaussians":
        """Run the learned initializer on a spread-out subset of ``batched_views``.

        ``batched_views``: a learn2splat ``BatchedViews`` (or a dict accepted by
        ``BatchedViews.from_dict``) with extrinsics / intrinsics / image.

        ``num_context_views``: how many input views to condition the feed-forward
        initializer on (it was trained on ~8; the Gaussian count and the MVS cost
        volume both scale ~linearly, so feeding all views of a large scene OOMs).
        ``strategy`` picks them: ``"fps"`` (farthest-point spread over camera
        positions — best multi-view-stereo coverage), ``"sequential"`` (evenly
        spaced) or ``"random"``.

        ``near`` / ``far`` set the MVS depth-sweep range (128 candidates,
        log-spaced). They default to the resplat training range (0.01 / 200) and
        are deliberately wide — NOT scene-tightened — so distant content is not
        clipped. (These are the depth-prediction bounds, unrelated to a renderer's
        clip planes.)

        ``max_image_side`` caps the longer image side fed to the MVS backbone
        (default 512 ~ the training resolution). Full COLMAP resolution is much
        larger: the cost volume would OOM and the model would run out of
        distribution. Downsampling is uniform and normalized intrinsics are
        resolution-independent, so no intrinsics change is needed; the Gaussian
        count then scales with this resolution (it produces one per 4x4 patch).

        Returns a learn2splat ``Gaussians`` (batch=1, post-activation), ready for
        :meth:`Learn2Splat.initialize_from_tensors`.
        """
        from learn2splat.dataset.data_types import BatchedViews
        from learn2splat.dataset.shims.patch_shim import apply_patch_shim_to_views

        bv = (
            batched_views
            if isinstance(batched_views, BatchedViews)
            else BatchedViews.from_dict(batched_views)
        )

        v = int(bv.image.shape[1])
        k = min(int(num_context_views), v) if int(num_context_views) > 0 else v
        if k < v:
            bv = bv.batchify_views(self._select_views(bv, k, strategy, first_idx_strategy))

        image = bv.image.to(device=self.device, dtype=self.dtype)  # [B, k, 3, H, W]
        b, _, _, h, w = image.shape
        # Downsample to <= max_image_side so the feed-forward MVS cost volume fits in
        # memory and runs at ~the training resolution. The uniform resize preserves the
        # (normalized) intrinsics, so only the image changes.
        if max_image_side and max(h, w) > max_image_side:
            scale = max_image_side / max(h, w)
            image = F.interpolate(
                image.flatten(0, 1), size=(round(h * scale), round(w * scale)),
                mode="bilinear", align_corners=False,
            ).unflatten(0, (b, k))

        # The ResplatInitializer.forward reads a plain views dict. near/far are the
        # depth-sweep bounds (fixed, training-matched) — not the views' own planes.
        ctx = {
            "image": image,
            "extrinsics": bv.extrinsics.to(device=self.device, dtype=self.dtype),
            "intrinsics": bv.intrinsics.to(device=self.device, dtype=self.dtype),
            "near": torch.full((b, k), float(near), device=self.device, dtype=self.dtype),
            "far": torch.full((b, k), float(far), device=self.device, dtype=self.dtype),
        }
        # Crop H/W to a multiple of patch_multiple (and fix intrinsics) as the training
        # data shim does — the MVS backbone requires it.
        ctx = apply_patch_shim_to_views(ctx, patch_size=self.patch_multiple)

        return self.initializer(ctx).gaussians

    @classmethod
    def estimate_num_gaussians(
        cls,
        checkpoint: str,
        height: int,
        width: int,
        *,
        num_context_views: int = _DEFAULT_NUM_CONTEXT_VIEWS,
        max_image_side: int = _DEFAULT_MAX_IMAGE_SIDE,
    ) -> int:
        """Estimate the Gaussian count :meth:`initialize` would produce for a
        ``height`` x ``width`` input at these settings.

        Uses the checkpoint's *config only* (memoized via
        ``_resplat_geometry_for_checkpoint`` — no weights download, no model build), so
        it is cheap enough to call live as a UI setting changes. Mirrors :meth:`initialize`'s
        downsample -> crop -> latent-grid math: each context view contributes
        ``(cropped_h / latent_downsample) * (cropped_w / latent_downsample)`` latent
        cells, times the per-cell Gaussian count.
        """
        cm, ld, per_cell = _resplat_geometry_for_checkpoint(checkpoint)
        h, w = int(height), int(width)
        if max_image_side and max(h, w) > max_image_side:
            scale = max_image_side / max(h, w)
            h, w = round(h * scale), round(w * scale)
        h, w = (h // cm) * cm, (w // cm) * cm
        return int(num_context_views) * (h // ld) * (w // ld) * per_cell

    def _select_views(self, bv, k: int, strategy: str, first_idx_strategy: str):
        """Pick ``k`` context-view indices ``[B, k]`` from ``bv`` per ``strategy``."""
        b, v = int(bv.extrinsics.shape[0]), int(bv.extrinsics.shape[1])
        if strategy == "fps":
            from learn2splat.dataset.view_sampler.view_sampler_bounded_v2 import (
                farthest_point_sample,
            )

            # FPS over the cameras' world positions (same idea as Learn2Splat._view_minibatch).
            positions = bv.extrinsics[..., :3, 3]  # [B, V, 3]
            return farthest_point_sample(positions, k, first_idx_strategy=first_idx_strategy)
        if strategy == "sequential":
            lin = torch.linspace(0, v - 1, k, device=bv.extrinsics.device).round().long()
            return lin.unsqueeze(0).expand(b, -1)
        if strategy == "random":
            return torch.stack(
                [torch.randperm(v, device=bv.extrinsics.device)[:k] for _ in range(b)]
            )
        raise Learn2SplatError(
            f"strategy={strategy!r} not supported (fps | sequential | random)."
        )
