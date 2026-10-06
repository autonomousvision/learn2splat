"""Shared orchestration for the CUDA-rasterizer decoders (inria, fastgs).

Both backends share identical forward/depth steps: flattening B×V cameras,
handling Gaussians vs GaussiansModule, optional view chunking, and reshaping the
rasterizer outputs back to (B, V). Only the rasterizer call itself differs, so
that is the single overridable hook (``_raster`` / ``_raster_depth``).

This module imports no rasterizer backend, so importing it never requires
``diff_gaussian_rasterization`` or ``diff_gaussian_rasterization_fastgs``.
"""

from typing import TypeVar

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from ...dataset import DatasetCfg
from ...scene_trainer.gaussian_module import GaussiansModule
from ..types import Gaussians
from .decoder import Decoder, DecoderOutput, DepthRenderingMode

T = TypeVar("T")


class SplattingCUDADecoder(Decoder[T]):
    """Base for the inria/fastgs splatting decoders. Subclasses implement only the two
    backend hooks below; everything else (camera flattening, output reshaping) is shared."""

    background_color: Float[Tensor, "3"]

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    # --- backend hooks -----------------------------------------------------
    # Both receive flat (B*V) camera tensors and return flat outputs.

    def _raster(self, ext, intr, near, far, image_shape, bg, means, covars, shs, opacities,
                scales, rotations_wxyz, means2d_out, means2d_abs_out=None):
        """Renders, writing screen-space means into ``means2d_out`` ([bv,N,2]) so its gradient
        is reachable via autograd.grad. Returns (images [bv,3,H,W], radii [bv,N]).

        ``means2d_abs_out`` is the optional FastGS Abs-GS leaf (cols [2:] of its screen tensor);
        backends that do not produce it ignore the argument."""
        raise NotImplementedError

    def _raster_depth(self, ext, intr, near, far, image_shape, means, covars, opacities, mode):
        """Returns depth [bv, H, W]."""
        raise NotImplementedError

    def _produces_abs_grad(self) -> bool:
        """Whether the backend exposes the FastGS abs-gradient (cols [2:]) via ``means2d_abs_out``.
        Only the FastGS decoder overrides this to True; gsplat/inria stay False."""
        return False

    # --- shared Gaussian tensor prep --------------------------------------

    def _prepare_flat_gaussians(self, gaussians: Gaussians | GaussiansModule, b: int, v: int):
        """Flatten Gaussian params to (B*V) rasterizer layout. Returns
        (means, shs, opacities, scales, rotations_wxyz, covars); scales/rotations are None when
        ``use_covariances`` (covars supplied instead) and vice-versa. Shared by forward and the
        FastGS metric-counts render so they build identical inputs."""
        bv = b * v
        scales = rotations_wxyz = covars = None
        if isinstance(gaussians, GaussiansModule):
            means = repeat(gaussians.means, "g xyz -> bv g xyz", bv=bv)
            shs = repeat(gaussians.harmonics, "g c d -> bv g c d", bv=bv)
            opacities = repeat(gaussians.opacities, "g -> bv g", bv=bv)
            if self.cfg.use_covariances:
                covars = repeat(gaussians.covariances, "g i j -> bv g i j", bv=bv)
            else:
                scales = repeat(gaussians.scales, "g d -> bv g d", bv=bv)
                # gaussians.rotations is xyzw post-normalization; the rasterizer wants wxyz.
                rotations_wxyz = repeat(gaussians.rotations[:, [3, 0, 1, 2]], "g d -> bv g d", bv=bv)
        elif isinstance(gaussians, Gaussians):
            means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
            shs = repeat(gaussians.harmonics, "b g c d -> (b v) g c d", v=v)
            opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)
            if self.cfg.use_covariances:
                if gaussians.covariances is None:
                    raise ValueError("use_covariances=true but gaussians.covariances is None.")
                covars = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)
            else:
                _scales = gaussians.scales if gaussians.stores_activated else torch.exp(gaussians.scales)
                scales = repeat(_scales, "b g d -> (b v) g d", v=v)
                # Normalize rotations_unnorm here (the always-present grad leaf) rather than
                # using gaussians.rotations: the learned optimizer precomputes the latter under
                # torch.no_grad(), so the screen-space loss must reach rotations_unnorm directly
                # (else autograd.grad sees it as unused). Numerically == gaussians.rotations;
                # xyzw -> wxyz for the rasterizer. Mirrors the gsplat decoder.
                rot = torch.nn.functional.normalize(gaussians.rotations_unnorm, dim=-1)
                rotations_wxyz = repeat(rot[..., [3, 0, 1, 2]], "b g d -> (b v) g d", v=v)
            if not gaussians.stores_activated:
                opacities = torch.sigmoid(opacities)
        else:
            raise ValueError(f"Unknown gaussians type: {type(gaussians)}")
        return means, shs, opacities, scales, rotations_wxyz, covars

    # --- shared forward ----------------------------------------------------

    def forward(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_radii: bool = False,
        iter_batch_size: int = -1,
        to_cpu: bool = False,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        bv = b * v

        # Flatten camera params to (B*V)
        flat_ext = rearrange(extrinsics, "b v i j -> (b v) i j")
        flat_int = rearrange(intrinsics, "b v i j -> (b v) i j")
        flat_near = rearrange(near, "b v -> (b v)")
        flat_far = rearrange(far, "b v -> (b v)")
        flat_bg = repeat(self.background_color, "c -> (b v) c", b=b, v=v)

        # Prepare Gaussian tensors in flat (B*V) format
        means, shs, opacities, scales, rotations_wxyz, covars = self._prepare_flat_gaussians(gaussians, b, v)

        # Single [B, V, N, 2] screen-space-means leaf. Each view's rasterizer call consumes a
        # slice of it (via means2d_flat), so the 2D gradient is reachable as
        # torch.autograd.grad(loss, out.means2d) — uniformly with the gsplat decoder, and
        # without a .backward()/.grad pass. Returned as-is (no reshape, which would detach it).
        n_gauss = means.shape[1]
        means2d = torch.zeros((b, v, n_gauss, 2), dtype=means.dtype, device=means.device,
                              requires_grad=True)
        means2d_flat = means2d.reshape(bv, n_gauss, 2)

        # FastGS only: a second leaf for the abs-gradient (cols [2:] of its screen tensor).
        means2d_abs = means2d_abs_flat = None
        if self._produces_abs_grad():
            means2d_abs = torch.zeros((b, v, n_gauss, 2), dtype=means.dtype, device=means.device,
                                      requires_grad=True)
            means2d_abs_flat = means2d_abs.reshape(bv, n_gauss, 2)

        def _render_flat(s: slice):
            return self._raster(
                flat_ext[s], flat_int[s], flat_near[s], flat_far[s], image_shape, flat_bg[s],
                means[s], covars[s] if covars is not None else None, shs[s], opacities[s],
                scales[s] if scales is not None else None,
                rotations_wxyz[s] if rotations_wxyz is not None else None,
                means2d_flat[s],
                means2d_abs_flat[s] if means2d_abs_flat is not None else None,
            )

        if iter_batch_size < 0:
            imgs, radii_flat = _render_flat(slice(None))
            if to_cpu:
                imgs = imgs.detach().cpu()
                radii_flat = radii_flat.detach().cpu()
        else:
            all_imgs, all_radii = [], []
            for i in tqdm(range(0, bv, iter_batch_size), desc="Rendering in batches"):
                s = slice(i, min(i + iter_batch_size, bv))
                imgs_c, rad_c = _render_flat(s)
                if to_cpu:
                    imgs_c = imgs_c.detach().cpu()
                    rad_c = rad_c.detach().cpu()
                all_imgs.append(imgs_c)
                all_radii.append(rad_c)
            imgs = torch.cat(all_imgs, dim=0)
            radii_flat = torch.cat(all_radii, dim=0)

        # Reshape (B*V) → (B, V)
        color = rearrange(imgs, "(b v) c h w -> b v c h w", b=b, v=v)
        radii_bv = rearrange(radii_flat, "(b v) n -> b v n", b=b, v=v)
        means2d_bv = means2d.detach().cpu() if to_cpu else means2d  # [B, V, N, 2]
        means2d_abs_bv = None
        if means2d_abs is not None:
            means2d_abs_bv = means2d_abs.detach().cpu() if to_cpu else means2d_abs  # [B, V, N, 2]

        # Expand scalar radii [B, V, N] → [B, V, N, 2] to match gsplat interface
        radii_out = radii_bv.unsqueeze(-1).expand(-1, -1, -1, 2).contiguous()
        visibility_filter = radii_bv > 0  # [B, V, N]

        depth = (
            self._render_depth(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode)
            if depth_mode is not None
            else None
        )

        return DecoderOutput(
            color=color,
            depth=depth,
            accumulated_alpha=None,
            means2d=means2d_bv,
            means2d_abs=means2d_abs_bv,
            radii=radii_out,
            visibility_filter=visibility_filter,
        )

    # --- shared depth ------------------------------------------------------

    def _render_depth(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape

        if isinstance(gaussians, GaussiansModule):
            means = repeat(gaussians.means, "g xyz -> (b v) g xyz", b=b, v=v)
            covars = repeat(gaussians.covariances, "g i j -> (b v) g i j", b=b, v=v)
            opacities = repeat(gaussians.opacities, "g -> (b v) g", b=b, v=v)
        else:
            means = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v)
            covars = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v)
            opacities = repeat(gaussians.opacities, "b g -> (b v) g", v=v)
            if not gaussians.stores_activated:
                opacities = torch.sigmoid(opacities)

        result = self._raster_depth(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            means,
            covars,
            opacities,
            mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
