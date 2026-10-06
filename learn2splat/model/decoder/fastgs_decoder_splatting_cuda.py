from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from ...scene_trainer.gaussian_module import GaussiansModule
from ..types import Gaussians
from .cuda_splatting_fastgs import (
    render_cuda_fastgs,
    render_depth_cuda_fastgs,
    render_metric_counts_fastgs,
)
from .splatting_cuda_decoder import SplattingCUDADecoder


@dataclass
class FastGSDecoderSplattingCUDACfg:
    name: Literal["fastgs"]
    scale_invariant: bool
    # False: pass scales+rotations and let the CUDA kernel compute the covariance.
    # True: precompute Python-side and pass cov3D_precomp.
    use_covariances: bool = False
    # FastGS compact-box multiplier (controls the tile count per splat); matches the
    # FastGS training default.
    mult: float = 0.5


class FastGSDecoderSplattingCUDA(SplattingCUDADecoder[FastGSDecoderSplattingCUDACfg]):
    """FastGS diff_gaussian_rasterization_fastgs backend. Only the rasterizer calls differ from
    the shared base; see splatting_cuda_decoder.SplattingCUDADecoder for the orchestration."""

    def _produces_abs_grad(self) -> bool:
        # FastGS's [N,4] screen tensor carries the Abs-GS split signal in cols [2:]; expose it.
        return True

    def _raster(self, ext, intr, near, far, image_shape, bg, means, covars, shs, opacities,
                scales, rotations_wxyz, means2d_out, means2d_abs_out=None):
        return render_cuda_fastgs(
            ext, intr, near, far, image_shape, bg, means, covars, shs, opacities,
            scale_invariant=self.cfg.scale_invariant,
            gaussian_scales=scales,
            gaussian_rotations=rotations_wxyz,
            mult=self.cfg.mult,
            means2d_out=means2d_out,
            means2d_abs_out=means2d_abs_out,
        )

    def _raster_depth(self, ext, intr, near, far, image_shape, means, covars, opacities, mode):
        return render_depth_cuda_fastgs(
            ext, intr, near, far, image_shape, means, covars, opacities,
            mode=mode, scale_invariant=self.cfg.scale_invariant, mult=self.cfg.mult,
        )

    @torch.no_grad()
    def render_metric_counts(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        metric_maps: Int[Tensor, "view height_width"],  # per-view binary high-error pixel maps
    ) -> Int[Tensor, "view gaussian"]:
        """FastGS multi-view importance signal: per-view per-Gaussian counts of flagged (high-error)
        pixels each Gaussian contributed to. Assumes batch size 1; ``metric_maps`` has one flattened
        [H*W] map per view. Reduce with ``adc.fastgs.compute_fastgs_scores``."""
        b, v, _, _ = extrinsics.shape
        assert b == 1, "render_metric_counts assumes scene batch size 1"
        means, shs, opacities, scales, rotations_wxyz, covars = self._prepare_flat_gaussians(gaussians, b, v)
        bg = repeat(self.background_color, "c -> (b v) c", b=b, v=v)
        return render_metric_counts_fastgs(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape, bg, means,
            covars,
            shs, opacities, metric_maps,
            scale_invariant=self.cfg.scale_invariant,
            gaussian_scales=scales,
            gaussian_rotations=rotations_wxyz,
            mult=self.cfg.mult,
        )
