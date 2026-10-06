from dataclasses import dataclass
from typing import Literal

from .cuda_splatting import render_cuda, render_depth_cuda
from .splatting_cuda_decoder import SplattingCUDADecoder


@dataclass
class InriaDecoderSplattingCUDACfg:
    name: Literal["inria"]
    scale_invariant: bool
    # False: pass scales+rotations and let the CUDA kernel compute the covariance.
    # True: precompute Python-side and pass cov3D_precomp (~42 dB pixel drift from
    # the kernel's own path, slightly faster on repeat calls).
    use_covariances: bool = False


class InriaDecoderSplattingCUDA(SplattingCUDADecoder[InriaDecoderSplattingCUDACfg]):
    """Inria diff_gaussian_rasterization backend. Only the rasterizer calls differ from the
    shared base; see splatting_cuda_decoder.SplattingCUDADecoder for the orchestration."""

    def _raster(self, ext, intr, near, far, image_shape, bg, means, covars, shs, opacities,
                scales, rotations_wxyz, means2d_out, means2d_abs_out=None):
        # means2d_abs_out is FastGS-only; the inria backend has no abs-gradient and ignores it.
        return render_cuda(
            ext, intr, near, far, image_shape, bg, means, covars, shs, opacities,
            scale_invariant=self.cfg.scale_invariant,
            gaussian_scales=scales,
            gaussian_rotations=rotations_wxyz,
            means2d_out=means2d_out,
        )

    def _raster_depth(self, ext, intr, near, far, image_shape, means, covars, opacities, mode):
        return render_depth_cuda(
            ext, intr, near, far, image_shape, means, covars, opacities,
            mode=mode, scale_invariant=self.cfg.scale_invariant,
        )
