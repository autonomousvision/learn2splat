from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import torch
from jaxtyping import Float, Int32, Bool, UInt8
from torch import Tensor, nn

from ..types import Gaussians
from ...dataset import DatasetCfg
from ...dataset.data_types import BatchedViews, BatchedViewsDict, BatchedExample
from ...scene_trainer.gaussian_module import GaussiansModule


DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"] | UInt8[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None
    normal: Float[Tensor, "batch view 3 height width"] | None = None
    distortion_map: Float[Tensor, "batch view height width"] | None = None
    accumulated_alpha: Float[Tensor, "batch view height width"] | None = None
    radii: Int32[Tensor, "batch view n 2"] | None = None
    means2d: Float[Tensor, "batch view n 2"] | None = None
    # FastGS Abs-GS split signal: gradient of the *absolute* screen-space mean (cols [2:] of the
    # FastGS [N,4] screen tensor). Only the FastGS decoder populates it; reachable via autograd.
    means2d_abs: Float[Tensor, "batch view n 2"] | None = None
    visibility_filter: Bool[Tensor, "batch view n"] | None = None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    def means2d_grad_to_ndc(
        self,
        grad: Float[Tensor, "*batch n 2"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, "*batch n 2"]:
        """Normalize an autograd.grad(loss, decoder_output.means2d) result to the resolution-
        independent NDC ([-1, 1]) screen convention.

        This makes the ADC / densification strategy renderer-agnostic: every backend hands it a
        uniform NDC gradient, so a single threshold (the 3DGS / FastGS
        ``densify_grad_threshold = 0.0002``) is correct for all of them, and the strategy itself no
        longer needs to know each renderer's pixel scale.

        The 3DGS-family backends (inria/fastgs) already emit NDC gradients, so the default is
        identity. Only the gsplat backend (pixel-space, gradient ∝ image size) overrides this."""
        return grad

    @abstractmethod
    def forward(
            self,
            gaussians: Gaussians | GaussiansModule,
            extrinsics: Float[Tensor, "batch view 4 4"],
            intrinsics: Float[Tensor, "batch view 3 3"],
            near: Float[Tensor, "batch view"],
            far: Float[Tensor, "batch view"],
            image_shape: tuple[int, int],
            depth_mode: DepthRenderingMode | None = None,
            to_cpu: bool = False,
    ) -> DecoderOutput:
        pass

    def forward_batch(
            self,
            gaussians: Gaussians | GaussiansModule,
            batch: BatchedExample,
            image_shape: tuple[int, int] | None = None,
            input_str: Literal["context", "target"] | None = None,
            eval_context_views: bool | None = None,
            depth_mode: DepthRenderingMode | None = None,
            start=None, end=None,
            camera_poses=None,  # In case of manipulating camera poses (e.g. for stabilization)
            to_cpu: bool = False,  # move outputs to cpu as they are rendered
            iter_batch_size: int = -1,  # -1 to render all views at once
    ) -> DecoderOutput:

        assert input_str is not None or eval_context_views is not None
        if input_str is None:
            input_str = "context" if eval_context_views else "target"

        input = batch[input_str]

        if image_shape is None:
            image_shape = input["image_shape"].shape[-2:]
        if camera_poses is None:
            camera_poses = input["extrinsics"]
        intrinsics, near, far = input["intrinsics"], input["near"], input["far"]
        if camera_poses.shape[1] != intrinsics.shape[1]:
            # A generated camera path (e.g. a video trajectory) has its own frame count, so
            # there is no per-view intrinsic to pair it with: every frame borrows the first
            # view's camera and depth range.
            frames = camera_poses.shape[1]
            intrinsics = intrinsics[:, :1].expand(-1, frames, -1, -1)
            near, far = near[:, :1].expand(-1, frames), far[:, :1].expand(-1, frames)
        return self.forward(
            gaussians,
            camera_poses[:, start:end],
            intrinsics[:, start:end],
            near[:, start:end],
            far[:, start:end],
            image_shape,
            depth_mode=depth_mode,
            to_cpu=to_cpu,
            iter_batch_size=iter_batch_size,
        )

    def forward_batch_subset(self, gaussians: Gaussians | GaussiansModule,
                             batch_subset: BatchedViewsDict | BatchedViews,
                             image_shape: tuple[int, int] | None = None,
                             start: int | None = None,
                             end: int | None = None,
                             indices: torch.Tensor | list | None = None,
                             **kwargs) -> DecoderOutput:

        assert not ((start is not None and end is not None) and (
                indices is not None)), "Either start and end or indices must be provided."
        if start is not None:
            indices = list(range(start, end))

        if indices is None:
            indices = list(range(batch_subset["extrinsics"].shape[1]))

        if isinstance(indices, list):
            # Convert list to tensor for one flow handling
            indices = torch.tensor(indices, device=batch_subset["extrinsics"].device)
            indices = indices.unsqueeze(0).expand(batch_subset["extrinsics"].shape[0], -1)  # (batch, num_indices)

        if image_shape is None:
            image_shape = batch_subset["image"].shape[-2:]

        assert indices.dim() == 2, "Indices tensor must be 2D (scene_batch, num_indices)."
        scene_batch = indices.size(0)
        scene_batch_idx = torch.arange(scene_batch, device=indices.device)[:, None]  # (batch, 1)
        return self.forward(gaussians,
                            batch_subset["extrinsics"][scene_batch_idx, indices],
                            batch_subset["intrinsics"][scene_batch_idx, indices],
                            batch_subset["near"][scene_batch_idx, indices],
                            batch_subset["far"][scene_batch_idx, indices],
                            image_shape,
                            **kwargs)

    def forward_context(
            self,
            gaussians: Gaussians | GaussiansModule,
            batch: BatchedExample,
            image_shape: tuple[int, int] | None = None,
            depth_mode: DepthRenderingMode | None = None,
            **kwargs,
    ) -> DecoderOutput:
        return self.forward_batch(
            gaussians,
            batch,
            image_shape,
            "context",
            depth_mode=depth_mode,
            **kwargs,
        )

    def forward_target(
            self,
            gaussians: Gaussians | GaussiansModule,
            batch: BatchedExample,
            image_shape: tuple[int, int] | None = None,
            depth_mode: DepthRenderingMode | None = None,
            **kwargs,
    ) -> DecoderOutput:
        return self.forward_batch(
            gaussians,
            batch,
            image_shape,
            "target",
            depth_mode=depth_mode,
            **kwargs,
        )
