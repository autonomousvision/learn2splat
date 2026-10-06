from abc import ABC
from dataclasses import dataclass
from typing import TypeVar, Generic

import torch
from torch import nn

from learn2splat.model.types import Gaussians
from learn2splat.model.decoder.decoder import DecoderOutput

T = TypeVar("T")


@dataclass
class InitializerOutput:
    gaussians: Gaussians
    features: torch.Tensor | None = None
    depths: list[torch.Tensor] | torch.Tensor | None = None
    target_render: DecoderOutput | None = None
    context_render: DecoderOutput | None = None
    # View indices used when rendering a subset (training); None means all views were rendered.
    target_render_index: torch.Tensor | None = None
    context_render_index: torch.Tensor | None = None

    def get_render(self, which: str) -> DecoderOutput | None:
        if which == "target":
            return self.target_render
        elif which == "context":
            return self.context_render
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render(self, which: str, value: DecoderOutput) -> None:
        if which == "target":
            self.target_render = value
        elif which == "context":
            self.context_render = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def get_render_index(self, which: str) -> torch.Tensor | None:
        if which == "target":
            return self.target_render_index
        elif which == "context":
            return self.context_render_index
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")

    def set_render_index(self, which: str, value: torch.Tensor | None) -> None:
        if which == "target":
            self.target_render_index = value
        elif which == "context":
            self.context_render_index = value
        else:
            raise ValueError(f"Unknown which: {which}, should be 'target' or 'context'")


@dataclass
class InitializerCfg:
    per_pixel: bool
    per_view: bool

    # Gaussian subsampling augmentation (applied before fixed_gaussians_num)
    # Set min=max for a fixed subsample count, or use floats for ratio-based sampling
    train_min_gaussians_subsample: int | float | None
    train_max_gaussians_subsample: int | float | None
    eval_min_gaussians_subsample: int | float | None
    eval_max_gaussians_subsample: int | float | None

    # Final fixed Gaussian count for DDP consistency (subsample or pad to reach this)
    # Applied after subsampling augmentation
    train_fixed_gaussians_num: int | None
    eval_fixed_gaussians_num: int | None

    def get_gaussian_param_num(self) -> int:
        """Per-Gaussian parameter count of this initializer's own representation:
        scale(3) + rotation(4) + SH(3*sh_d) + opacity(1) + position encoding (get_position_param_num)."""
        return 3 + 4 + 3 * self.get_sh_d() + 1 + self.get_position_param_num()

    def get_position_param_num(self) -> int:
        """Size of this initializer's position encoding. Most initializers place Gaussians at explicit
        3D positions (3). Initializers that encode position differently (resplat: a 2D pixel offset
        added to a per-pixel depth) override this."""
        return 3

    def get_init_gaussian_multiple(self) -> int:
        """Number of Gaussians produced per latent/pixel. Default 1 (one Gaussian per point).
        Resplat overrides this when init_gaussian_multiple > 1."""
        return 1

@dataclass
class NonlearnedInitializerCfg(InitializerCfg):
    pass

@dataclass
class LearnedInitializerCfg(InitializerCfg):
    pass


@dataclass
class PerPixelInitializerCfg(LearnedInitializerCfg):
    latent_gs: bool
    latent_downsample: int


class Initializer(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    def eval_preprocessing(self, batch, train_cfg) -> None:
        """Eval/validation-only batch prep (depth-range override + optional scale prediction),
        applied in-place before the initializer runs. Training does not call this. The
        universal patch-crop data shim is a separate, always-applied step (see MetaTrainer)."""
        pass

    @property
    def fixed_gaussians_num(self) -> int | None:
        """Target Gaussian count (train or eval, per mode). Point-based initializers subsample or
        pad to this so every DDP process optimizes the same number of Gaussians."""
        return self.cfg.train_fixed_gaussians_num if self.training else self.cfg.eval_fixed_gaussians_num

    def subsample_points(self, xyz: torch.Tensor, rgbs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-initialization "sample Gaussians" step, shared by the point-based initializers
        (colmap, pointcloud, depth). Sibling to the `eval_preprocessing` pre-hook.

        Two stages, both selecting the train/eval config values by mode: an optional subsampling
        augmentation (random count in the configured range), then a subsample down to the fixed
        DDP count. Padding up to the fixed count is left to each initializer (it happens after the
        per-point scales/opacities are built). Returns the possibly-reduced (xyz, rgbs)."""
        min_sub = self.cfg.train_min_gaussians_subsample if self.training else self.cfg.eval_min_gaussians_subsample
        max_sub = self.cfg.train_max_gaussians_subsample if self.training else self.cfg.eval_max_gaussians_subsample
        if min_sub is not None or max_sub is not None:
            target = self.sample_num_gaussians(xyz.shape[0], min_sub, max_sub)
            if xyz.shape[0] > target:
                idx = torch.randperm(xyz.shape[0], device=xyz.device)[:target]
                xyz, rgbs = xyz[idx], rgbs[idx]

        fixed_num = self.fixed_gaussians_num
        if fixed_num is not None and xyz.shape[0] > fixed_num:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)[:fixed_num]
            xyz, rgbs = xyz[idx], rgbs[idx]
        return xyz, rgbs

    @staticmethod
    def sample_num_gaussians(
        available: int,
        min_val: int | float | None,
        max_val: int | float | None,
    ) -> int:
        """Pick a target Gaussian count for the subsampling augmentation (int = absolute count,
        float = ratio of the available points)."""
        if min_val is None and max_val is None:
            return available
        assert min_val is not None and max_val is not None, \
            "Both min and max must be set together for Gaussian subsampling."
        assert type(min_val) == type(max_val), \
            "min and max must be the same type (both int or both float)."
        if isinstance(min_val, int):
            count = torch.randint(min_val, max_val + 1, (1,)).item()
        else:
            assert 0.0 < min_val <= 1.0 and 0.0 < max_val <= 1.0, \
                "Float subsampling ratios must be in (0, 1]."
            ratio = torch.empty(1).uniform_(min_val, max_val).item()
            count = int(available * ratio)
        return min(count, available)

    @property
    def strategy(self) -> str:
        raise NotImplementedError()


class LearnedInitializer(Initializer[T], ABC):
    @property
    def strategy(self) -> str:
        return "learned"


class NonlearnedInitializer(Initializer[T], ABC):
    @property
    def strategy(self) -> str:
        return "nonlearned"
