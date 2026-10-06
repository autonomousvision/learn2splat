from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from typeguard import value

from ...misc.step_tracker import StepTracker
from ..data_types import Stage

T = TypeVar("T")


@dataclass
class ViewSamplerCfg:
    name: Literal["base"]
    num_context_views: int
    num_target_views: int


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    is_overfitting: bool
    cameras_are_circular: bool
    step_tracker: StepTracker | None

    def __init__(
            self,
            cfg: T,
            stage: Stage,
            is_overfitting: bool,
            cameras_are_circular: bool,
            step_tracker: StepTracker | None,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

        self._all_context_indices = None
        self._all_target_indices = None

    @property
    def context_indices(self) -> Int64[Tensor, " target_view"]:
        return self._all_context_indices

    @context_indices.setter
    def context_indices(self, indices: Int64[Tensor, " context_view"]):
        if self._all_context_indices is None:
            self._all_context_indices = indices
        else:
            raise RuntimeError("Context indices have already been set.")

    @property
    def target_indices(self) -> Int64[Tensor, " target_view"]:
        return self._all_target_indices

    @target_indices.setter
    def target_indices(self, indices: Int64[Tensor, " target_view"]):
        if self._all_target_indices is None:
            self._all_target_indices = indices
        else:
            raise RuntimeError("Target indices have already been set.")

    @abstractmethod
    def _sample_impl(
            self,
            scene: str,
            extrinsics: Float[Tensor, "view 4 4"],
            intrinsics: Float[Tensor, "view 3 3"],
            device: torch.device = torch.device("cpu"),
            **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        pass

    def sample(
            self,
            scene: str,
            extrinsics: Float[Tensor, "view 4 4"],
            intrinsics: Float[Tensor, "view 3 3"],
            device: torch.device = torch.device("cpu"),
            **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        context_indices, target_indices = self._sample_impl(
            scene=scene,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            device=device,
            **kwargs,
        )
        # self.context_indices = context_indices
        # self.target_indices = target_indices

        return context_indices, target_indices

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()
