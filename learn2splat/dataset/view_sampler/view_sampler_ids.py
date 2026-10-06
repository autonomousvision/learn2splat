from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerIDsCfg:
    name: Literal["ids"]
    context_views_ids: list[int]
    target_views_ids: list[int]


class ViewSamplerIDs(ViewSampler[ViewSamplerIDsCfg]):
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
        v, _, _ = extrinsics.shape
        context_indices = torch.tensor(self.cfg.context_views_ids, device=device, dtype=torch.int64)
        target_indices = torch.tensor(self.cfg.target_views_ids, device=device, dtype=torch.int64)
        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return len(self.cfg.context_views_ids)
    
    @property
    def num_target_views(self) -> int:
        return len(self.cfg.target_views_ids)