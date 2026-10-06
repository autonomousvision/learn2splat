from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler, ViewSamplerCfg


@dataclass
class ViewSamplerDenseCfg(ViewSamplerCfg):
    name: Literal["dense"]
    target_every: int
    context_every: int

    sample_views_strategy: Literal["random", "neighbors"] = "random"

    def __post_init__(self):
        assert (self.target_every > 0) != (self.context_every > 0), \
            "Either target_every or context_every must be set, but not both."


class ViewSamplerDense(ViewSampler[ViewSamplerDenseCfg]):

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
        """Sample context and target views."""
        num_views, _, _ = extrinsics.shape

        all_views = torch.arange(num_views, device=device)

        if self.cfg.target_every > 0:
            target_views = all_views[::self.cfg.target_every]
            context_views = set(all_views.tolist()) - set(target_views.tolist())
            context_views = torch.tensor(list(context_views), device=device)
        elif self.cfg.context_every > 0:
            context_views = all_views[::self.cfg.context_every]
            target_views = set(all_views.tolist()) - set(context_views.tolist())
            target_views = torch.tensor(list(target_views), device=device)
        else:
            raise ValueError("Either target_every or context_every must be set to a positive integer.")

        def sample_views(extrinsics, index_views, num_views_to_sample: int, strategy: str,
                         center_idx: int | None = None) -> Tensor:
            if num_views_to_sample == -1 or num_views_to_sample >= len(index_views):
                return index_views
            if strategy == "random":
                return index_views[torch.randperm(len(index_views))[:num_views_to_sample]]
            elif strategy == "neighbors":
                raise NotImplementedError
                # Choose a random center view and choose views around it, based on cameras extrinsics
                if center_idx is None:
                    center_idx = np.random.choice(
                        len(index_views),
                        size=1,
                        replace=False
                    )[0]
                # Calculate distances to the center view
                rotations = extrinsics[:, :3, :3]  # [V, 3, 3]
                # Calculate camera center as -R^T * t
                translation = extrinsics[:, :3, [3]]  # [V, 3, 1]
                # poses = -rotations.transpose(1, 2) @ translation  # [V, 3, 1]
                poses = translation  # [V, 3, 1]
                center_pose = poses[center_idx]  # [3, 1]
                # Calculate Euclidean distances to the center view
                dists = torch.norm(poses - center_pose.unsqueeze(0), dim=1)[0]  # [V]
                # Calculate angular differences to the center view
                center_rot = extrinsics[center_idx, :3, :3]  # [3, 3]
                # Compute rotation difference
                rot_diffs = torch.matmul(rotations, center_rot.transpose(0, 1))  # [V, 3, 3]
                # Compute angles from rotation matrices
                cos_angles = (rot_diffs[:, 0, 0] + rot_diffs[:, 1, 1] + rot_diffs[:, 2, 2] - 1) / 2  # [V]
                cos_angles = torch.clamp(cos_angles, -1.0, 1.0)  # Numerical stability
                angles = torch.acos(cos_angles)  # [V]
                # Combine distance and angle into a single metric
                combined_metric = dists + angles  # [V]

                # Get the indices of the nearest neighbors
                combined_metric = combined_metric[index_views]
                sorted_indices = torch.argsort(combined_metric)

                return index_views[sorted_indices[:num_views_to_sample]]
            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}")

        index_context = sample_views(extrinsics, context_views, self.cfg.num_context_views,
                                     self.cfg.sample_views_strategy)
        index_target = sample_views(extrinsics, target_views, self.cfg.num_target_views, self.cfg.sample_views_strategy,
                                    center_idx=index_context[0].item())

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
