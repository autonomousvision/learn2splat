'''
Modifiedy from latentSplat and pixelSplat to handle extrapolate and more context views
'''
import copy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import random

from .view_sampler import ViewSampler


def farthest_point_sample(xyz, npoint, first_idx_strategy="max_dist"):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    if first_idx_strategy == 'max_dist':
        barycenter = torch.sum((xyz), 1)
        barycenter = barycenter / xyz.shape[1]
        barycenter = barycenter.view(B, 1, 3)

        dist = torch.sum((xyz - barycenter) ** 2, -1)
        curr_idx = torch.max(dist, 1)[1]
    elif first_idx_strategy == 'random':
        curr_idx = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        raise ValueError(f"Unknown first_idx_strategy: {first_idx_strategy}")

    for i in range(npoint):
        centroids[:, i] = curr_idx
        centroid = xyz[batch_indices, curr_idx, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        curr_idx = torch.max(distance, -1)[1]

    return centroids


@dataclass
class ViewSamplerBoundedV2Cfg:
    name: Literal["boundedv2"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    max_distance_to_context_views: int
    context_gap_warm_up_steps: int
    target_gap_warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    initial_max_distance_to_context_views: int
    extra_views_sampling_strategy: Optional[Literal["random", "farthest_point", "equal"]] = "random"
    target_views_replace_sample: Optional[bool] = True


class ViewSamplerBoundedV2(ViewSampler[ViewSamplerBoundedV2Cfg]):

    def __init__(self, cfg, stage, is_overfitting: bool, cameras_are_circular: bool,
                 step_tracker) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)
        self._cfg_backup = copy.deepcopy(cfg)

    def schedule(self, initial: int, final: int, steps: int) -> int:
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def _sample_impl(
            self,
            scene: str,
            extrinsics: Float[Tensor, "view 4 4"],
            intrinsics: Float[Tensor, "view 3 3"],
            device: torch.device = torch.device("cpu"),
            max_num_views: Optional[int] = None,
            min_context_views: int = 0,
            max_context_views: int = 0,
            min_view_dist: int | None = None,
            max_view_dist: int | None = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        if max_num_views is not None:
            num_views = min(num_views, max_num_views)

        def determine_per_scene_values(name, value):
            if getattr(self._cfg_backup, name) < 0:
                setattr(self.cfg, name, value)

        determine_per_scene_values('max_distance_between_context_views', num_views)
        determine_per_scene_values('initial_max_distance_between_context_views', num_views)
        determine_per_scene_values('min_distance_between_context_views', num_views-1)
        determine_per_scene_values('initial_min_distance_between_context_views', num_views-1)

        if min_context_views > 0 and max_context_views > 0 and self.stage != "test":
            random_num_views = random.randint(min_context_views, max_context_views)
        else:
            random_num_views = None

        context_gap = self.get_context_gap(device, max_context_views, max_view_dist, min_view_dist, num_views,
                                           random_num_views)
        if context_gap < 0:
            context_gap = num_views

        # Compute the margin from context window to target window based on the current global step
        max_target_gap = self.get_max_target_gap()
        if max_target_gap < 0:
            max_target_gap = num_views + 1

        # Pick the left and right context indices.
        index_context_left, index_context_right, index_target_left, index_target_right = self.get_bound_indices(
            context_gap, device, max_target_gap, num_views)

        # Note: targets are sampled before extra context views — order matters for reproducibility.
        index_target = self.get_target_indices(device, index_target_left, index_target_right,
                                               [index_context_left, index_context_right])

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.
        if random_num_views is not None:
            total_num_views = random_num_views
        else:
            total_num_views = self.cfg.num_context_views

        extra_views, index_context_left, index_context_right = self.get_extra_views(extrinsics, index_context_left,
                                                                                    index_context_right,
                                                                                    total_num_views,
                                                                                    index_target)
        index_context = torch.tensor((index_context_left, *extra_views, index_context_right))
        assert set(index_context.tolist()).isdisjoint(set(index_target.tolist())), \
            f"Context and target views overlap! Context: {index_context}, target: {index_target}"

        return index_context, index_target

    def get_extra_views(self, extrinsics, index_context_left, index_context_right, total_num_views, index_target):
        if total_num_views > 2:
            num_extra_views = total_num_views - 2
            extra_views = []
            if self.cfg.extra_views_sampling_strategy == 'random':
                extra_views = self.sample_unique_excluding(
                    index_context_left + 1,
                    index_context_right - 1,
                    num_extra_views,
                    index_target,
                )
            elif self.cfg.extra_views_sampling_strategy == 'farthest_point':
                context_bounded_index = torch.arange(index_context_left, index_context_right + 1)
                # remove target views from candidates
                context_bounded_index = torch.tensor([i for i in context_bounded_index if i not in index_target])
                candidate_views_position = extrinsics[context_bounded_index, :3, -1].unsqueeze(0)
                index_context_local = farthest_point_sample(candidate_views_position, total_num_views).squeeze(0)
                # remap context index back to global scene based index
                index_context = context_bounded_index[index_context_local]
                index_context = index_context.sort().values
                index_context_left = index_context[0].item()
                index_context_right = index_context[-1].item()
                extra_views = index_context[1:-1].tolist()
            elif self.cfg.extra_views_sampling_strategy == 'equal':
                pass

            # sort the index
            extra_views = sorted(extra_views)
        else:
            extra_views = []
        return extra_views, index_context_left, index_context_right

    def get_max_target_gap(self):
        if self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps,
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views
        return max_target_gap

    def get_context_gap(self, device, max_context_views, max_view_dist, min_view_dist, num_views, random_num_views):
        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
            )
            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
            )
        else:
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.min_distance_between_context_views
        if min_view_dist is not None and max_view_dist is not None:
            # for mixed dataset training, with different sampling distance
            min_context_gap = min_view_dist
            max_context_gap = max_view_dist
        if random_num_views is not None:
            # smaller context gap accordingly
            scale_factor = max(max_context_views // random_num_views, 1)
            max_context_gap = max_context_gap // scale_factor
            min_context_gap = min_context_gap // scale_factor
        if not self.cameras_are_circular:
            max_context_gap = min(
                num_views - 1, max_context_gap
            )
        # Pick the gap between the context views.
        if max_context_gap < min_context_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_context_gap,
            max_context_gap + 1,
            size=tuple(),
            device=device,
        ).item()
        return context_gap

    @staticmethod
    def sample_unique_excluding(left, right, num_samples, exclude_list):
        candidates = [i for i in range(left, right + 1) if i not in exclude_list]
        if len(candidates) < num_samples:
            raise ValueError("Not enough candidates to sample from!")

        # Sample without replacement
        indices = torch.randperm(len(candidates))[:num_samples]
        samples = [candidates[i] for i in indices]
        assert len(set(samples)) == num_samples, f"Expected {num_samples} unique samples, got {set(samples)}"
        return samples

    def get_target_indices(self, device, index_target_left, index_target_right, excluded_indices):
        if self.stage == "test":
            candidates = [i for i in range(index_target_left, index_target_right + 1)
                          if i not in excluded_indices]
            index_target = torch.tensor(candidates[:self.cfg.num_target_views], device=device)
        else:
            if self.cfg.target_views_replace_sample:
                # Sample with replacement from candidates excluding context views.
                candidates = [i for i in range(index_target_left, index_target_right + 1)
                              if i not in excluded_indices]
                rand_indices = torch.randint(0, len(candidates), size=(self.cfg.num_target_views,), device=device)
                index_target = torch.tensor([candidates[i] for i in rand_indices], device=device)
            else:
                index_target = self.sample_unique_excluding(
                    index_target_left,
                    index_target_right,
                    self.cfg.num_target_views,
                    excluded_indices,
                )
                index_target = torch.tensor(index_target, device=device)
        return index_target

    def get_bound_indices(self, context_gap, device, max_target_gap, num_views):
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap
        index_target_left = index_context_left - max_target_gap
        index_target_right = index_context_right + max_target_gap
        if not self.cameras_are_circular:
            index_target_left = max(0, index_target_left)
            index_target_right = min(num_views - 1, index_target_right)
        return index_context_left, index_context_right, index_target_left, index_target_right

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
