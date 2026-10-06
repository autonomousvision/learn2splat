from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from learn2splat.dataset.data_types import BatchedExample
from learn2splat.loss import Loss
from learn2splat.scene_trainer.optimizer.optimizer import OptimizerOutput


@dataclass
class LossStabilityCfg:
    weight: float | int
    # When per-step view subsampling is active, also penalize the subsampled inputs (each view is
    # compared against its previous visit). Off by default: subsampled inputs are skipped and the
    # loss only covers full-view inputs. NOTE: this path is prototyped but not fully tested in training.
    subset_aware: bool = False
    # Apply stability only to the target views, never the context views. In the dense setting the
    # context views are subsampled per step and so are already skipped, so dense stability is
    # target-only; in the sparse setting all context views are present every step, so they are
    # included and roughly double the effective penalty. Set this to match the dense behavior
    # (target-only) and keep the recipe consistent across dense/sparse.
    targets_only: bool = False


@dataclass
class LossStabilityCfgWrapper:
    stability: LossStabilityCfg


class LossStability(Loss[LossStabilityCfg, LossStabilityCfgWrapper]):
    """Penalizes per-view increases in reconstruction error between consecutive optimizer iterations,
    pushing the refinement to improve (or at least not worsen) each view across steps. Unlike the other
    losses it reads the optimizer's full render trajectory, so it is computed outside the per-step loop."""
    use_in_inner_steps = False

    def forward(
            self,
            optimizer_output: OptimizerOutput,
            batch: BatchedExample,
            **kwargs,
    ) -> Float[Tensor, ""]:
        total_loss = torch.tensor(0.0, device=optimizer_output.get_render_list("context")[0].color.device)
        # Stability loss: encourage the model to produce similar outputs for the same input across iterations.
        inputs = ["target"] if self.cfg.targets_only else ["context", "target"]
        for input_str in inputs:
            render_list = optimizer_output.get_render_list(input_str)
            index_list = optimizer_output.get_index_list(input_str)  # list of I-1 tensors of shape [B, V]

            # A non-empty index_list means this input was optimized on a different subset of its
            # views at each step (per-step view subsampling), so the renders don't cover the same
            # views every step. The default computation below assumes all views are present at every
            # step, so it only handles inputs rendered with all their views; skip subsampled inputs
            # unless the subset-aware path is explicitly enabled.
            if len(index_list) > 0 and not self.cfg.subset_aware:
                continue

            # Stack the I renders the optimizer kept for this input: the initial render followed by
            # one render per update step.
            predictions = torch.stack([render.color for render in render_list], dim=0)  # [I, B, V, C, H, W]
            gt = batch[input_str]["image"]  # [B, V_all, C, H, W]

            if len(index_list) == 0:
                # V == V_all
                # Compute l1 loss between predictions and gt for each iteration
                loss = torch.abs(predictions - gt).mean(dim=[3, 4, 5])  # [I, B, V]
                change_in_loss = loss[1:] - loss[:-1].detach()  # [I-1, B, V]
                change_in_loss = torch.relu(change_in_loss)  # Only consider increases in loss as contributing to the stability loss
            else:
                # Subset-aware path (subset_aware=True). NOTE: prototyped but not fully tested in
                # training.

                # One index tensor per render: duplicate the first step's indices to stand in for the
                # initial render, matching the I renders in `predictions`.
                assert len(index_list) == predictions.shape[0] - 1, (
                    f"stability loss expects one index entry per update step: "
                    f"{len(index_list)} indices vs {predictions.shape[0]} renders"
                )
                index_list = [index_list[0]] + index_list  # Now we have I tensors of shape [B, V]
                index_list = torch.stack(index_list, dim=0)  # [I, B, V]

                b = gt.shape[0]
                device = gt.device
                batch_idx = torch.arange(b, device=device)[None, :, None]  # [1, B, 1]
                gt_indexed = gt[batch_idx, index_list]  # [I, B, V, C, H, W]

                # Compute l1 loss between predictions and gt for each iteration
                # Consider the indexing of the views within the full batch
                loss = torch.abs(predictions - gt_indexed).mean(dim=[3, 4, 5])  # [I, B, V]

                # We want to make sure that the loss decreases across iterations for specific views
                I, B, V_all = predictions.shape[0], gt.shape[0], gt.shape[1]

                # Scatter losses into full view space
                # Don't use scatter_ in-place to enable backpropagation through the loss values
                loss_full = torch.zeros(I, B, V_all, device=loss.device).scatter(2, index_list, loss)  # [I, B, V_all]

                iter_idx = torch.arange(I, device=device).view(-1, 1, 1)  # [I,1,1]

                # mark unvisited as -1
                visited = loss_full > 0  # [I, B, V_all]
                visit_ids = torch.where(visited, iter_idx, torch.full_like(iter_idx, -1))  # [I, B, V]

                # running max gives last visit index
                last_visit = torch.cummax(visit_ids, dim=0).values  # [I,B,V]

                # shift to get strictly previous visit
                prev_visit = torch.roll(last_visit, shifts=1, dims=0)
                prev_visit[0] = -1  # first iter has no previous

                safe_prev = prev_visit.clamp(min=0)

                prev_loss = loss_full.gather(0, safe_prev).detach()

                has_prev = prev_visit >= 0

                change_in_loss = torch.relu(loss_full - prev_loss)
                change_in_loss = change_in_loss * has_prev.detach()

            # loss
            total_loss += change_in_loss.sum()
        return total_loss * self.cfg.weight
