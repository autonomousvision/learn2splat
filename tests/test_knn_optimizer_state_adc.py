"""Regression + policy tests for ``KnnBasedOptimizerState`` ADC mutations.

The learned optimizer carries a per-Gaussian latent ``state`` that adaptive density control
(clone/split/prune/relocate/add) must keep paired with the Gaussian it belongs to, AND initialize per
the configured policy (copy / zero / random). This policy governs the *learned latent state only* —
the Adam optimizer moments follow FastGS and are handled separately by AdamInputSmoothing.

Split is the subtle alignment case: the Gaussian tensors and the vanilla ``adc_state`` drop the split
parents and append ``S*N`` block-repeated children, so the latent state must use the same layout.
``test_split_state_aligns_with_gaussian_parents`` is the regression for a prior chunk-based split that
silently mis-paired state with Gaussians.
"""
import torch

from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.adc.base import _clone_objects, _split_objects
from learn2splat.scene_trainer.adc.vanilla import VanillaStrategyState, cloning, splitting
from learn2splat.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizerState


def _tagged_gaussians(n: int, sh_d: int = 1) -> Gaussians:
    """N Gaussians whose identity is recorded in ``harmonics[:, :, 0, 0] = index``. ``splitting()``
    and ``cloning()`` copy harmonics to children by exact block-repeat (only means/scales are
    perturbed), so this channel reports each output row's true parent after densification."""
    harmonics = torch.zeros(1, n, 3, sh_d)
    harmonics[0, :, 0, 0] = torch.arange(n).float()
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(1, n, 1).clone()
    return Gaussians(
        means=torch.arange(n).float().view(1, n, 1).repeat(1, 1, 3).clone(),
        harmonics=harmonics,
        opacities=torch.full((1, n), 0.5),
        scales=torch.full((1, n, 3), 0.1),
        rotations_unnorm=quat,
        rotations=quat.clone(),
        covariances=None,
        stores_activated=True,
    )


def _tracking_adc_state(n: int) -> VanillaStrategyState:
    """Vanilla adc_state whose grad accumulator is the identity tag; ``splitting``/``cloning`` keep
    it aligned with the Gaussians via ``_densification_postfix``, so it is the canonical anchor."""
    s = VanillaStrategyState.initialize(nr_points=n, device=torch.device("cpu"), scene_extent=1.0)
    s.grad2d_norm_accum = torch.arange(n).float()
    return s


class TestKnnStateSplitAlignment:
    """Default policy is "copy", so clone/split keep each latent-state row paired with its Gaussian."""

    def test_split_state_aligns_with_gaussian_parents(self):
        """Regression: after a real ``splitting()`` + ``_split_objects()`` with the same mask, every
        latent-state row belongs to the same parent as its Gaussian (and as the adc_state)."""
        torch.manual_seed(0)
        n = 4
        gaussians = _tagged_gaussians(n)
        adc = _tracking_adc_state(n)
        knn = KnnBasedOptimizerState(state=torch.arange(n).float().view(n, 1))
        split_mask = torch.tensor([False, True, True, False])  # split parents 1 and 2

        splitting(gaussians=gaussians, adc_state=adc, split_mask=split_mask, N=2)
        _split_objects(split_mask, {"optimizer_state": knn}, 2)

        parents = gaussians.harmonics[0, :, 0, 0]  # true parent of each output row
        torch.testing.assert_close(adc.grad2d_norm_accum, parents)  # canonical anchor
        torch.testing.assert_close(knn.state[:, 0], parents)        # latent state

    def test_split_layout_drops_parents_and_block_repeats(self):
        """Unit (copy): split keeps ``rest`` first, then appends ``num_splits`` block-repeated parent
        copies (the same ordering ``splitting()`` gives the Gaussian params)."""
        knn = KnnBasedOptimizerState(state=torch.arange(5).float().view(5, 1))
        split_mask = torch.tensor([False, True, False, True, False])  # parents {1, 3}
        knn.split(split_mask, num_splits=2)
        expected = torch.tensor([0, 2, 4, 1, 3, 1, 3]).float().view(-1, 1)  # rest + block[1,3]x2
        torch.testing.assert_close(knn.state, expected)

    def test_split_num_splits_three(self):
        """The append count scales with ``num_splits`` (the old chunk version always appended S)."""
        knn = KnnBasedOptimizerState(state=torch.arange(4).float().view(4, 1))
        split_mask = torch.tensor([False, True, True, False])  # S = 2 parents {1, 2}
        knn.split(split_mask, num_splits=3)
        assert knn.state.shape[0] == 2 + 2 * 3  # rest(2) + S*num_splits(6)
        torch.testing.assert_close(
            knn.state[:, 0], torch.tensor([0.0, 3.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        )

    def test_split_also_copies_init_state(self):
        """``init_state`` follows the identical layout."""
        knn = KnnBasedOptimizerState(
            state=torch.arange(4).float().view(4, 1),
            init_state=(torch.arange(4).float() * 10).view(4, 1),
        )
        split_mask = torch.tensor([False, True, True, False])
        knn.split(split_mask, num_splits=2)
        torch.testing.assert_close(knn.state[:, 0], torch.tensor([0.0, 3.0, 1.0, 2.0, 1.0, 2.0]))
        torch.testing.assert_close(
            knn.init_state[:, 0], torch.tensor([0.0, 30.0, 10.0, 20.0, 10.0, 20.0])
        )

    def test_clone_state_aligns_with_gaussian_parents(self):
        """Guard that clone stays aligned: full state + appended copies."""
        n = 4
        gaussians = _tagged_gaussians(n)
        adc = _tracking_adc_state(n)
        knn = KnnBasedOptimizerState(state=torch.arange(n).float().view(n, 1))
        clone_mask = torch.tensor([False, True, False, True])

        cloning(gaussians=gaussians, adc_state=adc, clone_mask=clone_mask)
        _clone_objects(clone_mask, {"optimizer_state": knn})

        parents = gaussians.harmonics[0, :, 0, 0]
        torch.testing.assert_close(adc.grad2d_norm_accum, parents)
        torch.testing.assert_close(knn.state[:, 0], parents)


class TestNewStatePolicy:
    """Per-op ``new_state_policy`` (copy | zero | random) governs the fresh latent state of densified
    Gaussians. "copy" inherits the parent/source; "zero" resets; "random" is randn * new_state_scale."""

    @staticmethod
    def _state(n=4, policy=None, scale=1.0):
        return KnnBasedOptimizerState(
            state=torch.arange(n * 3).float().view(n, 3),
            new_state_policy=policy or {},  # missing keys default to "copy"
            new_state_scale=scale,
        )

    def test_copy_inherits_parent(self):
        s = self._state(policy={"clone": "copy"})
        before = s.state.clone()
        s.clone(torch.tensor([False, True, False, True]))
        torch.testing.assert_close(s.state[4:], before[[1, 3]])  # copied parents

    def test_zero_resets(self):
        s = self._state(policy={"clone": "zero"})
        s.clone(torch.tensor([False, True, False, False]))
        assert torch.count_nonzero(s.state[4:]) == 0
        s = self._state(policy={"split": "zero"})
        s.split(torch.tensor([True, False, False, False]), num_splits=2)
        assert torch.count_nonzero(s.state[3:]) == 0  # rest=3 kept, 2 fresh zeros
        s = self._state(policy={"add": "zero"})
        s.add(3)
        assert torch.count_nonzero(s.state[4:]) == 0

    def test_random_per_op(self):
        torch.manual_seed(0)
        # clone: fresh rows random (non-zero, not the parent)
        s = self._state(policy={"clone": "random"}, scale=2.0)
        parent = s.state[[1, 3]].clone()
        s.clone(torch.tensor([False, True, False, True]))
        new = s.state[4:]
        assert new.shape == (2, 3) and torch.count_nonzero(new) > 0 and not torch.allclose(new, parent)
        # split
        s = self._state(n=6, policy={"split": "random"})
        s.split(torch.tensor([True, True, False, False, False, False]), num_splits=2)
        assert torch.count_nonzero(s.state[4:]) > 0  # rest=4, then 4 random rows
        # relocate
        s = self._state(n=4, policy={"relocate": "random"})
        s.replace(from_indices=torch.tensor([2]), dest_indices=torch.tensor([0]))
        assert torch.count_nonzero(s.state[0]) > 0
        # add
        s = self._state(n=4, policy={"add": "random"})
        s.add(3)
        assert torch.count_nonzero(s.state[4:]) > 0

    def test_default_policy_is_copy(self):
        # A state built with no explicit policy copies on clone (no surprise for plain callers).
        s = KnnBasedOptimizerState(state=torch.arange(6).float().view(2, 3))
        before = s.state.clone()
        s.clone(torch.tensor([True, False]))
        torch.testing.assert_close(s.state[2:], before[[0]])

    def test_random_block_matches_init_scale(self):
        torch.manual_seed(0)
        s = KnnBasedOptimizerState(state=torch.zeros(4000, 4),
                                   new_state_policy={"add": "random"}, new_state_scale=3.0)
        s.add(4000)
        new = s.state[4000:]
        assert abs(new.std().item() - 3.0) < 0.2
        assert abs(new.mean().item()) < 0.2


class TestPerOpStateConfig:
    """Schema: four per-op fields on the shared base; the old single field and the MCMC one are gone."""

    def test_per_op_fields_on_base(self):
        import dataclasses
        from learn2splat.scene_trainer.adc.base import BaseStrategyCfg
        names = {f.name for f in dataclasses.fields(BaseStrategyCfg)}
        for op in ("clone", "split", "relocate", "add"):
            assert f"{op}_gaussian_state" in names
        assert "new_gaussian_state" not in names  # replaced by the per-op fields

    def test_relocate_copy_state_removed_from_mcmc(self):
        import dataclasses
        from learn2splat.scene_trainer.adc.mcmc import McmcStrategyCfg
        assert "relocate_copy_state" not in {f.name for f in dataclasses.fields(McmcStrategyCfg)}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
