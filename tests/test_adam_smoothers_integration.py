"""Integration tests for ``AdamInputSmoothing`` in its two real call sites.

The existing ``test_adam_smoothers.py`` exercises the class in isolation. This file pins down
how it behaves when driven *exactly* the way the two optimizers drive it through adaptive density
control (ADC), since that is where alignment bugs hide:

1. **Adam baseline** (``optimizer_adam.py``): a ``dict`` of one ``AdamInputSmoothing`` per Gaussian
   parameter group (means/scales/rotations/opacities/sh0s/shNs), smoothed via
   ``optimizer_utils.smooth_grads`` and mutated through the generic ADC helpers in ``adc/base.py``
   (``_clone_objects`` / ``_split_objects`` / ``_prune_objects`` / ``_replace_objects`` /
   ``_add_to_objects``). The invariant under test: every group stays length-aligned and correct
   across clone/split/prune/relocate/add.

2. **Learned optimizer** (``optimizer_knn_based.py``): a *single* ``AdamInputSmoothing`` with
   ``input_slice=slice(-param_num, None)``, exposed to ADC as memory-sharing ``subgroups_view`` and
   stitched back with ``aggregate_from_subgroups`` (the round trip in ``_forward_impl``). The
   invariant under test: subgroup-ADC-then-aggregate is identical to applying the same ADC op to the
   whole state tensor — including for the strided ``sh0`` and fancy-indexed ``shN`` slices.
"""

import pytest
import torch

from learn2splat.scene_trainer.adc.base import (
    _add_to_objects,
    _clone_objects,
    _prune_objects,
    _replace_objects,
    _split_objects,
)
from learn2splat.scene_trainer.optimizer.layer import AdamInputSmoothing, AdamState
from learn2splat.scene_trainer.optimizer.optimizer_utils import (
    get_gaussian_param_sizes,
    get_gaussian_param_slices,
    smooth_grads,
)

BETAS = (0.9, 0.999)
EPS = 1e-8
SH_D = 4  # re10k-style SH degree; param_num = 11 + 3*SH_D = 23


# --------------------------------------------------------------------------------------------------
# Builders that mirror the two production call sites exactly.
# --------------------------------------------------------------------------------------------------
def _adam_group_shapes(n: int, sh_d: int, with_shN: bool) -> dict:
    """Per-group state shapes as built in ``AdamOptimizer._on_scene_start_impl`` (shape[1:] of the
    batched Gaussian params: opacities is [B, N] -> (N,), sh0s/shNs keep the (3, k) tail)."""
    return {
        "means": (n, 3),
        "scales": (n, 3),
        "rotations": (n, 4),
        "opacities": (n,),
        "sh0s": (n, 3, 1),
        "shNs": (n, 3, sh_d - 1) if with_shN else None,
    }


def _make_adam_baseline_smoothers(n: int, sh_d: int = SH_D, with_shN: bool = True) -> dict:
    """Mirror of the ``self.smoothers`` dict in ``optimizer_adam.py`` (one smoother per group,
    pre-initialized with ``shape=`` so ``is_reset()`` is already False)."""
    shapes = _adam_group_shapes(n, sh_d, with_shN)
    return {
        k: (None if s is None else AdamInputSmoothing(beta1=BETAS[0], beta2=BETAS[1], eps=EPS, shape=s))
        for k, s in shapes.items()
    }


def _drive_adam_baseline(smoothers: dict, n: int, steps: int, sh_d: int = SH_D, seed: int = 0) -> None:
    """Run a few ``smooth_grads`` iterations so the moments/timesteps are non-trivial, exactly as the
    Adam baseline does each optimization step (optimizer_adam.py: ``smooth_grads(grads, self.smoothers)``)."""
    g = torch.Generator().manual_seed(seed)
    has_shN = smoothers["shNs"] is not None
    for _ in range(steps):
        grads = {
            "means": torch.randn(n, 3, generator=g),
            "scales": torch.randn(n, 3, generator=g),
            "rotations": torch.randn(n, 4, generator=g),
            "opacities": torch.randn(n, generator=g),
            "sh0s": torch.randn(n, 3, 1, generator=g),
            "shNs": torch.randn(n, 3, sh_d - 1, generator=g) if has_shN else None,
        }
        smooth_grads(grads, smoothers)


def _make_input_norm(n: int, sh_d: int = SH_D, extra: int = 5, steps: int = 5, seed: int = 0):
    """Mirror of the learned optimizer's ``self.input_norm`` (``input_gradient_normalize_type=='adam'``):
    a single smoother over the trailing ``param_num`` channels, initialized as in
    ``_on_scene_start_impl`` and advanced a few steps. ``extra`` prefix channels stand in for the
    non-gradient part of the network input and must be left untouched by the slice."""
    param_num = sum(get_gaussian_param_sizes(sh_d).values())
    norm = AdamInputSmoothing(beta1=BETAS[0], beta2=BETAS[1], eps=EPS, input_slice=slice(-param_num, None))
    norm.initialize(shape=(n, param_num), device=torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        norm(torch.randn(n, extra + param_num, generator=g))
    return norm, param_num


def _snapshot(smoothers: dict) -> dict:
    return {
        k: None if s is None else (s.m.clone(), s.v.clone(), s.t.clone())
        for k, s in smoothers.items()
    }


# ==================================================================================================
# 1. Adam baseline: dict-of-smoothers driven through the generic ADC helpers.
# ==================================================================================================
class TestAdamBaselineSmootherDict:
    N = 12

    def test_drive_then_state_is_populated(self):
        """Sanity: after a few steps every group has matching leading dim N and t == steps."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=4)
        for k, s in sm.items():
            if s is None:
                continue
            assert s.m.shape[0] == self.N, k
            assert s.t.shape == (self.N,), k
            assert int(s.t[0]) == 4, k

    def test_clone_keeps_every_group_aligned(self):
        """``_clone_objects`` appends one copy per masked Gaussian to *every* group with the FastGS
        Adam behavior (the helper threads no policy): survivors untouched, new moment rows zeroed, and
        t copied from the cloned parents (step preserved, as in the reference)."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=5)
        before = _snapshot(sm)

        clone_mask = torch.zeros(self.N, dtype=torch.bool)
        clone_mask[[1, 4, 9]] = True
        k_new = int(clone_mask.sum())

        _clone_objects(clone_mask, sm)

        for key, s in sm.items():
            if s is None:
                continue
            m0, v0, t0 = before[key]
            assert s.m.shape[0] == self.N + k_new, key
            # survivors preserved
            torch.testing.assert_close(s.m[: self.N], m0)
            torch.testing.assert_close(s.v[: self.N], v0)
            torch.testing.assert_close(s.t[: self.N], t0)
            # cloned moment rows fresh; t copied from the cloned parents (FastGS)
            assert torch.count_nonzero(s.m[self.N :]) == 0, key
            assert torch.count_nonzero(s.v[self.N :]) == 0, key
            torch.testing.assert_close(s.t[self.N :], t0[clone_mask])

    def test_split_drops_parents_and_appends_N_per_group(self):
        """``_split_objects`` uses the canonical 3DGS layout (drop the split parents, append ``S*N``
        block-repeated rows) with the FastGS Adam behavior: new moment rows zeroed, t block-repeated
        from the parents (step preserved). Same ordering the Gaussian tensors / adc_state use."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=6)
        before = _snapshot(sm)

        split_mask = torch.zeros(self.N, dtype=torch.bool)
        split_mask[[2, 7]] = True
        s_cnt = int(split_mask.sum())
        N = 2
        rest = ~split_mask
        expected_len = self.N - s_cnt + s_cnt * N

        _split_objects(split_mask, sm, N)

        for key, s in sm.items():
            if s is None:
                continue
            m0, v0, t0 = before[key]
            assert s.m.shape[0] == expected_len, key
            # the kept (rest) rows come first, unchanged
            torch.testing.assert_close(s.m[: int(rest.sum())], m0[rest])
            torch.testing.assert_close(s.v[: int(rest.sum())], v0[rest])
            torch.testing.assert_close(s.t[: int(rest.sum())], t0[rest])
            # appended split rows: moments zero; t block-repeated from the parents (FastGS)
            tail = slice(int(rest.sum()), None)
            assert torch.count_nonzero(s.m[tail]) == 0, key
            assert torch.count_nonzero(s.v[tail]) == 0, key
            torch.testing.assert_close(s.t[tail], t0[split_mask].repeat(N))

    def test_prune_removes_same_rows_from_all_groups(self):
        """``_prune_objects`` drops the masked rows from every group, keeping the survivors aligned."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=3)
        before = _snapshot(sm)

        prune_mask = torch.zeros(self.N, dtype=torch.bool)
        prune_mask[[0, 5, 11]] = True
        keep = ~prune_mask

        _prune_objects(prune_mask, sm)

        for key, s in sm.items():
            if s is None:
                continue
            m0, v0, t0 = before[key]
            assert s.m.shape[0] == int(keep.sum()), key
            torch.testing.assert_close(s.m, m0[keep])
            torch.testing.assert_close(s.v, v0[keep])
            torch.testing.assert_close(s.t, t0[keep])

    def test_replace_matches_mcmc_relocate(self):
        """``_replace_objects(dead, alive, ...)`` is the MCMC relocate path with the FastGS Adam
        behavior: dead Gaussians inherit the sampled-alive moments and timestep. Length unchanged."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=4)
        before = _snapshot(sm)

        dead = torch.tensor([0, 3])
        alive = torch.tensor([8, 5])  # source rows the dead ones are relocated onto

        # adc/base signature: _replace_objects(dest_indices, from_indices, objects)
        _replace_objects(dead, alive, sm)

        for key, s in sm.items():
            if s is None:
                continue
            m0, v0, t0 = before[key]
            assert s.m.shape[0] == self.N, key  # relocate never changes the count
            torch.testing.assert_close(s.m[dead], m0[alive])
            torch.testing.assert_close(s.v[dead], v0[alive])
            torch.testing.assert_close(s.t[dead], t0[alive])
            # an untouched row stays put
            assert int(s.t[1]) == int(t0[1])

    def test_add_matches_mcmc_add_new(self):
        """``_add_to_objects`` (MCMC add_new) appends ``nr_new`` all-zero rows to every group."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=2)
        nr_new = 4

        _add_to_objects(nr_new, sm)

        for key, s in sm.items():
            if s is None:
                continue
            assert s.m.shape[0] == self.N + nr_new, key
            assert torch.count_nonzero(s.m[self.N :]) == 0, key
            assert torch.count_nonzero(s.v[self.N :]) == 0, key
            assert torch.count_nonzero(s.t[self.N :]) == 0, key

    def test_none_group_is_skipped_by_all_helpers(self):
        """When SH degree is 1 the ``shNs`` smoother is ``None`` (as in optimizer_adam). Every ADC
        helper must skip it without error and keep the real groups length-consistent."""
        sm = _make_adam_baseline_smoothers(self.N, with_shN=False)
        assert sm["shNs"] is None
        _drive_adam_baseline(sm, self.N, steps=2)

        def mask(size, idx):
            m = torch.zeros(size, dtype=torch.bool)
            m[idx] = True
            return m

        n = self.N  # track the count so each mask matches the current size
        _clone_objects(mask(n, 3), sm)
        n += 1
        _split_objects(mask(n, 2), sm, 2)
        n += 1  # split: -1 parent, +2 children
        _prune_objects(mask(n, 0), sm)
        n -= 1
        _add_to_objects(2, sm)
        n += 2
        _replace_objects(torch.tensor([0]), torch.tensor([1]), sm)

        assert sm["shNs"] is None
        lengths = {k: s.m.shape[0] for k, s in sm.items() if s is not None}
        assert set(lengths.values()) == {n}, (lengths, n)

    def test_forward_still_works_after_densification(self):
        """After clone via the helper the per-group smoother is still a valid optimizer state: the
        next ``smooth_grads`` step runs on the grown size, advances every t, and stays finite. Under
        FastGS the cloned rows carry zeroed moments and the parent's step, so all t advance together."""
        sm = _make_adam_baseline_smoothers(self.N)
        _drive_adam_baseline(sm, self.N, steps=5)

        clone_mask = torch.zeros(self.N, dtype=torch.bool)
        clone_mask[[1, 2]] = True
        _clone_objects(clone_mask, sm)
        n2 = self.N + int(clone_mask.sum())

        g = torch.Generator().manual_seed(123)
        grads = {
            "means": torch.randn(n2, 3, generator=g),
            "scales": torch.randn(n2, 3, generator=g),
            "rotations": torch.randn(n2, 4, generator=g),
            "opacities": torch.randn(n2, generator=g),
            "sh0s": torch.randn(n2, 3, 1, generator=g),
            "shNs": torch.randn(n2, 3, SH_D - 1, generator=g),
        }
        out = smooth_grads(grads, sm)
        for key, o in out.items():
            if o is None:
                continue
            assert o.shape[0] == n2, key
            assert torch.isfinite(o).all(), key
        # survivors were at t=5, cloned rows inherited the parent's t=5 -> all advance to 6
        assert int(sm["means"].t[0]) == 6
        assert int(sm["means"].t[-1]) == 6


# ==================================================================================================
# 2. Learned optimizer: single sliced smoother, mutated via subgroups_view + aggregate_from_subgroups.
# ==================================================================================================
class TestLearnedOptimizerInputNorm:
    N = 16

    def test_subgroup_views_share_contiguous_slices_but_not_fancy_index(self):
        """``subgroups_view`` returns memory-sharing views for contiguous/strided slices
        (means, sh0) and a *copy* for the fancy-indexed ``shN`` group. This is precisely why
        ``aggregate_from_subgroups`` is required to write the subgroup state back."""
        norm, _ = _make_input_norm(self.N)
        slices = get_gaussian_param_slices(SH_D)
        subs = norm.subgroups_view(slices)

        # contiguous slice -> shared storage
        before_means = norm.m[..., slices["means"]].clone()
        subs["means"].m += 1.0
        torch.testing.assert_close(norm.m[..., slices["means"]], before_means + 1.0)

        # strided slice (sh0 = slice(11, 23, 4)) -> still a view
        before_sh0 = norm.m[..., slices["sh0"]].clone()
        subs["sh0"].m += 2.0
        torch.testing.assert_close(norm.m[..., slices["sh0"]], before_sh0 + 2.0)

        # fancy-indexed slice (shN = list of ints) -> independent copy, parent unchanged
        before_shN = norm.m[..., slices["shN"]].clone()
        subs["shN"].m += 3.0
        torch.testing.assert_close(norm.m[..., slices["shN"]], before_shN)

    def test_clone_via_subgroups_equals_whole_tensor(self):
        self._assert_subgroup_adc_matches_whole("clone")

    def test_split_via_subgroups_equals_whole_tensor(self):
        self._assert_subgroup_adc_matches_whole("split")

    def test_prune_via_subgroups_equals_whole_tensor(self):
        self._assert_subgroup_adc_matches_whole("prune")

    def _assert_subgroup_adc_matches_whole(self, op: str):
        """The core guarantee of the learned-optimizer path: applying an ADC op to the per-group
        ``subgroups_view`` and then ``aggregate_from_subgroups`` reproduces, bit for bit, applying the
        same op directly to the whole [N, param_num] state — across the strided/fancy SH slices. The
        helpers drive the subgroups with the FastGS default, so the reference uses the same."""
        ref, param_num = _make_input_norm(self.N, seed=7)
        test, _ = _make_input_norm(self.N, seed=7)  # identical state (same seed)
        # guard: the two builders really did produce identical starting state
        torch.testing.assert_close(ref.m, test.m)

        slices = get_gaussian_param_slices(SH_D)

        if op == "clone":
            mask = torch.zeros(self.N, dtype=torch.bool)
            mask[[2, 5, 10]] = True
            ref.clone(mask)
            subs = test.subgroups_view(slices)
            _clone_objects(mask, subs)
        elif op == "split":
            mask = torch.zeros(self.N, dtype=torch.bool)
            mask[[3, 8]] = True
            ref.split(mask, 2)
            subs = test.subgroups_view(slices)
            _split_objects(mask, subs, 2)
        elif op == "prune":
            mask = torch.zeros(self.N, dtype=torch.bool)
            mask[[0, 4, 15]] = True
            ref.prune(mask)
            subs = test.subgroups_view(slices)
            _prune_objects(mask, subs)
        else:
            raise ValueError(op)

        test.aggregate_from_subgroups(subs, slices)

        assert test.m.shape == ref.m.shape
        torch.testing.assert_close(test.m, ref.m)
        torch.testing.assert_close(test.v, ref.v)
        torch.testing.assert_close(test.t, ref.t)

    def test_aggregate_is_required_after_subgroup_adc(self):
        """Documents the contract: ``clone``/``split`` on a subgroup *reassign* its ``.m``/``.v``,
        breaking the shared view, so the parent stays stale until ``aggregate_from_subgroups`` runs."""
        norm, param_num = _make_input_norm(self.N)
        slices = get_gaussian_param_slices(SH_D)
        subs = norm.subgroups_view(slices)

        mask = torch.zeros(self.N, dtype=torch.bool)
        mask[1] = True
        _clone_objects(mask, subs)

        # Parent is still the old size right after the subgroup mutation...
        assert norm.m.shape[0] == self.N
        norm.aggregate_from_subgroups(subs, slices)
        # ...and only now reflects the grown count.
        assert norm.m.shape[0] == self.N + 1
        assert norm.m.shape[-1] == param_num

    def test_get_state_update_state_roundtrip(self):
        """Ckpt-buffer path (``_on_scene_start_impl``: ``get_state`` on save,
        ``update_state(adam_state)`` on resume) preserves m/v/t exactly."""
        norm, _ = _make_input_norm(self.N, steps=4)
        state = norm.get_state()
        assert isinstance(state, AdamState)

        restored = AdamInputSmoothing(beta1=BETAS[0], beta2=BETAS[1], eps=EPS,
                                      input_slice=slice(-norm.m.shape[-1], None))
        restored.update_state(state)
        torch.testing.assert_close(restored.m, norm.m)
        torch.testing.assert_close(restored.v, norm.v)
        torch.testing.assert_close(restored.t, norm.t)

    def test_forward_after_densification_smooths_only_the_slice(self):
        """After a subgroup clone (+aggregate) the grown smoother still smooths only its trailing
        ``param_num`` slice, leaves the prefix channels untouched, and stays finite (FastGS: the new
        rows carry zeroed moments with the parent's step)."""
        norm, param_num = _make_input_norm(self.N, extra=5, steps=5)
        slices = get_gaussian_param_slices(SH_D)
        subs = norm.subgroups_view(slices)

        mask = torch.zeros(self.N, dtype=torch.bool)
        mask[[0, 1]] = True
        _clone_objects(mask, subs)
        norm.aggregate_from_subgroups(subs, slices)

        n2 = self.N + int(mask.sum())
        g = torch.Generator().manual_seed(99)
        x = torch.randn(n2, 5 + param_num, generator=g)
        out = norm(x)

        assert out.shape == x.shape
        torch.testing.assert_close(out[..., :5], x[..., :5])  # prefix passes through untouched
        assert torch.isfinite(out[..., 5:]).all()             # smoothed slice is finite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
