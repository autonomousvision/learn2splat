import math

import pytest
import torch

from learn2splat.loss import LOSSES, get_losses, LossGaussiansCfgWrapper
from learn2splat.loss.loss_deltas import LossDeltas, LossDeltasCfg, LossDeltasCfgWrapper
from learn2splat.loss.loss_gaussians import LossGaussiansCfg, LossGaussians
from learn2splat.loss.loss_iso_scales import LossIsoScales, LossIsoScalesCfg, LossIsoScalesCfgWrapper
from learn2splat.loss.loss_mse import LossMse, LossMseCfg, LossMseCfgWrapper
from learn2splat.loss.loss_render_depth import LossRenderDepth, LossRenderDepthCfg, LossRenderDepthCfgWrapper
from learn2splat.loss.loss_sgd import LossSGD, LossSGDCfg, LossSGDCfgWrapper
from learn2splat.loss.loss_ssim import LossSsim, LossSsimCfg, LossSsimCfgWrapper
from learn2splat.loss.loss_stability import LossStability, LossStabilityCfg, LossStabilityCfgWrapper
from learn2splat.misc.detaching_cpu_list import DetachingCPUList
from learn2splat.model.decoder.decoder import DecoderOutput
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.optimizer.optimizer import OptimizerOutput

B, G, V, C, H, W = 1, 64, 2, 3, 32, 32
D_SH = 16  # degree 0-3


def make_gaussians(d_sh=D_SH, device="cpu"):
    return Gaussians(
        means=torch.randn(B, G, 3, device=device),
        harmonics=torch.randn(B, G, 3, d_sh, device=device),
        opacities=torch.rand(B, G, device=device),
        scales=torch.rand(B, G, 3, device=device) + 0.01,
        rotations_unnorm=torch.randn(B, G, 4, device=device),
    )


def make_prediction(device="cpu"):
    return DecoderOutput(
        color=torch.rand(B, V, C, H, W, device=device),
        depth=torch.rand(B, V, H, W, device=device),
    )


def make_batch(device="cpu"):
    views = {
        "image": torch.rand(B, V, C, H, W, device=device),
        "extrinsics": torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, 4, 4),
        "intrinsics": torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, 3, 3),
        "near": torch.tensor([0.1], device=device).expand(B, V),
        "far": torch.tensor([100.0], device=device).expand(B, V),
    }
    return {"context": views, "target": {**views}}


def loss_value(result):
    """Read a loss as a float. LossGaussians returns a plain 0 (not a tensor) when no weighted
    term contributes (all-zero weights, or an SH-only loss with d_sh=1), so handle both cases."""
    return result.item() if torch.is_tensor(result) else float(result)


def run_loss(loss_fn, prediction, batch, gaussians, global_step=0, **extra):
    """Call a per-step loss with the production convention (see MetaTrainer.compute_losses):
    losses receive the target-view GT (gt_rgb) and rendered (pred_rgb) colors as tensors, not the batch."""
    return loss_fn(
        prediction, gaussians, global_step,
        gt_rgb=batch["target"]["image"], pred_rgb=prediction.color,
        valid_depth_mask=None, **extra,
    )


# Config factories: every loss cfg field is required (no dataclass defaults), so build them
# with sensible defaults here and let each test override only the fields it exercises.


def mse_cfg(weight=1.0, l1_loss=False, clamp_large_error=0.0):
    return LossMseCfgWrapper(mse=LossMseCfg(
        weight=weight, l1_loss=l1_loss, clamp_large_error=clamp_large_error))


def gaussians_cfg(weight=1.0, weight_scales=1.0, weight_opacities=1.0, weight_sh=1.0, sh_alpha=1.0):
    return LossGaussiansCfgWrapper(gaussians=LossGaussiansCfg(
        weight=weight, weight_scales=weight_scales, weight_opacities=weight_opacities,
        weight_sh=weight_sh, sh_alpha=sh_alpha))


def iso_cfg(weight=1.0):
    return LossIsoScalesCfgWrapper(iso_scales=LossIsoScalesCfg(weight=weight))


def sgd_cfg(l1_loss=False, clamp_large_error=0.0):
    return LossSGDCfgWrapper(sgd=LossSGDCfg(l1_loss=l1_loss, clamp_large_error=clamp_large_error))


def ssim_cfg(weight=1.0):
    return LossSsimCfgWrapper(ssim=LossSsimCfg(weight=weight))


def stability_cfg(weight=1.0, subset_aware=False):
    return LossStabilityCfgWrapper(stability=LossStabilityCfg(weight=weight, subset_aware=subset_aware))


def _sh_only_cfg(**kwargs):
    return gaussians_cfg(weight_scales=0.0, weight_opacities=0.0, **kwargs)


# ---------- Registry tests ----------


class TestLossRegistry:
    def test_all_losses_registered(self):
        assert len(LOSSES) >= 9

    def test_get_losses(self):
        wrappers = [
            mse_cfg(weight=1.0),
            gaussians_cfg(),
        ]
        losses = get_losses(wrappers)
        assert len(losses) == 2
        assert isinstance(losses[0], LossMse)
        assert isinstance(losses[1], LossGaussians)


# ---------- LossGaussians tests ----------


class TestLossGaussians:
    def test_forward_returns_scalar(self):
        loss_fn = LossGaussians(gaussians_cfg())
        result = run_loss(loss_fn, make_prediction(), make_batch(), make_gaussians())
        assert result.shape == ()
        assert result.item() > 0

    def test_independent_weights(self):
        gaussians = make_gaussians()

        only_scales = LossGaussians(gaussians_cfg(weight_scales=1.0, weight_opacities=0.0, weight_sh=0.0))
        only_opacities = LossGaussians(gaussians_cfg(weight_scales=0.0, weight_opacities=1.0, weight_sh=0.0))
        only_sh = LossGaussians(gaussians_cfg(weight_scales=0.0, weight_opacities=0.0, weight_sh=1.0))
        all_zero = LossGaussians(gaussians_cfg(weight_scales=0.0, weight_opacities=0.0, weight_sh=0.0))

        pred, batch = make_prediction(), make_batch()
        ls = run_loss(only_scales, pred, batch, gaussians)
        lo = run_loss(only_opacities, pred, batch, gaussians)
        lsh = run_loss(only_sh, pred, batch, gaussians)
        lz = run_loss(all_zero, pred, batch, gaussians)

        assert ls.item() > 0
        assert lo.item() > 0
        assert lsh.item() > 0
        assert loss_value(lz) == 0.0

        # Sum of individual components should equal all-enabled loss
        all_one = LossGaussians(gaussians_cfg())
        ltotal = run_loss(all_one, pred, batch, gaussians)
        torch.testing.assert_close(ltotal, ls + lo + lsh, atol=1e-6, rtol=1e-5)

    def test_weight_scaling(self):
        gaussians = make_gaussians()
        pred, batch = make_prediction(), make_batch()

        loss1 = LossGaussians(gaussians_cfg(weight_scales=1.0, weight_opacities=0.0, weight_sh=0.0))
        loss2 = LossGaussians(gaussians_cfg(weight_scales=2.0, weight_opacities=0.0, weight_sh=0.0))

        r1 = run_loss(loss1, pred, batch, gaussians)
        r2 = run_loss(loss2, pred, batch, gaussians)
        torch.testing.assert_close(r2, r1 * 2, atol=1e-6, rtol=1e-5)

    def test_sh_excludes_degree_zero(self):
        gaussians = make_gaussians(d_sh=1)  # only degree 0
        loss_fn = LossGaussians(_sh_only_cfg(weight_sh=1.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert loss_value(result) == 0.0

    def test_requires_gaussians(self):
        loss_fn = LossGaussians(gaussians_cfg())
        with pytest.raises(ValueError, match="LossGaussians"):
            run_loss(loss_fn, make_prediction(), make_batch(), None)


# ---------- sh_alpha (per-degree SH weighting) tests ----------


class TestShAlpha:
    def test_alpha_1_is_uniform(self):
        """With alpha=1.0 (default), all degrees get equal weight."""
        gaussians = make_gaussians()
        pred, batch = make_prediction(), make_batch()

        loss_a1 = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=1.0))
        loss_a2 = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=2.0))

        r1 = run_loss(loss_a1, pred, batch, gaussians)
        r2 = run_loss(loss_a2, pred, batch, gaussians)
        assert r1.item() != r2.item()

    def test_alpha_scaling_increases_loss(self):
        """Higher alpha should increase the loss when higher degrees have large coefficients."""
        gaussians = make_gaussians()
        gaussians.harmonics[..., 9:16] = 5.0  # make degree 3 large
        pred, batch = make_prediction(), make_batch()

        loss_a1 = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=1.0))
        loss_a2 = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=2.0))

        r1 = run_loss(loss_a1, pred, batch, gaussians)
        r2 = run_loss(loss_a2, pred, batch, gaussians)
        assert r2.item() > r1.item()

    def test_per_degree_weighting_manual(self):
        """Verify exact per-degree weighting with known values."""
        gaussians = make_gaussians(d_sh=4)  # degrees 0-1
        gaussians.harmonics.zero_()
        gaussians.harmonics[..., 1:4] = 1.0  # degree 1 only

        loss_fn = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=3.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)

        # weight_sh * alpha^1 * mean(1.0^2) = 1.0 * 3.0 * 1.0 = 3.0
        torch.testing.assert_close(result, torch.tensor(3.0), atol=1e-6, rtol=1e-5)

    def test_degree_zero_excluded_with_alpha(self):
        gaussians = make_gaussians(d_sh=1)
        loss_fn = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=2.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert loss_value(result) == 0.0

    def test_different_sh_sizes(self):
        """Test with d_sh = 1, 4, 9, 16."""
        loss_fn = LossGaussians(_sh_only_cfg(weight_sh=1.0, sh_alpha=2.0))
        pred, batch = make_prediction(), make_batch()

        for d_sh in [1, 4, 9, 16]:
            gaussians = make_gaussians(d_sh=d_sh)
            result = run_loss(loss_fn, pred, batch, gaussians)
            if d_sh == 1:
                assert loss_value(result) == 0.0
            else:
                assert result.shape == ()
                assert result.item() > 0


# ---------- LossIsoScales tests ----------


class TestLossIsoScales:
    def test_forward_returns_scalar(self):
        loss_fn = LossIsoScales(iso_cfg(weight=1.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), make_gaussians())
        assert result.shape == ()

    def test_isotropic_scales_zero_loss(self):
        gaussians = make_gaussians()
        gaussians.scales = torch.ones(B, G, 3) * 0.5  # perfectly isotropic
        loss_fn = LossIsoScales(iso_cfg(weight=1.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


# ---------- LossMse tests ----------


class TestLossMse:
    def test_forward_returns_scalar(self):
        loss_fn = LossMse(mse_cfg(weight=1.0))
        result = run_loss(loss_fn, make_prediction(), make_batch(), make_gaussians())
        assert result.shape == ()
        assert result.item() > 0

    def test_perfect_prediction_zero_loss(self):
        batch = make_batch()
        pred = DecoderOutput(
            color=batch["target"]["image"].clone(),
            depth=None,
        )
        loss_fn = LossMse(mse_cfg(weight=1.0))
        result = run_loss(loss_fn, pred, batch, None)
        assert result.item() == pytest.approx(0.0, abs=1e-6)


# ---------- LossDeltas tests ----------


class TestLossDeltas:
    def test_forward_returns_scalar(self):
        cfg = LossDeltasCfg(
            weight=1.0, exclude_by_norm_grad=False,
            exclude_by_norm_grad_opposite=False, eps=0.1, apply_after_step=0,
        )
        loss_fn = LossDeltas(LossDeltasCfgWrapper(deltas=cfg))
        gaussians = make_gaussians()
        gaussians.deltas = torch.randn(B, G, 10)
        gaussians.gradients = torch.randn(B, G, 10)
        gaussians.norm_gradients = torch.randn(B, G, 10)
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.shape == ()

    def test_before_apply_step_returns_zero(self):
        cfg = LossDeltasCfg(
            weight=1.0, exclude_by_norm_grad=False,
            exclude_by_norm_grad_opposite=False, eps=0.1, apply_after_step=100,
        )
        loss_fn = LossDeltas(LossDeltasCfgWrapper(deltas=cfg))
        gaussians = make_gaussians()
        gaussians.deltas = torch.randn(B, G, 10)
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.item() == 0.0


# ---------- LossSsim tests ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused_ssim requires CUDA")
class TestLossSsim:
    def test_forward_returns_scalar(self):
        loss_fn = LossSsim(ssim_cfg(weight=1.0))
        result = run_loss(loss_fn, make_prediction("cuda"), make_batch("cuda"), None)
        assert result.shape == ()
        assert result.item() > 0

    def test_perfect_prediction_zero_loss(self):
        batch = make_batch("cuda")
        pred = DecoderOutput(color=batch["target"]["image"].clone(), depth=None)
        loss_fn = LossSsim(ssim_cfg(weight=1.0))
        result = run_loss(loss_fn, pred, batch, None)
        assert result.item() == pytest.approx(0.0, abs=1e-4)

    def test_weight_scaling(self):
        pred, batch = make_prediction("cuda"), make_batch("cuda")
        loss1 = LossSsim(ssim_cfg(weight=1.0))
        loss2 = LossSsim(ssim_cfg(weight=2.0))
        r1 = run_loss(loss1, pred, batch, None)
        r2 = run_loss(loss2, pred, batch, None)
        torch.testing.assert_close(r2, r1 * 2, atol=1e-6, rtol=1e-5)


# ---------- LossSGD tests ----------


class TestLossSGD:
    def test_forward_returns_scalar(self):
        loss_fn = LossSGD(sgd_cfg())
        gaussians = make_gaussians()
        gaussians.deltas = torch.randn(B, G, 10)
        gaussians.gradients = torch.randn(B, G, 10)
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.shape == ()
        assert result.item() > 0

    def test_perfect_match_zero_loss(self):
        loss_fn = LossSGD(sgd_cfg())
        gaussians = make_gaussians()
        grad = torch.randn(B, G, 10)
        gaussians.deltas = grad.clone()
        gaussians.gradients = grad.clone()
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_l1_mode(self):
        loss_fn = LossSGD(sgd_cfg(l1_loss=True))
        gaussians = make_gaussians()
        gaussians.deltas = torch.randn(B, G, 10)
        gaussians.gradients = torch.randn(B, G, 10)
        result = run_loss(loss_fn, make_prediction(), make_batch(), gaussians)
        assert result.shape == ()
        assert result.item() > 0


# ---------- LossStability tests ----------


class TestLossStability:
    @staticmethod
    def _make_optimizer_output(num_iters=3):
        """Create a minimal OptimizerOutput with render lists."""
        ctx_renders = DetachingCPUList()
        tgt_renders = DetachingCPUList()
        for _ in range(num_iters):
            ctx_renders.append(DecoderOutput(
                color=torch.rand(B, V, C, H, W), depth=None))
            tgt_renders.append(DecoderOutput(
                color=torch.rand(B, V, C, H, W), depth=None))
        return OptimizerOutput(
            gaussian_list=DetachingCPUList(),
            context_render_list=ctx_renders,
            target_render_list=tgt_renders,
        )

    def test_forward_returns_scalar(self):
        loss_fn = LossStability(stability_cfg(weight=1.0))
        opt_output = self._make_optimizer_output()
        batch = make_batch()
        result = loss_fn(opt_output, batch)
        assert result.shape == ()

    def test_no_change_zero_loss(self):
        """Identical renders across iterations should produce zero stability loss."""
        loss_fn = LossStability(stability_cfg(weight=1.0))
        batch = make_batch()
        # All iterations render the exact GT
        ctx_renders = DetachingCPUList()
        tgt_renders = DetachingCPUList()
        for _ in range(3):
            ctx_renders.append(DecoderOutput(
                color=batch["context"]["image"].clone(), depth=None))
            tgt_renders.append(DecoderOutput(
                color=batch["target"]["image"].clone(), depth=None))
        opt_output = OptimizerOutput(
            gaussian_list=DetachingCPUList(),
            context_render_list=ctx_renders,
            target_render_list=tgt_renders,
        )
        result = loss_fn(opt_output, batch)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_weight_scaling(self):
        opt_output = self._make_optimizer_output()
        batch = make_batch()
        loss1 = LossStability(stability_cfg(weight=1.0))
        loss2 = LossStability(stability_cfg(weight=2.0))
        r1 = loss1(opt_output, batch)
        r2 = loss2(opt_output, batch)
        torch.testing.assert_close(r2, r1 * 2, atol=1e-6, rtol=1e-5)

    def test_subset_aware_skips_subsampled_by_default(self):
        """With per-step view subsampling and subset_aware off (default), subsampled inputs are
        skipped. make_batch shares one GT across context/target, so with identical renders the two
        inputs contribute equally: subsampling context only must halve the loss (target survives)."""
        batch = make_batch()  # context and target reference the same GT image
        renders = DetachingCPUList()
        for _ in range(3):
            renders.append(DecoderOutput(color=torch.rand(B, V, C, H, W), depth=None))
        kwargs = dict(gaussian_list=DetachingCPUList(),
                      context_render_list=renders, target_render_list=renders)
        loss_fn = LossStability(stability_cfg(weight=1.0, subset_aware=False))
        both = loss_fn(OptimizerOutput(**kwargs), batch)
        # Subsample context only (mirrors the standard config where target stays full-view).
        target_only = loss_fn(OptimizerOutput(
            **kwargs, context_index_list=[torch.randint(0, V, (B, V)) for _ in range(2)]), batch)
        torch.testing.assert_close(target_only * 2, both, atol=1e-6, rtol=1e-5)

    def test_subset_aware_runs_and_backprops(self):
        """subset_aware=True runs the per-step-subset path and produces gradients. Renders are the
        per-step subset (V dim == subset size) aligned with index_list, as produced during training."""
        nv, num_steps = V, 3  # subset size; index_list has one entry per update step
        net = torch.nn.Conv2d(C, C, 1)

        def subset_traj():
            idx = [torch.randint(0, V, (B, nv)) for _ in range(num_steps)]
            renders = DetachingCPUList()
            cur = torch.rand(B, nv, C, H, W)
            renders.append(DecoderOutput(color=cur, depth=None))  # initial render
            for _ in range(num_steps):
                cur = cur + net(cur.view(B * nv, C, H, W)).view(B, nv, C, H, W)
                renders.append(DecoderOutput(color=cur, depth=None))
            return renders, idx

        ctx_renders, ctx_idx = subset_traj()
        tgt_renders, tgt_idx = subset_traj()
        opt_output = OptimizerOutput(
            gaussian_list=DetachingCPUList(),
            context_render_list=ctx_renders, target_render_list=tgt_renders,
            context_index_list=ctx_idx, target_index_list=tgt_idx)
        result = LossStability(stability_cfg(weight=1.0, subset_aware=True))(opt_output, make_batch())
        assert result.shape == ()
        result.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


# ---------- LossLpips tests ----------
# LossLpips requires downloading VGG weights; tested separately if available.


class TestLossLpips:
    @pytest.fixture(autouse=True)
    def _load_lpips(self):
        try:
            from learn2splat.loss.loss_lpips import LossLpips, LossLpipsCfg, LossLpipsCfgWrapper
            self.LossLpips = LossLpips
            self.LossLpipsCfg = LossLpipsCfg
            self.LossLpipsCfgWrapper = LossLpipsCfgWrapper
        except Exception:
            pytest.skip("LPIPS dependencies not available")

    def test_forward_returns_scalar(self):
        cfg = self.LossLpipsCfg(weight=1.0, apply_after_step=0, perceptual_loss=True, half_res=False)
        loss_fn = self.LossLpips(self.LossLpipsCfgWrapper(lpips=cfg))
        result = run_loss(loss_fn, make_prediction(), make_batch(), None)
        assert result.shape == ()
        assert result.item() > 0

    def test_before_apply_step_returns_zero(self):
        cfg = self.LossLpipsCfg(weight=1.0, apply_after_step=100, perceptual_loss=True, half_res=False)
        loss_fn = self.LossLpips(self.LossLpipsCfgWrapper(lpips=cfg))
        result = run_loss(loss_fn, make_prediction(), make_batch(), None)
        assert result.item() == 0.0


# ---------- LossRenderDepth tests ----------


class TestLossRenderDepth:
    def _make_loss(self, weight=1.0):
        return LossRenderDepth(LossRenderDepthCfgWrapper(render_depth=LossRenderDepthCfg(weight=weight)))

    def _call(self, loss_fn, prediction, gt_depth, near, far, depth_mask=None):
        # near/far arrive as [B, V, 1, 1] (matching the MetaTrainer pre-broadcast shape).
        return loss_fn(prediction, None, 0, gt_depth=gt_depth, depth_mask=depth_mask, near=near, far=far)

    def _near_far(self, near=0.1, far=100.0):
        return (torch.full((B, V, 1, 1), near),
                torch.full((B, V, 1, 1), far))

    def test_use_in_inner_steps_is_true(self):
        assert LossRenderDepth.use_in_inner_steps is True

    def test_returns_scalar(self):
        render_depth = torch.rand(B, V, H, W) * 9 + 1.0  # in (1, 10), safely within near/far
        gt_depth = torch.rand(B, V, H, W) * 9 + 1.0
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(), pred, gt_depth, *self._near_far())
        assert result.shape == ()
        assert result.item() > 0

    def test_depth_none_returns_zero(self):
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=None)
        gt_depth = torch.ones(B, V, H, W) * 5.0
        result = self._call(self._make_loss(), pred, gt_depth, *self._near_far())
        assert result.shape == ()
        assert result.item() == 0.0

    def test_empty_valid_mask_returns_zero(self):
        # gt_depth is below near everywhere → valid mask is empty.
        near, far = self._near_far(near=10.0, far=100.0)
        gt_depth = torch.ones(B, V, H, W) * 1.0  # < near
        render_depth = torch.ones(B, V, H, W) * 5.0
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(), pred, gt_depth, near, far)
        assert result.shape == ()
        assert result.item() == 0.0

    def test_exact_log_l1_value(self):
        # gt=e^1, render=e^2, all pixels valid → |log(e) - log(e^2)| = |1 - 2| = 1.
        gt_depth = torch.full((B, V, H, W), math.exp(1))
        render_depth = torch.full((B, V, H, W), math.exp(2))
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(weight=1.0), pred, gt_depth, *self._near_far())
        assert result.item() == pytest.approx(1.0, abs=1e-5)

    def test_weight_scaling(self):
        gt_depth = torch.rand(B, V, H, W) * 9 + 1.0
        render_depth = torch.rand(B, V, H, W) * 9 + 1.0
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        near, far = self._near_far()
        r1 = self._call(self._make_loss(weight=1.0), pred, gt_depth, near, far)
        r3 = self._call(self._make_loss(weight=3.0), pred, gt_depth, near, far)
        torch.testing.assert_close(r3, r1 * 3.0, atol=1e-5, rtol=1e-5)

    def test_invalid_gt_pixels_excluded(self):
        # Right half of gt_depth exceeds far — those pixels are excluded.
        # render_depth == gt_depth for the valid left half → loss is zero.
        near, far = self._near_far(near=0.5, far=10.0)
        gt_depth = torch.ones(B, V, H, W) * 5.0
        gt_depth[..., W // 2:] = 200.0  # > far → invalid
        render_depth = torch.ones(B, V, H, W) * 5.0  # perfect match for valid pixels
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(), pred, gt_depth, near, far)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_invalid_render_depth_still_supervised(self):
        # Render depth below near for right half — these pixels are NOT excluded (only gt bounds matter).
        # gt == render for left half (loss zero there), render != gt for right half (nonzero contribution).
        near, far = self._near_far(near=1.0, far=100.0)
        gt_depth = torch.ones(B, V, H, W) * 5.0
        render_depth = gt_depth.clone()
        render_depth[..., W // 2:] = 0.1  # out of range but still supervised
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(), pred, gt_depth, near, far)
        assert result.item() > 0

    def test_perfect_prediction_zero_loss(self):
        depth = torch.rand(B, V, H, W) * 9 + 1.0
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=depth)
        result = self._call(self._make_loss(), pred, depth.clone(), *self._near_far())
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_explicit_depth_mask_overrides_validity(self):
        # The dataset-supplied depth_mask defines validity, overriding the near/far fallback. gt is
        # fully in range, but the mask marks the right half invalid; the right half mismatches gt, so
        # only if the mask is honored does that mismatch get excluded and the loss come out zero.
        gt_depth = torch.full((B, V, H, W), math.exp(1))
        render_depth = gt_depth.clone()
        render_depth[..., W // 2:] = math.exp(2)  # right half mismatches (would add |1-2|=1 if counted)
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        mask = torch.ones(B, V, H, W, dtype=torch.bool)
        mask[..., W // 2:] = False  # dataset marks the right half invalid
        result = self._call(self._make_loss(), pred, gt_depth, *self._near_far(), depth_mask=mask)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_clamped_render_depth_is_finite(self):
        # Empty/zero-coverage pixels render depth ~0; log(0) would be -inf. The loss clamps rendered
        # depth into [near, far] so it stays finite while still penalizing those pixels.
        gt_depth = torch.full((B, V, H, W), 5.0)
        render_depth = torch.full((B, V, H, W), 5.0)
        render_depth[..., 0, 0] = 0.0   # empty pixel
        render_depth[..., 0, 1] = -2.0  # negative pixel
        pred = DecoderOutput(color=torch.rand(B, V, C, H, W), depth=render_depth)
        result = self._call(self._make_loss(), pred, gt_depth, *self._near_far())
        assert torch.isfinite(result).item()
        assert result.item() > 0  # the clamped invalid pixels are still supervised (nonzero penalty)


# ---------- LossSh0 tests ----------


class TestLossSh0:
    def test_forward_returns_scalar(self):
        from learn2splat.loss.loss_sh0 import LossSh0, LossSh0Cfg, LossSh0CfgWrapper
        loss_fn = LossSh0(LossSh0CfgWrapper(sh0=LossSh0Cfg(weight=1.0)))
        gaussians = make_gaussians()
        pred = make_prediction()
        pred.means2d = torch.rand(B, V, G, 2)
        pred.radii = torch.ones(B, V, G, 2, dtype=torch.int32)
        batch = make_batch()
        result = loss_fn(
            pred, gaussians, 0,
            gt_rgb=batch["target"]["image"], pred_rgb=pred.color,
            valid_depth_mask=None, gt_image=batch["target"]["image"],
        )
        assert result.shape == ()