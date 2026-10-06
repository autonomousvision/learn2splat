"""Tests for update MLP head configurations: baseline, per-param heads (A), per-param scales (B).

Tests correctness of all three approaches and benchmarks execution time + parameter counts.
"""
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from learn2splat.scene_trainer.optimizer.optimizer_knn_based import get_activation_cls


# ---------------------------------------------------------------------------
# Minimal config stubs (only the fields touched by the update head logic)
# ---------------------------------------------------------------------------

@dataclass
class _UpdateHeadCfg:
    # Core head config
    delta_head_layer_num: int = 2
    delta_head_act: str = "gelu"
    delta_head_final_act: str = "identity"
    delta_head_concat_img: bool = False
    delta_head_scalar_scale: bool = True
    delta_head_scalar_scale_act: str = "relu"
    # Feature A
    delta_head_per_param_heads: bool = False
    delta_head_per_param_hidden_dim: int = 48
    # Feature B
    delta_head_per_param_scales: bool = False
    # Gaussian config
    sh_d: int = 16
    freeze_mean: bool = False
    freeze_scale: bool = False
    freeze_rotation: bool = False
    freeze_opacity: bool = False
    freeze_sh0: bool = False
    freeze_shN: bool = False
    delta_gaussian_multiple: int = 1
    same_num_points: bool = True
    init_gaussian_multiple: int = 1


# ---------------------------------------------------------------------------
# Standalone builder functions (mirrors optimizer_knn_based.py logic)
# ---------------------------------------------------------------------------

def compute_group_dims(cfg):
    """Compute per-param-group output dimensions, omitting frozen groups.

    Mirrors KnnBasedOptimizer._frozen_param_groups + _compute_per_param_group_dims:
    any combination of freeze_* is supported; SH is frozen only when both freeze_sh0
    and freeze_shN are set.
    """
    all_groups = {"means": 3, "scales": 3, "rotations": 4, "opacities": 1, "shs": 3 * cfg.sh_d}

    frozen = set()
    if cfg.freeze_mean:
        frozen.add("means")
    if cfg.freeze_scale:
        frozen.add("scales")
    if cfg.freeze_rotation:
        frozen.add("rotations")
    if cfg.freeze_opacity:
        frozen.add("opacities")
    assert cfg.freeze_sh0 == cfg.freeze_shN, "SH is one group; freeze_sh0 and freeze_shN must match"
    if cfg.freeze_sh0 and cfg.freeze_shN:
        frozen.add("shs")

    multiplier = cfg.delta_gaussian_multiple
    if not cfg.same_num_points:
        multiplier *= cfg.init_gaussian_multiple
    return {name: dim * multiplier for name, dim in all_groups.items() if name not in frozen}


def build_baseline_head(cfg, channels, out_channels):
    """Build single-head baseline (with optional global scalar scale)."""
    act_cls = get_activation_cls(cfg.delta_head_act)
    final_act_cls = get_activation_cls(cfg.delta_head_final_act)
    total_out = out_channels + (1 if cfg.delta_head_scalar_scale else 0)
    hidden = channels
    layers = [nn.Linear(channels, hidden), act_cls()]
    for _ in range(cfg.delta_head_layer_num - 2):
        layers += [nn.Linear(hidden, hidden), act_cls()]
    layers += [nn.Linear(hidden, total_out), final_act_cls()]
    head = nn.Sequential(*layers)
    nn.init.zeros_(head[-2].weight)
    nn.init.zeros_(head[-2].bias)
    return head


def build_per_param_heads(cfg, channels, group_dims):
    """Build per-parameter-group heads (Feature A)."""
    act_cls = get_activation_cls(cfg.delta_head_act)
    hidden_dim = cfg.delta_head_per_param_hidden_dim
    heads = nn.ModuleDict()
    for name, dim in group_dims.items():
        h = hidden_dim * 2 if name == "shs" else hidden_dim
        layers = [nn.Linear(channels, h), act_cls()]
        for _ in range(cfg.delta_head_layer_num - 2):
            layers += [nn.Linear(h, h), act_cls()]
        layers.append(nn.Linear(h, dim + 1))
        head = nn.Sequential(*layers)
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        heads[name] = head
    return heads


def build_per_param_scales_head(cfg, channels, out_channels, num_groups):
    """Build single-head with per-group scalar scales (Feature B)."""
    act_cls = get_activation_cls(cfg.delta_head_act)
    final_act_cls = get_activation_cls(cfg.delta_head_final_act)
    total_out = out_channels + num_groups
    hidden = channels
    layers = [nn.Linear(channels, hidden), act_cls()]
    for _ in range(cfg.delta_head_layer_num - 2):
        layers += [nn.Linear(hidden, hidden), act_cls()]
    layers += [nn.Linear(hidden, total_out), final_act_cls()]
    head = nn.Sequential(*layers)
    nn.init.zeros_(head[-2].weight)
    nn.init.zeros_(head[-2].bias)
    return head


# ---------------------------------------------------------------------------
# Forward pass functions
# ---------------------------------------------------------------------------

def forward_baseline(head, x, scale_act):
    """Baseline forward: global L2 normalize + single scalar scale."""
    raw = head(x)
    scale = scale_act(raw[:, -1:])
    deltas = raw[:, :-1]
    deltas = deltas / (deltas.norm(p=2, dim=-1, keepdim=True) + 1e-8) * scale
    return deltas


def forward_per_param_heads(heads, x, group_dims, scale_act):
    """Feature A forward: per-group normalize + scale."""
    deltas = []
    for name, dim in group_dims.items():
        raw = heads[name](x)
        scale = scale_act(raw[:, -1:])
        delta = raw[:, :-1]
        if dim > 1:
            delta = delta / (delta.norm(p=2, dim=-1, keepdim=True) + 1e-8) * scale
        else:
            delta = delta * scale
        deltas.append(delta)
    return torch.cat(deltas, dim=-1)


def forward_per_param_scales(head, x, group_dims, scale_act):
    """Feature B forward: single head, per-group normalize + scale."""
    raw = head(x)
    num_groups = len(group_dims)
    scales = scale_act(raw[:, -num_groups:])
    deltas_raw = raw[:, :-num_groups]
    normalized = []
    offset = 0
    for i, (name, dim) in enumerate(group_dims.items()):
        group = deltas_raw[:, offset:offset + dim]
        s = scales[:, i:i + 1]
        if dim > 1:
            group = group / (group.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        group = group * s
        normalized.append(group)
        offset += dim
    return torch.cat(normalized, dim=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdateMLPCorrectness:
    """Verify output shapes, zero-init, and gradient flow for all approaches."""

    CHANNELS = 256
    OUT_CHANNELS = 59  # 3+3+4+1+48

    @pytest.fixture
    def cfg(self):
        return _UpdateHeadCfg()

    @pytest.fixture
    def group_dims(self, cfg):
        return compute_group_dims(cfg)

    @pytest.fixture
    def state(self):
        return torch.randn(128, self.CHANNELS)

    def test_baseline_output_shape(self, cfg, state):
        head = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS)
        scale_act = nn.ReLU()
        out = forward_baseline(head, state, scale_act)
        assert out.shape == (128, self.OUT_CHANNELS)

    def test_per_param_heads_output_shape(self, cfg, group_dims, state):
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        scale_act = nn.ReLU()
        out = forward_per_param_heads(heads, state, group_dims, scale_act)
        assert out.shape == (128, self.OUT_CHANNELS)

    def test_per_param_scales_output_shape(self, cfg, group_dims, state):
        head = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))
        scale_act = nn.ReLU()
        out = forward_per_param_scales(head, state, group_dims, scale_act)
        assert out.shape == (128, self.OUT_CHANNELS)

    def test_baseline_zero_init(self, cfg, state):
        """Deltas should be ~0 at init (zero-init last layer + relu scale)."""
        head = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS)
        nn.init.constant_(head[-2].bias[-1], 1e-8)  # relu scale init
        scale_act = nn.ReLU()
        out = forward_baseline(head, state, scale_act)
        assert out.abs().max() < 1e-4, f"Expected near-zero init deltas, got max={out.abs().max()}"

    def test_per_param_heads_zero_init(self, cfg, group_dims, state):
        """Each head's deltas should be ~0 at init."""
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        scale_act = nn.ReLU()
        out = forward_per_param_heads(heads, state, group_dims, scale_act)
        assert out.abs().max() < 1e-4, f"Expected near-zero init deltas, got max={out.abs().max()}"

    def test_per_param_scales_zero_init(self, cfg, group_dims, state):
        """Per-param-scale deltas should be ~0 at init."""
        head = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))
        num_groups = len(group_dims)
        for i in range(num_groups):
            nn.init.constant_(head[-2].bias[-(num_groups - i)], 1e-8)
        scale_act = nn.ReLU()
        out = forward_per_param_scales(head, state, group_dims, scale_act)
        assert out.abs().max() < 1e-4, f"Expected near-zero init deltas, got max={out.abs().max()}"

    def test_baseline_gradient_flow(self, cfg):
        """Gradients should flow through the baseline head."""
        head = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS)
        scale_act = nn.Softplus(beta=1)
        # Need non-zero last-layer weights for gradient flow through L2 normalization
        nn.init.normal_(head[-2].weight, std=0.01)
        nn.init.constant_(head[-2].bias[-1], -1)
        x = torch.randn(32, self.CHANNELS, requires_grad=True)
        out = forward_baseline(head, x, scale_act)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_per_param_heads_gradient_flow(self, cfg, group_dims):
        """Gradients should flow through all per-param heads."""
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        scale_act = nn.Softplus(beta=1)
        for name in group_dims:
            nn.init.normal_(heads[name][-1].weight, std=0.01)
            nn.init.constant_(heads[name][-1].bias[-1], -1)
        x = torch.randn(32, self.CHANNELS, requires_grad=True)
        out = forward_per_param_heads(heads, x, group_dims, scale_act)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        # Verify all heads received gradients
        for name in group_dims:
            grad_norm = sum(p.grad.abs().sum() for p in heads[name].parameters() if p.grad is not None)
            assert grad_norm > 0, f"No gradient flow for head '{name}'"

    def test_per_param_scales_gradient_flow(self, cfg, group_dims):
        """Gradients should flow through per-param-scale head."""
        head = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))
        scale_act = nn.Softplus(beta=1)
        nn.init.normal_(head[-2].weight, std=0.01)
        nn.init.constant_(head[-2].bias[-1], -1)
        x = torch.randn(32, self.CHANNELS, requires_grad=True)
        out = forward_per_param_scales(head, x, group_dims, scale_act)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_per_param_heads_independent_scaling(self, cfg, group_dims):
        """Verify that per-param heads produce independent scales per group."""
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        # Deliberately init one head with non-zero weights to break symmetry
        nn.init.normal_(heads["means"][-1].weight, std=0.1)
        scale_act = nn.Softplus(beta=1)
        x = torch.randn(64, self.CHANNELS)
        out = forward_per_param_heads(heads, x, group_dims, scale_act)

        # means should have non-zero deltas, others should be near-zero (still zero-init)
        means_delta = out[:, :3]
        other_delta = out[:, 3:]
        assert means_delta.abs().mean() > other_delta.abs().mean() * 10

    def test_per_param_scales_independent_scaling(self, cfg, group_dims):
        """Verify that per-param scales allow independent magnitude per group."""
        head = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))
        scale_act = nn.Softplus(beta=1)
        x = torch.randn(64, self.CHANNELS)

        # Set different scale biases per group
        num_groups = len(group_dims)
        with torch.no_grad():
            for i in range(num_groups):
                # means gets large scale, others get small
                bias_val = 2.0 if i == 0 else -5.0
                head[-2].bias[-(num_groups - i)] = bias_val
            # Put some signal in the delta part too
            nn.init.normal_(head[-2].weight[:self.OUT_CHANNELS], std=0.01)

        out = forward_per_param_scales(head, x, group_dims, scale_act)
        means_delta = out[:, :3]
        shs_delta = out[:, 11:]  # last group
        # means should have larger magnitude than shs
        assert means_delta.abs().mean() > shs_delta.abs().mean()

    def test_softplus_scale_activation(self, cfg, group_dims):
        """Softplus should give non-zero gradients even at init."""
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        scale_act = nn.Softplus(beta=1)
        # Set scale bias to -1 (softplus(-1) ≈ 0.31)
        for name in group_dims:
            nn.init.constant_(heads[name][-1].bias[-1], -1.0)
            nn.init.normal_(heads[name][-1].weight, std=0.01)
        x = torch.randn(32, self.CHANNELS, requires_grad=True)
        out = forward_per_param_heads(heads, x, group_dims, scale_act)
        # With softplus init, deltas should be small but non-zero
        assert out.abs().max() > 1e-6

    def test_3layer_head(self):
        """Test with delta_head_layer_num=3 (deeper heads)."""
        cfg = _UpdateHeadCfg(delta_head_layer_num=3)
        group_dims = compute_group_dims(cfg)
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        scale_act = nn.ReLU()
        x = torch.randn(32, self.CHANNELS)
        out = forward_per_param_heads(heads, x, group_dims, scale_act)
        assert out.shape == (32, self.OUT_CHANNELS)
        # Check 3 layers: Linear, act, Linear, act, Linear
        for name in group_dims:
            assert len(heads[name]) == 5, f"Expected 5 layers for 3-layer head, got {len(heads[name])}"


class TestFreezeCombinations:
    """Verify generalized freeze_* handling (replaces the old sh_only special case)."""

    SH_D = 16
    SH_DIM = 3 * 16  # 48
    FULL = {"means": 3, "scales": 3, "rotations": 4, "opacities": 1, "shs": 48}

    def test_nothing_frozen(self):
        dims = compute_group_dims(_UpdateHeadCfg())
        assert dims == self.FULL
        assert sum(dims.values()) == 59

    def test_freeze_mean_only(self):
        dims = compute_group_dims(_UpdateHeadCfg(freeze_mean=True))
        assert "means" not in dims
        assert sum(dims.values()) == 56

    def test_sh_only_equivalent(self):
        """freeze mean+scale+rotation+opacity == the old sh_only (predict SH only)."""
        cfg = _UpdateHeadCfg(freeze_mean=True, freeze_scale=True,
                             freeze_rotation=True, freeze_opacity=True)
        dims = compute_group_dims(cfg)
        assert dims == {"shs": self.SH_DIM}

    def test_arbitrary_combination(self):
        """Combinations that the old if/elif could not express now work."""
        dims = compute_group_dims(_UpdateHeadCfg(freeze_mean=True, freeze_scale=True))
        assert set(dims) == {"rotations", "opacities", "shs"}
        assert sum(dims.values()) == 4 + 1 + self.SH_DIM

    def test_freeze_shs_requires_both(self):
        dims = compute_group_dims(_UpdateHeadCfg(freeze_sh0=True, freeze_shN=True))
        assert "shs" not in dims
        with pytest.raises(AssertionError):
            compute_group_dims(_UpdateHeadCfg(freeze_sh0=True, freeze_shN=False))


class TestUpdateMLPBenchmark:
    """Benchmark parameter counts and execution time for all approaches."""

    CHANNELS = 256
    OUT_CHANNELS = 59
    N_GAUSSIANS = 100_000
    N_WARMUP = 5
    N_ITERS = 20

    @pytest.fixture
    def cfg(self):
        return _UpdateHeadCfg()

    @pytest.fixture
    def group_dims(self, cfg):
        return compute_group_dims(cfg)

    def _count_params(self, module):
        return sum(p.numel() for p in module.parameters())

    def _benchmark_cpu(self, fn, n_warmup, n_iters):
        """Benchmark a function on CPU, return mean time in ms."""
        for _ in range(n_warmup):
            fn()
        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            fn()
            times.append((time.perf_counter() - start) * 1000)
        return sum(times) / len(times)

    def test_param_count_comparison(self, cfg, group_dims):
        """Compare parameter counts across all approaches."""
        baseline = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS)
        per_param = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        per_scales = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))

        baseline_params = self._count_params(baseline)
        per_param_params = self._count_params(per_param)
        per_scales_params = self._count_params(per_scales)

        print(f"\n{'='*60}")
        print(f"{'Parameter Count Comparison':^60}")
        print(f"{'='*60}")
        print(f"  Baseline (global scalar scale):    {baseline_params:>8,}")
        print(f"  Feature A (per-param heads h={cfg.delta_head_per_param_hidden_dim}):  {per_param_params:>8,}")
        print(f"  Feature B (per-param scales):      {per_scales_params:>8,}")
        print(f"{'='*60}")
        print(f"  A vs baseline: {(per_param_params/baseline_params - 1)*100:+.1f}%")
        print(f"  B vs baseline: {(per_scales_params/baseline_params - 1)*100:+.1f}%")
        print(f"{'='*60}")

        # Assert param counts are within 10% of each other
        max_params = max(baseline_params, per_param_params, per_scales_params)
        min_params = min(baseline_params, per_param_params, per_scales_params)
        ratio = max_params / min_params
        assert ratio < 1.10, (
            f"Param counts differ by more than 10%: "
            f"baseline={baseline_params}, A={per_param_params}, B={per_scales_params} "
            f"(ratio={ratio:.2f})"
        )

    def test_per_param_head_breakdown(self, cfg, group_dims):
        """Show per-head parameter breakdown for Feature A."""
        heads = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        print(f"\n{'='*60}")
        print(f"{'Feature A: Per-Head Parameter Breakdown':^60}")
        print(f"{'='*60}")
        total = 0
        for name, dim in group_dims.items():
            p = self._count_params(heads[name])
            total += p
            h = cfg.delta_head_per_param_hidden_dim * (2 if name == "shs" else 1)
            print(f"  {name:>12s}: {self.CHANNELS}->{h}->{dim+1}  = {p:>6,} params")
        print(f"  {'TOTAL':>12s}: {total:>22,} params")
        print(f"{'='*60}")

    def test_execution_time_cpu(self, cfg, group_dims):
        """Benchmark forward pass execution time on CPU."""
        x = torch.randn(self.N_GAUSSIANS, self.CHANNELS)
        scale_act = nn.ReLU()

        baseline = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS)
        per_param = build_per_param_heads(cfg, self.CHANNELS, group_dims)
        per_scales = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims))

        baseline.eval()
        per_param.eval()
        per_scales.eval()

        with torch.no_grad():
            t_baseline = self._benchmark_cpu(
                lambda: forward_baseline(baseline, x, scale_act),
                self.N_WARMUP, self.N_ITERS
            )
            t_per_param = self._benchmark_cpu(
                lambda: forward_per_param_heads(per_param, x, group_dims, scale_act),
                self.N_WARMUP, self.N_ITERS
            )
            t_per_scales = self._benchmark_cpu(
                lambda: forward_per_param_scales(per_scales, x, group_dims, scale_act),
                self.N_WARMUP, self.N_ITERS
            )

        print(f"\n{'='*60}")
        print(f"{'CPU Forward Pass Benchmark (N=' + f'{self.N_GAUSSIANS:,}' + ')':^60}")
        print(f"{'='*60}")
        print(f"  Baseline:  {t_baseline:>8.2f} ms")
        print(f"  Feature A: {t_per_param:>8.2f} ms ({(t_per_param/t_baseline - 1)*100:+.1f}%)")
        print(f"  Feature B: {t_per_scales:>8.2f} ms ({(t_per_scales/t_baseline - 1)*100:+.1f}%)")
        print(f"{'='*60}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_execution_time_gpu(self, cfg, group_dims):
        """Benchmark forward pass execution time on GPU."""
        device = torch.device("cuda")
        x = torch.randn(self.N_GAUSSIANS, self.CHANNELS, device=device)
        scale_act = nn.ReLU().to(device)

        baseline = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS).to(device)
        per_param = build_per_param_heads(cfg, self.CHANNELS, group_dims).to(device)
        per_scales = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims)).to(device)

        baseline.eval()
        per_param.eval()
        per_scales.eval()

        def benchmark_cuda(fn, n_warmup, n_iters):
            for _ in range(n_warmup):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000 / n_iters

        with torch.no_grad():
            t_baseline = benchmark_cuda(
                lambda: forward_baseline(baseline, x, scale_act),
                self.N_WARMUP, self.N_ITERS
            )
            t_per_param = benchmark_cuda(
                lambda: forward_per_param_heads(per_param, x, group_dims, scale_act),
                self.N_WARMUP, self.N_ITERS
            )
            t_per_scales = benchmark_cuda(
                lambda: forward_per_param_scales(per_scales, x, group_dims, scale_act),
                self.N_WARMUP, self.N_ITERS
            )

        print(f"\n{'='*60}")
        print(f"{'GPU Forward Pass Benchmark (N=' + f'{self.N_GAUSSIANS:,}' + ')':^60}")
        print(f"{'='*60}")
        print(f"  Baseline:  {t_baseline:>8.3f} ms")
        print(f"  Feature A: {t_per_param:>8.3f} ms ({(t_per_param/t_baseline - 1)*100:+.1f}%)")
        print(f"  Feature B: {t_per_scales:>8.3f} ms ({(t_per_scales/t_baseline - 1)*100:+.1f}%)")
        print(f"{'='*60}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_time_gpu(self, cfg, group_dims):
        """Benchmark backward pass execution time on GPU."""
        device = torch.device("cuda")
        scale_act = nn.Softplus(beta=1).to(device)

        baseline = build_baseline_head(cfg, self.CHANNELS, self.OUT_CHANNELS).to(device)
        per_param = build_per_param_heads(cfg, self.CHANNELS, group_dims).to(device)
        per_scales = build_per_param_scales_head(cfg, self.CHANNELS, self.OUT_CHANNELS, len(group_dims)).to(device)
        # Init with softplus for gradient flow
        nn.init.constant_(baseline[-2].bias[-1], -1)
        for name in group_dims:
            nn.init.constant_(per_param[name][-1].bias[-1], -1)
            nn.init.normal_(per_param[name][-1].weight, std=0.01)
        nn.init.constant_(per_scales[-2].bias[-1], -1)

        def benchmark_backward(model, forward_fn, n_warmup, n_iters):
            for _ in range(n_warmup):
                x = torch.randn(self.N_GAUSSIANS, self.CHANNELS, device=device, requires_grad=True)
                out = forward_fn(x)
                out.sum().backward()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_iters):
                x = torch.randn(self.N_GAUSSIANS, self.CHANNELS, device=device, requires_grad=True)
                out = forward_fn(x)
                out.sum().backward()
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000 / n_iters

        t_baseline = benchmark_backward(
            baseline, lambda x: forward_baseline(baseline, x, scale_act),
            self.N_WARMUP, self.N_ITERS
        )
        t_per_param = benchmark_backward(
            per_param, lambda x: forward_per_param_heads(per_param, x, group_dims, scale_act),
            self.N_WARMUP, self.N_ITERS
        )
        t_per_scales = benchmark_backward(
            per_scales, lambda x: forward_per_param_scales(per_scales, x, group_dims, scale_act),
            self.N_WARMUP, self.N_ITERS
        )

        print(f"\n{'='*60}")
        print(f"{'GPU Backward Pass Benchmark (N=' + f'{self.N_GAUSSIANS:,}' + ')':^60}")
        print(f"{'='*60}")
        print(f"  Baseline:  {t_baseline:>8.3f} ms")
        print(f"  Feature A: {t_per_param:>8.3f} ms ({(t_per_param/t_baseline - 1)*100:+.1f}%)")
        print(f"  Feature B: {t_per_scales:>8.3f} ms ({(t_per_scales/t_baseline - 1)*100:+.1f}%)")
        print(f"{'='*60}")
