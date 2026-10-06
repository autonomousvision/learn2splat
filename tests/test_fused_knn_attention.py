"""
Test numerical equivalence between fused and unfused KNN attention.

Tests:
1. Forward output matches between fused CUDA kernel, PyTorch fallback, and original code
2. Backward gradients (grad_Q, grad_K, grad_V) match for all three implementations
3. Gradient flow through the full KNNAttention module (with qkv projection + output projection)
4. Various shapes: small/large N, different C, different K
5. Edge cases: K=1, K=N
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

# Add the submodules dir to path so `fused_knn_attn` is importable. Append (not insert at 0) so the
# installed `pointops` wins over the unbuilt `submodules/pointops` (whose `_C` ext isn't compiled).
SUBMODULES_DIR = os.path.join(os.path.dirname(__file__), "..", "submodules")
sys.path.append(SUBMODULES_DIR)

from fused_knn_attn import (
    FusedKNNAttentionFunction,
    FusedKNNAttentionFunctionPyTorch,
    FUSED_KNN_ATTN_CUDA_AVAILABLE,
)


def _gpu_total_gb() -> float:
    """Total VRAM of the current CUDA device in GiB (0 if no CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)


# The benchmark sweep runs every size (incl. 500k) and builds the unfused reference, which materializes
# large neighbor tensors and needs tens of GB — a perf benchmark for big-VRAM GPUs (A100/H100), not a
# correctness check. It is skipped on smaller cards so a full `pytest tests/` stays green everywhere.
# (The large-scale correctness tests below instead adapt: they run each size that fits and skip on OOM.)
_BENCHMARK_MIN_GB = 38
_skip_if_small_gpu = pytest.mark.skipif(
    _gpu_total_gb() < _BENCHMARK_MIN_GB,
    reason=f"fused-kNN perf benchmark needs a large-VRAM GPU (~A100/H100); have {_gpu_total_gb():.0f} GB",
)


def _reference_knn_attention(q, k, v, idx, scale):
    """Reference implementation matching the original KNNAttention.forward() logic.

    This is the exact sequence from layer.py lines 82-177 (unfused path):
      1. Gather K neighbors: x_k[n, kk, :] = k[idx[n, kk], :]
      2. Gather V neighbors: x_v[n, kk, :] = v[idx[n, kk], :]
      3. scores = Q @ K_grouped^T * scale
      4. attn = softmax(scores)
      5. out = attn @ V_grouped
    """
    N, C = q.shape
    num_k = idx.shape[1]

    # Gather: exactly what pointops.grouping does via fancy indexing
    idx_long = idx.long()
    x_k = k[idx_long.view(-1)].view(N, num_k, C)   # [N, K, C]
    x_v = v[idx_long.view(-1)].view(N, num_k, C)    # [N, K, C]

    # Attention: exactly the matmul path from layer.py
    scores = torch.matmul(q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale  # [N, 1, K]
    out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)     # [N, C]
    return out


def _make_test_data(N, C, K, device="cuda", seed=42):
    """Create deterministic test data with known KNN indices."""
    torch.manual_seed(seed)

    q = torch.randn(N, C, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(N, C, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(N, C, device=device, dtype=torch.float32, requires_grad=True)

    # Random valid KNN indices (each point's neighbors are random other points)
    idx = torch.randint(0, N, (N, K), device=device, dtype=torch.int32)

    return q, k, v, idx


# ============================================================================
# Forward equivalence tests
# ============================================================================

@pytest.mark.parametrize("N,C,K", [
    (64, 32, 4),       # small
    (256, 64, 8),      # medium
    (1024, 128, 16),   # typical
    (4096, 256, 16),   # production-like
    (128, 64, 1),      # K=1 edge case
    (32, 256, 32),     # large K relative to N
])
class TestForwardEquivalence:

    def test_pytorch_fallback_matches_reference(self, N, C, K):
        """PyTorch fallback (loop-based) matches the reference (gather-based)."""
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5

        ref_out = _reference_knn_attention(q.detach(), k.detach(), v.detach(), idx, scale)
        fused_out = FusedKNNAttentionFunctionPyTorch.apply(
            q.detach().clone().requires_grad_(False),
            k.detach().clone().requires_grad_(False),
            v.detach().clone().requires_grad_(False),
            idx, scale
        )

        torch.testing.assert_close(fused_out, ref_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_cuda_kernel_matches_reference(self, N, C, K):
        """CUDA fused kernel matches the reference."""
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5

        ref_out = _reference_knn_attention(q.detach(), k.detach(), v.detach(), idx, scale)
        cuda_out = FusedKNNAttentionFunction.apply(
            q.detach().clone().requires_grad_(False),
            k.detach().clone().requires_grad_(False),
            v.detach().clone().requires_grad_(False),
            idx, scale
        )

        torch.testing.assert_close(cuda_out, ref_out, atol=1e-4, rtol=1e-4)


# ============================================================================
# Backward equivalence tests
# ============================================================================

def _get_reference_grads(q, k, v, idx, scale, grad_out):
    """Compute gradients through the reference implementation."""
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)

    out = _reference_knn_attention(q, k, v, idx, scale)
    out.backward(grad_out)
    return q.grad.clone(), k.grad.clone(), v.grad.clone()


@pytest.mark.parametrize("N,C,K", [
    (64, 32, 4),
    (256, 64, 8),
    (1024, 128, 16),
    (128, 64, 1),
])
class TestBackwardEquivalence:

    def test_pytorch_fallback_grads_match_reference(self, N, C, K):
        """PyTorch fallback backward matches reference backward."""
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5
        grad_out = torch.randn(N, C, device="cuda", dtype=torch.float32)

        ref_gq, ref_gk, ref_gv = _get_reference_grads(q, k, v, idx, scale, grad_out)

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        out2 = FusedKNNAttentionFunctionPyTorch.apply(q2, k2, v2, idx, scale)
        out2.backward(grad_out)

        torch.testing.assert_close(q2.grad, ref_gq, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k2.grad, ref_gk, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v2.grad, ref_gv, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_cuda_kernel_grads_match_reference(self, N, C, K):
        """CUDA fused kernel backward matches reference backward."""
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5
        grad_out = torch.randn(N, C, device="cuda", dtype=torch.float32)

        ref_gq, ref_gk, ref_gv = _get_reference_grads(q, k, v, idx, scale, grad_out)

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        out2 = FusedKNNAttentionFunction.apply(q2, k2, v2, idx, scale)
        out2.backward(grad_out)

        # Slightly higher tolerance for CUDA kernel due to atomicAdd non-determinism
        torch.testing.assert_close(q2.grad, ref_gq, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(k2.grad, ref_gk, atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(v2.grad, ref_gv, atol=5e-4, rtol=5e-4)


# ============================================================================
# Gradient flow test: full KNNAttention module
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestKNNAttentionModule:
    """Test that gradients flow correctly through the full KNNAttention module
    (including qkv projection and output projection) with use_fused=True."""

    def _make_module_test_data(self, N=512, C=128, K=16, seed=42):
        torch.manual_seed(seed)
        p = torch.randn(N, 3, device="cuda")
        x = torch.randn(N, C, device="cuda", requires_grad=True)
        # Offsets for single batch
        o = torch.tensor([N], device="cuda", dtype=torch.int32)
        return p, x, o, K

    def test_fused_vs_unfused_module_forward(self):
        """KNNAttention(use_fused=True) produces same output as use_fused=False."""
        from learn2splat.model.backbones.point_transformer.layer import KNNAttention
        import pointops

        p, x, o, K = self._make_module_test_data()
        C = x.shape[1]

        # Pre-compute KNN indices (both paths will use these)
        knn_idx, _ = pointops.knn_query(K, p, o, p, o)

        # Create two identical modules
        torch.manual_seed(0)
        mod_unfused = KNNAttention(C, knn_samples=K, use_fused=False).cuda()
        torch.manual_seed(0)
        mod_fused = KNNAttention(C, knn_samples=K, use_fused=True).cuda()

        # Copy weights to ensure identical
        mod_fused.load_state_dict(mod_unfused.state_dict())

        with torch.no_grad():
            out_unfused = mod_unfused((p, x, o), knn_idx=knn_idx)
            out_fused = mod_fused((p, x, o), knn_idx=knn_idx)

        torch.testing.assert_close(out_fused, out_unfused, atol=1e-3, rtol=1e-3)

    def test_fused_module_backward_produces_gradients(self):
        """KNNAttention(use_fused=True) produces valid gradients for input and params."""
        from learn2splat.model.backbones.point_transformer.layer import KNNAttention
        import pointops

        p, x, o, K = self._make_module_test_data()
        C = x.shape[1]

        knn_idx, _ = pointops.knn_query(K, p, o, p, o)

        mod = KNNAttention(C, knn_samples=K, use_fused=True).cuda()

        x_input = x.detach().clone().requires_grad_(True)
        out = mod((p, x_input, o), knn_idx=knn_idx)
        loss = out.sum()
        loss.backward()

        # Input gradient exists and is non-zero
        assert x_input.grad is not None
        assert x_input.grad.abs().sum() > 0, "Input gradient is all zeros"

        # Module parameter gradients exist
        for name, param in mod.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_fused_vs_unfused_module_backward(self):
        """Gradients through fused and unfused modules match."""
        from learn2splat.model.backbones.point_transformer.layer import KNNAttention
        import pointops

        p, x, o, K = self._make_module_test_data()
        C = x.shape[1]

        knn_idx, _ = pointops.knn_query(K, p, o, p, o)

        torch.manual_seed(0)
        mod_unfused = KNNAttention(C, knn_samples=K, use_fused=False).cuda()
        torch.manual_seed(0)
        mod_fused = KNNAttention(C, knn_samples=K, use_fused=True).cuda()
        mod_fused.load_state_dict(mod_unfused.state_dict())

        # Same input, same grad_output
        x1 = x.detach().clone().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)

        out1 = mod_unfused((p, x1, o), knn_idx=knn_idx)
        out2 = mod_fused((p, x2, o), knn_idx=knn_idx)

        grad_out = torch.randn_like(out1)
        out1.backward(grad_out)
        out2.backward(grad_out)

        # Input gradients match
        torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)

        # Parameter gradients match
        for (n1, p1), (n2, p2) in zip(
            mod_unfused.named_parameters(), mod_fused.named_parameters()
        ):
            assert n1 == n2
            torch.testing.assert_close(
                p2.grad, p1.grad, atol=1e-3, rtol=1e-3,
                msg=f"Gradient mismatch for parameter {n1}"
            )


# ============================================================================
# Large-scale correctness tests (500K+ points)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
class TestLargeScale:
    """Correctness at production scale (100K-500K+ points)."""

    @pytest.fixture(autouse=True)
    def _free_cuda(self):
        # Release the caching allocator before and after each large-N case so these tests start with
        # the most free VRAM (other CUDA test files run first in a full `pytest tests/` and leave cached
        # blocks behind) and don't accumulate across the size sweep.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        yield
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run each size that fits and skip (not fail) sizes too large for this GPU's VRAM, so a free
    # 4090 still covers the largest size it can while A100/H100 cover all of them.
    @pytest.mark.parametrize("N", [100_000, 250_000, 500_000])
    def test_large_forward(self, N):
        """CUDA fused forward matches reference at large N."""
        C, K = 256, 16
        try:
            q, k, v, idx = _make_test_data(N, C, K)
            scale = C ** -0.5
            ref_out = _reference_knn_attention(q.detach(), k.detach(), v.detach(), idx, scale)
            cuda_out = FusedKNNAttentionFunction.apply(
                q.detach().clone(), k.detach().clone(), v.detach().clone(), idx, scale
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise  # a real error, not VRAM pressure
            torch.cuda.empty_cache()
            pytest.skip(f"not enough VRAM for N={N} on this GPU")

        torch.testing.assert_close(cuda_out, ref_out, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("N", [100_000, 250_000, 500_000])
    def test_large_backward(self, N):
        """CUDA fused backward matches reference at large N."""
        C, K = 256, 16
        try:
            q, k, v, idx = _make_test_data(N, C, K)
            scale = C ** -0.5
            grad_out = torch.randn(N, C, device="cuda", dtype=torch.float32)

            ref_gq, ref_gk, ref_gv = _get_reference_grads(q, k, v, idx, scale, grad_out)

            q2 = q.detach().clone().requires_grad_(True)
            k2 = k.detach().clone().requires_grad_(True)
            v2 = v.detach().clone().requires_grad_(True)
            out2 = FusedKNNAttentionFunction.apply(q2, k2, v2, idx, scale)
            out2.backward(grad_out)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise  # a real error, not VRAM pressure
            torch.cuda.empty_cache()
            pytest.skip(f"not enough VRAM for N={N} on this GPU")

        # Slightly more tolerance at large N due to more atomicAdd contention
        torch.testing.assert_close(q2.grad, ref_gq, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(k2.grad, ref_gk, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(v2.grad, ref_gv, atol=1e-3, rtol=1e-3)


# ============================================================================
# Autograd gradcheck
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGradcheck:
    """Use torch.autograd.gradcheck for rigorous gradient verification."""

    def test_pytorch_fallback_gradcheck(self):
        N, C, K = 16, 8, 4
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5
        # Use double precision for gradcheck
        q = q.double().requires_grad_(True)
        k = k.double().requires_grad_(True)
        v = v.double().requires_grad_(True)

        assert torch.autograd.gradcheck(
            FusedKNNAttentionFunctionPyTorch.apply,
            (q, k, v, idx, scale),
            eps=1e-6, atol=1e-4, rtol=1e-3,
        )

    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_cuda_kernel_gradcheck(self):
        """CUDA kernel passes gradcheck (float64 not supported, use float32 with higher eps)."""
        N, C, K = 16, 8, 4
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5

        # CUDA kernel only supports float32, so use higher tolerances
        assert torch.autograd.gradcheck(
            FusedKNNAttentionFunction.apply,
            (q.requires_grad_(True), k.requires_grad_(True),
             v.requires_grad_(True), idx, scale),
            eps=1e-3, atol=1e-2, rtol=1e-2,
            nondet_tol=1e-3,  # atomicAdd is non-deterministic
        )


# ============================================================================
# Benchmarks
# ============================================================================

def _cuda_timer(fn, warmup=10, repeats=100):
    """Time a CUDA function using cuda events. Returns median ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@_skip_if_small_gpu
class TestBenchmark:
    """Benchmark forward and backward passes across implementations.

    Run with: pytest tests/test_fused_knn_attention.py::TestBenchmark -v -s
    """

    CONFIGS = [
        # (N, C, K) — representative of real workloads
        (1024,  128,  16),    # small scene
        (4096,  256,  16),    # typical RealEstate10K (8 views, 64x112 latent)
        (8192,  256,  16),    # larger scene (16 views)
        (25000, 256,  16),    # production-like (8 views, 256x448 / 4)
        (100000, 256, 16),    # large scene (init_gaussian_multiple=4)
        (500000, 256, 16),    # very large (many views or dense init)
    ]

    def _run_forward_benchmark(self, N, C, K, impl):
        """Run forward benchmark for a given implementation."""
        q, k, v, idx = _make_test_data(N, C, K)
        scale = C ** -0.5

        if impl == "reference":
            fn = lambda: _reference_knn_attention(q, k, v, idx, scale)
        elif impl == "pytorch_fallback":
            fn = lambda: FusedKNNAttentionFunctionPyTorch.apply(q, k, v, idx, scale)
        elif impl == "cuda_fused":
            fn = lambda: FusedKNNAttentionFunction.apply(q, k, v, idx, scale)
        else:
            raise ValueError(impl)

        return _cuda_timer(fn)

    def _run_backward_benchmark(self, N, C, K, impl):
        """Run forward+backward benchmark for a given implementation."""
        q_base, k_base, v_base, idx = _make_test_data(N, C, K)
        scale = C ** -0.5
        grad_out = torch.randn(N, C, device="cuda", dtype=torch.float32)

        if impl == "reference":
            def fn():
                q = q_base.detach().clone().requires_grad_(True)
                k = k_base.detach().clone().requires_grad_(True)
                v = v_base.detach().clone().requires_grad_(True)
                out = _reference_knn_attention(q, k, v, idx, scale)
                out.backward(grad_out)
        elif impl == "pytorch_fallback":
            def fn():
                q = q_base.detach().clone().requires_grad_(True)
                k = k_base.detach().clone().requires_grad_(True)
                v = v_base.detach().clone().requires_grad_(True)
                out = FusedKNNAttentionFunctionPyTorch.apply(q, k, v, idx, scale)
                out.backward(grad_out)
        elif impl == "cuda_fused":
            def fn():
                q = q_base.detach().clone().requires_grad_(True)
                k = k_base.detach().clone().requires_grad_(True)
                v = v_base.detach().clone().requires_grad_(True)
                out = FusedKNNAttentionFunction.apply(q, k, v, idx, scale)
                out.backward(grad_out)
        else:
            raise ValueError(impl)

        return _cuda_timer(fn, warmup=5, repeats=50)

    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_benchmark_forward(self):
        """Benchmark forward pass: reference vs pytorch_fallback vs cuda_fused."""
        print("\n" + "=" * 80)
        print("FORWARD PASS BENCHMARK (median ms, 100 repeats)")
        print("=" * 80)
        header = f"{'N':>6} {'C':>4} {'K':>3} | {'Reference':>10} {'PyTorch':>10} {'CUDA Fused':>10} | {'Speedup':>8}"
        print(header)
        print("-" * len(header))

        for N, C, K in self.CONFIGS:
            t_ref = self._run_forward_benchmark(N, C, K, "reference")
            t_pt  = self._run_forward_benchmark(N, C, K, "pytorch_fallback")
            t_cu  = self._run_forward_benchmark(N, C, K, "cuda_fused")
            speedup = t_ref / t_cu if t_cu > 0 else float("inf")
            print(
                f"{N:>6} {C:>4} {K:>3} | "
                f"{t_ref:>9.3f}ms {t_pt:>9.3f}ms {t_cu:>9.3f}ms | "
                f"{speedup:>7.2f}x"
            )

    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_benchmark_backward(self):
        """Benchmark forward+backward pass: reference vs pytorch_fallback vs cuda_fused."""
        print("\n" + "=" * 80)
        print("FORWARD + BACKWARD PASS BENCHMARK (median ms, 50 repeats)")
        print("=" * 80)
        header = f"{'N':>6} {'C':>4} {'K':>3} | {'Reference':>10} {'PyTorch':>10} {'CUDA Fused':>10} | {'Speedup':>8}"
        print(header)
        print("-" * len(header))

        for N, C, K in self.CONFIGS:
            t_ref = self._run_backward_benchmark(N, C, K, "reference")
            t_pt  = self._run_backward_benchmark(N, C, K, "pytorch_fallback")
            t_cu  = self._run_backward_benchmark(N, C, K, "cuda_fused")
            speedup = t_ref / t_cu if t_cu > 0 else float("inf")
            print(
                f"{N:>6} {C:>4} {K:>3} | "
                f"{t_ref:>9.3f}ms {t_pt:>9.3f}ms {t_cu:>9.3f}ms | "
                f"{speedup:>7.2f}x"
            )

    @pytest.mark.skipif(not FUSED_KNN_ATTN_CUDA_AVAILABLE, reason="CUDA extension not built")
    def test_benchmark_memory(self):
        """Benchmark peak GPU memory: reference vs cuda_fused."""
        print("\n" + "=" * 80)
        print("PEAK MEMORY BENCHMARK (forward + backward)")
        print("=" * 80)
        header = f"{'N':>6} {'C':>4} {'K':>3} | {'Reference':>12} {'CUDA Fused':>12} | {'Saved':>10}"
        print(header)
        print("-" * len(header))

        for N, C, K in self.CONFIGS:
            q_base, k_base, v_base, idx = _make_test_data(N, C, K)
            scale = C ** -0.5
            grad_out = torch.randn(N, C, device="cuda", dtype=torch.float32)

            # Reference
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            v = v_base.detach().clone().requires_grad_(True)
            out = _reference_knn_attention(q, k, v, idx, scale)
            out.backward(grad_out)
            torch.cuda.synchronize()
            mem_ref = torch.cuda.max_memory_allocated()

            del q, k, v, out
            torch.cuda.empty_cache()

            # CUDA fused
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            v = v_base.detach().clone().requires_grad_(True)
            out = FusedKNNAttentionFunction.apply(q, k, v, idx, scale)
            out.backward(grad_out)
            torch.cuda.synchronize()
            mem_fused = torch.cuda.max_memory_allocated()

            del q, k, v, out
            torch.cuda.empty_cache()

            saved = mem_ref - mem_fused
            print(
                f"{N:>6} {C:>4} {K:>3} | "
                f"{mem_ref / 1024**2:>10.1f}MB {mem_fused / 1024**2:>10.1f}MB | "
                f"{saved / 1024**2:>8.1f}MB"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
