"""Post-install smoke test for the compiled CUDA submodules.

Each extension is *imported* and then made to *launch a kernel* on this machine's
GPU. The launch is the point: an extension built for the wrong compute arch
imports fine and only fails when a kernel runs ("not compiled for SM XX",
cudaErrorInvalidDevice) — which an import-time check can't catch. Run this on a
GPU node after setup.sh to confirm the build matches the hardware:

    pytest tests/test_cuda_extensions.py -v

The whole module is skipped without CUDA (e.g. on a login node).
"""
import os
import sys

import pytest
import torch

# Make the in-tree `fused_knn_attn` importable when it isn't pip-installed (it ships as a submodule
# package). Append (not insert at 0) so the installed `pointops` wins over the unbuilt
# `submodules/pointops` (whose `_C` ext isn't compiled).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "submodules"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _points(n: int = 256) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(n, 3, device="cuda")


def _offset(n: int) -> torch.Tensor:
    # pointops marks batch boundaries by cumulative point counts; one scene of n points -> [n].
    return torch.tensor([n], dtype=torch.int32, device="cuda")


def test_simple_knn_distcuda2():
    from simple_knn._C import distCUDA2

    out = distCUDA2(_points().contiguous())  # [N] squared NN distances
    assert out.shape == (256,) and out.is_cuda and torch.isfinite(out).all()


def test_pointops_knn_query():
    import pointops

    p, o = _points(), _offset(256)
    idx, _ = pointops.knn_query(4, p, o, p, o)
    assert idx.shape == (256, 4) and idx.is_cuda


def test_simple_knn_knn_indices():
    from simple_knn._C import knn_indices

    p, o = _points(), _offset(256)
    idx, _ = knn_indices(p, 4, o)  # [N, K] global neighbour indices
    assert idx.shape == (256, 4) and idx.is_cuda
    # Self-inclusive: each point is among its own K nearest neighbours.
    self_idx = torch.arange(256, device="cuda")[:, None]
    assert (idx.long() == self_idx).any(dim=1).all()


def test_knn_indices_matches_pointops():
    """simple_knn.knn_indices is the point transformer's default (and mandatory) KNN backend;
    it must return the same neighbour sets as the pointops.knn_query reference it replaced."""
    import pointops
    from simple_knn._C import knn_indices

    p, o, k = _points(), _offset(256), 4
    ref, _ = pointops.knn_query(k, p, o, p, o)
    got, _ = knn_indices(p, k, o)
    # Neighbour order within a row may differ between backends; compare the sets per point.
    assert torch.equal(ref.long().sort(dim=1).values, got.long().sort(dim=1).values)


def test_fused_ssim():
    from fused_ssim import fused_ssim

    a = torch.rand(1, 3, 32, 32, device="cuda")
    b = torch.rand(1, 3, 32, 32, device="cuda")
    score = fused_ssim(a, b, padding="valid")
    assert torch.isfinite(score).all()


def test_fused_knn_attention():
    import pointops
    from fused_knn_attn import fused_knn_attention

    n, c, k = 256, 16, 4
    p = _points(n)
    idx, _ = pointops.knn_query(k, p, _offset(n), p, _offset(n))
    q, key, v = (torch.randn(n, c, device="cuda") for _ in range(3))
    out = fused_knn_attention(q, key, v, idx.int(), scale=c**-0.5)
    assert out.shape == (n, c) and torch.isfinite(out).all()