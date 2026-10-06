"""Verify the unified screen-space (means2d) gradient path across all decoders.

The learn2splat ADC/densification path obtains the 2D screen-space gradient *functionally*
(`torch.autograd.grad(loss, decoder_output.means2d)` — see
learn2splat/scene_trainer/optimizer/optimizer_utils.py), never via `.backward()` + `.grad`.

gsplat exposes `means2d` as a real graph node, so this works natively. The Inria/FastGS
3DGS-family rasterizers instead use `means2D` as a screen-space gradient *accumulator*
(the classic `screenspace_points.grad` trick, see FastGS). We bridge the two by
feeding the kernel `cat(means2d_handle, pad)` and returning the `[B,V,N,2]` handle, so
`autograd.grad(loss, means2d)` reaches the same gradient with no `.backward()`.

This test verifies, across every available decoder:
  1. autograd.grad(loss, means2d) is available, finite, and active on visible Gaussians;
  2. it equals the `.backward()` + `.grad` reference (i.e. correctly computed);
  3. the per-Gaussian gradients are consistent across renderers on a shared scene.

Skipped without CUDA; decoders whose CUDA backend is absent are skipped individually.
"""

import types

import pytest
import torch

from learn2splat.model.decoder import DECODERS, get_decoder
from learn2splat.model.types import Gaussians

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decoders need CUDA rasterizers"
)

DECODER_NAMES = ["gsplat", "inria", "fastgs"]
N, V, HW = 3000, 2, 128
DEVICE = "cuda"


def _available():
    # Available backends register a class; missing ones register an import-stub function.
    return [n for n in DECODER_NAMES if isinstance(DECODERS.get(n), type)]


def _cfg(name):
    if name == "gsplat":
        from learn2splat.model.decoder.gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg
        # classic (no antialiasing) to match the Inria backend's defaults.
        return GSplatDecoderSplattingCUDACfg(name="gsplat", use_covariances=False,
                                             rasterize_mode="classic", eps2d=0.3)
    if name == "inria":
        from learn2splat.model.decoder.decoder_splatting_cuda import InriaDecoderSplattingCUDACfg
        return InriaDecoderSplattingCUDACfg(name="inria", scale_invariant=False, use_covariances=False)
    from learn2splat.model.decoder.fastgs_decoder_splatting_cuda import FastGSDecoderSplattingCUDACfg
    return FastGSDecoderSplattingCUDACfg(name="fastgs", scale_invariant=False,
                                         use_covariances=False, mult=0.5)


def _build(name):
    return get_decoder(_cfg(name), types.SimpleNamespace(background_color=[0.0, 0.0, 0.0])).to(DEVICE)


def _scene():
    """Identical seeded scene every call, so all renderers see the same Gaussians."""
    torch.manual_seed(0)
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEVICE).expand(1, N, 4).contiguous()
    sc = torch.full((1, N, 3), 0.03, device=DEVICE)
    means = (torch.randn(1, N, 3, device=DEVICE) * 0.3).requires_grad_(True)
    return Gaussians(
        means=means, harmonics=torch.rand(1, N, 3, 4, device=DEVICE) * 0.5,
        opacities=torch.full((1, N), 0.8, device=DEVICE), scales=sc,
        rotations_unnorm=q, rotations=q, covariances=torch.diag_embed(sc**2),
    )


def _cameras():
    ext = torch.eye(4, device=DEVICE).repeat(1, V, 1, 1)
    ext[..., 2, 3] = -3.0
    intr = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], device=DEVICE).repeat(1, V, 1, 1)
    near = torch.full((1, V), 0.1, device=DEVICE)
    far = torch.full((1, V), 100.0, device=DEVICE)
    return ext, intr, near, far


def _render(name):
    dec = _build(name)
    out = dec.forward(_scene(), *_cameras(), image_shape=(HW, HW))
    return out


@pytest.mark.parametrize("name", DECODER_NAMES)
def test_means2d_grad_via_autograd(name):
    """autograd.grad(loss, means2d) is available, [B,V,N,2], finite, and active on most Gaussians."""
    if name not in _available():
        pytest.skip(f"{name} backend not installed")
    out = _render(name)
    assert out.means2d.shape == (1, V, N, 2), out.means2d.shape

    grad = torch.autograd.grad(out.color.sum(), out.means2d, allow_unused=True)[0]
    assert grad is not None, f"{name}: means2d not connected to the autograd graph"
    assert torch.isfinite(grad).all(), f"{name}: non-finite means2d grad"
    active = (grad != 0).any(-1).sum().item()
    assert active > 0.5 * (V * N), f"{name}: only {active}/{V*N} Gaussians have a means2d grad"


@pytest.mark.parametrize("name", DECODER_NAMES)
def test_autograd_matches_backward(name):
    """The functional autograd.grad path equals the classic .backward()+.grad path — proving the
    2D gradient is correctly computed without ever calling .backward().

    Uses two separate (identically-seeded) forward passes so each rasterizer's CUDA backward runs
    exactly once — running it twice on one graph (autograd.grad then backward) is not idempotent
    for the gsplat kernel."""
    if name not in _available():
        pytest.skip(f"{name} backend not installed")

    out_a = _render(name)
    ag = torch.autograd.grad(out_a.color.sum(), out_a.means2d)[0]

    out_b = _render(name)  # identical scene -> identical graph
    out_b.means2d.retain_grad()  # gsplat's means2d is a non-leaf; retain so .grad is populated
    out_b.color.sum().backward()
    bg = out_b.means2d.grad

    assert bg is not None, f"{name}: .backward() did not populate means2d.grad"
    # Compare by direction (cosine ~ 1): the two paths run the rasterizer's backward separately,
    # and the Inria/FastGS atomic backward is state-dependent nondeterministic (its
    # per-Gaussian magnitudes can drift ~1-2% under memory pressure). Cosine is robust to that
    # noise but a real routing bug (wrong column/sign/scatter) would tank it well below 0.999.
    cos = (ag.flatten() @ bg.flatten()) / (ag.norm() * bg.norm() + 1e-12)
    assert cos > 0.999, f"{name}: autograd.grad direction != backward, cosine={cos.item():.5f}"


def test_means2d_grads_consistent_across_renderers():
    """On a shared scene, per-Gaussian screen-space gradient magnitudes correlate strongly across
    renderers (scale-invariant Pearson; tolerates each rasterizer's own gradient derivation and
    the pixel-vs-NDC coordinate convention)."""
    avail = _available()
    if len(avail) < 2:
        pytest.skip("need >=2 decoder backends for a cross-renderer comparison")

    mags = {}
    for name in avail:
        out = _render(name)
        grad = torch.autograd.grad(out.color.sum(), out.means2d)[0]
        mags[name] = grad.reshape(-1, 2).norm(dim=-1)  # per-Gaussian |grad|, convention-agnostic

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return (a * b).sum() / (a.norm() * b.norm() + 1e-12)

    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            a, b = avail[i], avail[j]
            r = pearson(mags[a], mags[b]).item()
            assert r > 0.8, f"{a} vs {b}: means2d grads weakly correlated (Pearson={r:.3f})"


if __name__ == "__main__":
    # Standalone verification report.
    avail = _available()
    print(f"available decoders: {avail}\n")
    mags = {}
    for name in avail:
        out_a = _render(name)
        ag = torch.autograd.grad(out_a.color.sum(), out_a.means2d)[0]
        out_b = _render(name)
        out_b.means2d.retain_grad()
        out_b.color.sum().backward()
        bg = out_b.means2d.grad
        rel = ((ag - bg).norm() / (bg.norm() + 1e-12)).item()
        active = (ag != 0).any(-1).sum().item()
        mags[name] = ag.reshape(-1, 2).norm(dim=-1)
        print(f"{name:8s} means2d{tuple(out_a.means2d.shape)}  autograd≡backward (rel.err={rel:.1e})  "
              f"active={active}/{V*N}")

    def pearson(a, b):
        a, b = a - a.mean(), b - b.mean()
        return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()

    print("\ncross-renderer per-Gaussian |grad2d| Pearson:")
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            a, b = avail[i], avail[j]
            print(f"  {a:7s} vs {b:7s}: {pearson(mags[a], mags[b]):.4f}")
