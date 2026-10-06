"""Parity smoke tests for every registered decoder backend.

Drives each available decoder (``gsplat`` always; ``inria`` and ``fastgs`` when
their optional CUDA rasterizers are installed) with the same synthetic Gaussians
and cameras, and asserts a sane render: correct output shapes, finite pixels in
range, some Gaussians visible, and gradients flowing back to the means.

Backends whose CUDA extension is missing are skipped (the registry replaces them
with an import stub, which is a function rather than a class). Skipped without CUDA.
"""

import types

import pytest
import torch

from learn2splat.model.decoder import DECODERS, get_decoder

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="decoders need CUDA rasterizers"
)

DECODER_NAMES = ["gsplat", "inria", "fastgs"]


def _backend_available(name: str) -> bool:
    # Available backends register a class; missing ones register an import-stub function.
    return isinstance(DECODERS.get(name), type)


def _make_cfg(name: str):
    if name == "gsplat":
        from learn2splat.model.decoder.gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg
        return GSplatDecoderSplattingCUDACfg(
            name="gsplat", use_covariances=False, rasterize_mode="antialiased", eps2d=0.3
        )
    if name == "inria":
        from learn2splat.model.decoder.decoder_splatting_cuda import InriaDecoderSplattingCUDACfg
        return InriaDecoderSplattingCUDACfg(name="inria", scale_invariant=False, use_covariances=False)
    if name == "fastgs":
        from learn2splat.model.decoder.fastgs_decoder_splatting_cuda import FastGSDecoderSplattingCUDACfg
        return FastGSDecoderSplattingCUDACfg(
            name="fastgs", scale_invariant=False, use_covariances=False, mult=0.5
        )
    raise ValueError(name)


def _synthetic_gaussians(n=2000, d_sh=4, device="cuda", requires_grad=False):
    from learn2splat.model.types import Gaussians

    torch.manual_seed(0)
    means = (torch.randn(1, n, 3, device=device) * 0.3).requires_grad_(requires_grad)
    harmonics = torch.rand(1, n, 3, d_sh, device=device)
    opacities = torch.full((1, n), 0.7, device=device)  # activated
    scales = torch.full((1, n, 3), 0.02, device=device)  # activated
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # xyzw identity
    quats = quat.expand(1, n, 4).contiguous()
    # Covariances (needed by the inria/fastgs depth path): identity rotation -> diag(scales^2).
    covariances = torch.diag_embed(scales**2)  # [1, n, 3, 3]
    return Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations_unnorm=quats,
        rotations=quats,
        covariances=covariances,
    )


def _cameras(v=3, hw=128, device="cuda"):
    # OpenCV camera-to-world: cameras backed off along -Z looking at the origin (+Z into screen).
    ext = torch.eye(4, device=device).repeat(1, v, 1, 1)  # [1, V, 4, 4]
    ext[..., 2, 3] = -3.0
    # Normalized intrinsics: focal = image dimension (fx_norm = fy_norm = 1), principal point centered.
    intr = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], device=device)
    intr = intr.repeat(1, v, 1, 1)  # [1, V, 3, 3]
    near = torch.full((1, v), 0.1, device=device)
    far = torch.full((1, v), 100.0, device=device)
    return ext, intr, near, far


def _build(name, device="cuda"):
    return get_decoder(
        _make_cfg(name),
        types.SimpleNamespace(background_color=[0.0, 0.0, 0.0]),
    ).to(device)


@pytest.mark.parametrize("name", DECODER_NAMES)
@pytest.mark.parametrize("d_sh", [1, 4])  # sh degree 0 and 1 (exercises FastGS dc/rest split)
def test_decoder_renders(name, d_sh):
    if not _backend_available(name):
        pytest.skip(f"{name} backend not installed")

    device = torch.device("cuda")
    v, hw, n = 3, 128, 2000
    g = _synthetic_gaussians(n=n, d_sh=d_sh, device=device)
    ext, intr, near, far = _cameras(v=v, hw=hw, device=device)

    dec = _build(name, device)
    out = dec.forward(g, ext, intr, near, far, image_shape=(hw, hw), depth_mode="depth")

    # Color: [B, V, 3, H, W], finite, non-negative, not empty. (No tight upper bound: SH
    # degree 1 is view-dependent and can legitimately exceed 1.0 before any clamp.)
    assert out.color.shape == (1, v, 3, hw, hw), out.color.shape
    assert torch.isfinite(out.color).all(), f"{name}: non-finite pixels"
    assert out.color.min() >= -1e-3, (name, float(out.color.min()))
    assert out.color.max() < 10.0, (name, float(out.color.max()))  # sanity ceiling
    assert out.color.sum() > 0, f"{name}: rendered an empty image"

    # Visibility / radii / means2d shapes consistent with the gsplat interface.
    assert out.visibility_filter.shape == (1, v, n), out.visibility_filter.shape
    assert out.visibility_filter.any(), f"{name}: no Gaussians visible"
    assert out.radii.shape == (1, v, n, 2), out.radii.shape
    assert out.means2d.shape == (1, v, n, 2), out.means2d.shape

    # Depth: [B, V, H, W], finite.
    assert out.depth.shape == (1, v, hw, hw), out.depth.shape
    assert torch.isfinite(out.depth).all(), f"{name}: non-finite depth"


@pytest.mark.parametrize("name", DECODER_NAMES)
def test_decoder_backward_to_means(name):
    if not _backend_available(name):
        pytest.skip(f"{name} backend not installed")

    device = torch.device("cuda")
    g = _synthetic_gaussians(n=1000, d_sh=4, device=device, requires_grad=True)
    ext, intr, near, far = _cameras(v=2, hw=96, device=device)

    dec = _build(name, device)
    out = dec.forward(g, ext, intr, near, far, image_shape=(96, 96))
    out.color.mean().backward()

    assert g.means.grad is not None, f"{name}: no gradient on means"
    assert torch.isfinite(g.means.grad).all(), f"{name}: non-finite means gradient"
    assert (g.means.grad != 0).any(), f"{name}: zero means gradient"


def test_all_decoders_agree_on_shapes():
    """Every available backend must return identically-shaped outputs for the same input."""
    device = torch.device("cuda")
    available = [n for n in DECODER_NAMES if _backend_available(n)]
    assert "gsplat" in available  # the default backend must always be present

    g = _synthetic_gaussians(n=500, d_sh=4, device=device)
    ext, intr, near, far = _cameras(v=2, hw=64, device=device)

    shapes = {}
    for name in available:
        out = _build(name, device).forward(g, ext, intr, near, far, image_shape=(64, 64))
        shapes[name] = (out.color.shape, out.radii.shape, out.visibility_filter.shape)

    ref = shapes["gsplat"]
    for name, shp in shapes.items():
        assert shp == ref, f"{name} shapes {shp} != gsplat {ref}"
