"""Convention round-trip check (no CUDA, no checkpoint).

The learn2splat Gaussians PLY round-trip exercises the exact schema inria's
GaussianModel.save_ply/load_ply uses, so it validates the Gaussian bridge
end to end.
"""

import torch


def test_ply_roundtrip_preserves_gaussians(tmp_path):
    from learn2splat.model.ply_export import load_gaussians_ply, save_gaussian_ply
    from learn2splat.model.types import Gaussians

    torch.manual_seed(0)
    n, sh_deg = 256, 3
    d_sh = (sh_deg + 1) ** 2

    means = torch.randn(1, n, 3)
    harmonics = torch.randn(1, n, 3, d_sh)
    opacities = torch.empty(1, n).uniform_(0.2, 0.8)  # away from sigmoid saturation
    scales = torch.empty(1, n, 3).uniform_(0.01, 1.0)  # positive (post-exp space)
    quats = torch.nn.functional.normalize(torch.randn(1, n, 4), dim=-1)

    g = Gaussians(
        means=means,
        harmonics=harmonics,
        opacities=opacities,
        scales=scales,
        rotations_unnorm=quats,
        rotations=quats,
    )

    ply = tmp_path / "rt.ply"
    save_gaussian_ply(g, save_path=ply)
    assert ply.exists()

    g2 = load_gaussians_ply(str(ply), max_sh_degree=sh_deg)

    torch.testing.assert_close(g2.means, means, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(g2.scales, scales, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(g2.opacities, opacities, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(g2.harmonics, harmonics, atol=1e-4, rtol=1e-4)
    # quaternions: xyzw->wxyz (save) -> wxyz->xyzw + normalize (load); sign-stable
    torch.testing.assert_close(g2.rotations, quats, atol=1e-5, rtol=1e-5)
