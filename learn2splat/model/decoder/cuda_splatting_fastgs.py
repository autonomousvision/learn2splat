"""FastGS rasterization wrappers.

Mirrors :mod:`cuda_splatting` (the Inria backend) but targets the FastGS
fork ``diff_gaussian_rasterization_fastgs``. The fork's API differs from the
original: the SH DC term is passed separately from the rest (``dc`` vs ``shs``),
the rasterization settings carry ``mult`` / ``get_flag`` / ``metric_map``, the
screen-space means tensor is ``[N, 4]``, and ``forward`` returns a third value
(``accum_metric_counts``) that we drop.

Kept as a separate module (with its own guarded import + a local copy of
``get_projection_matrix``) so importing the FastGS backend does NOT require the
``diff_gaussian_rasterization`` package, and vice versa.
"""

from dataclasses import dataclass
from math import isqrt
from typing import Literal

import torch

try:
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as e:
    raise ImportError(
        "The fastgs decoder requires diff_gaussian_rasterization_fastgs, which "
        "is not installed. Build it with: WITH_OPTIONAL_RASTERIZERS=1 bash setup.sh"
    ) from e

from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Local copy of ``cuda_splatting.get_projection_matrix`` (see module docstring).
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


@dataclass
class _FastGSViews:
    """Per-view FastGS rasterizer inputs after the scale-invariant rescale + projection setup.
    Shared by ``render_cuda_fastgs`` (forward) and ``render_metric_counts_fastgs`` (scoring)."""
    extrinsics: Tensor        # [b, 4, 4], rescaled
    means: Tensor             # [b, g, 3], rescaled
    covariances: Tensor | None
    scales: Tensor | None     # [b, g, 3], rescaled (None when covariances supplied)
    shs: Tensor               # [b, g, n, 3]
    degree: int
    view_matrix: Tensor       # [b, 4, 4]
    full_projection: Tensor   # [b, 4, 4]
    tan_fov_x: Tensor         # [b]
    tan_fov_y: Tensor         # [b]


def _fastgs_view_setup(
    extrinsics, intrinsics, near, far, gaussian_means, gaussian_covariances,
    gaussian_scales, gaussian_sh_coefficients, scale_invariant,
) -> _FastGSViews:
    """Scale-invariant rescale (keeps the rasterizer numerically well-conditioned) + camera
    projection + SH reshape — the setup common to both FastGS render paths."""
    using_cov = gaussian_covariances is not None
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        if using_cov:
            gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        else:
            gaussian_scales = gaussian_scales * scale[:, None, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        near, far = near * scale, far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    # [b, g, xyz, n] -> [b, g, n, xyz]; FastGS wants the DC term (n=0) split from the rest.
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    projection_matrix = rearrange(get_projection_matrix(near, far, fov_x, fov_y), "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    return _FastGSViews(
        extrinsics=extrinsics, means=gaussian_means, covariances=gaussian_covariances,
        scales=gaussian_scales, shs=shs, degree=isqrt(n) - 1, view_matrix=view_matrix,
        full_projection=view_matrix @ projection_matrix,
        tan_fov_x=(0.5 * fov_x).tan(), tan_fov_y=(0.5 * fov_y).tan(),
    )


def _fastgs_settings(views: _FastGSViews, i, background_color, mult, image_shape, get_flag, metric_map):
    """Per-view GaussianRasterizationSettings; only get_flag/metric_map differ across the two paths."""
    h, w = image_shape
    return GaussianRasterizationSettings(
        image_height=h, image_width=w,
        tanfovx=views.tan_fov_x[i].item(), tanfovy=views.tan_fov_y[i].item(),
        bg=background_color[i], scale_modifier=1.0,
        viewmatrix=views.view_matrix[i], projmatrix=views.full_projection[i],
        sh_degree=views.degree, campos=views.extrinsics[i, :3, 3], mult=mult,
        prefiltered=False, debug=False, get_flag=get_flag, metric_map=metric_map,
    )


def _fastgs_appearance_kwargs(views: _FastGSViews, i, use_sh, gaussian_rotations) -> dict:
    """Per-view SH/appearance + geometry rasterizer kwargs (DC/rest split; covariance or scale+rot)."""
    kw = {}
    if use_sh:
        # FastGS splits the DC term from the rest; the rest stays a tensor ([N, 0, 3] at degree 0),
        # not None — the rasterizer validates on shs.
        kw["dc"] = views.shs[i, :, :1].contiguous()  # [N, 1, 3]
        kw["shs"] = views.shs[i, :, 1:].contiguous()  # [N, n-1, 3]
    else:
        kw["colors_precomp"] = views.shs[i, :, 0, :]
    if views.covariances is not None:
        row, col = torch.triu_indices(3, 3)
        kw["cov3D_precomp"] = views.covariances[i, :, row, col]
    else:
        kw["scales"] = views.scales[i]
        kw["rotations"] = gaussian_rotations[i]
    return kw


def render_cuda_fastgs(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"] | None,
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
    gaussian_scales: Float[Tensor, "batch gaussian 3"] | None = None,
    gaussian_rotations: Float[Tensor, "batch gaussian 4"] | None = None,
    mult: float = 0.5,
    means2d_out: Float[Tensor, "batch gaussian 2"] | None = None,
    means2d_abs_out: Float[Tensor, "batch gaussian 2"] | None = None,
) -> tuple[
    Float[Tensor, "batch 3 height width"],
    Int[Tensor, "batch gaussian"],
]:
    # means2d_out: see render_cuda. FastGS's screen-space tensor is [N, 4] (cols [:2] = the
    # standard 2D-mean gradient, [2:] = its abs-gradient densification signal); the standard half
    # is exposed via autograd.grad(loss, means2d_out). When means2d_abs_out is supplied it backs
    # cols [2:], so autograd.grad(loss, means2d_abs_out) yields FastGS's Abs-GS split signal.
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    # Exactly one of (covariances) or (scales+rotations) must be supplied.
    using_cov = gaussian_covariances is not None
    using_sr = gaussian_scales is not None and gaussian_rotations is not None
    assert using_cov ^ using_sr, "Provide either gaussian_covariances or (gaussian_scales+gaussian_rotations)."

    views = _fastgs_view_setup(extrinsics, intrinsics, near, far, gaussian_means,
                               gaussian_covariances, gaussian_scales, gaussian_sh_coefficients, scale_invariant)
    b, _, _ = extrinsics.shape
    h, w = image_shape
    n_gauss = gaussian_means.shape[1]
    if means2d_out is None:
        means2d_out = torch.zeros((b, n_gauss, 2), dtype=gaussian_means.dtype,
                                  device=gaussian_means.device, requires_grad=True)

    all_images, all_radii = [], []
    for i in range(b):
        # FastGS's [N, 4] screen tensor: cols [:2] take the standard 2D-mean gradient (routed to
        # means2d_out via autograd), cols [2:] the abs-gradient — routed to means2d_abs_out when
        # supplied, else a detached zero pad (abs-gradient discarded).
        abs_half = means2d_abs_out[i] if means2d_abs_out is not None \
            else torch.zeros((n_gauss, 2), dtype=gaussian_means.dtype, device=gaussian_means.device)
        means2D = torch.cat([means2d_out[i], abs_half], dim=1)  # [N, 4]
        metric_map = torch.zeros(h * w, dtype=torch.int, device=gaussian_means.device)  # required, unused here

        rasterizer = GaussianRasterizer(
            _fastgs_settings(views, i, background_color, mult, image_shape, False, metric_map))
        kw = dict(means3D=views.means[i], means2D=means2D, opacities=gaussian_opacities[i, ..., None])
        kw.update(_fastgs_appearance_kwargs(views, i, use_sh, gaussian_rotations))

        # FastGS returns (image, radii, accum_metric_counts); the normal forward drops the counts.
        image, radii, _ = rasterizer(**kw)
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images), torch.stack(all_radii)


def render_metric_counts_fastgs(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"] | None,
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    metric_maps: Int[Tensor, "batch height_width"],
    scale_invariant: bool = True,
    gaussian_scales: Float[Tensor, "batch gaussian 3"] | None = None,
    gaussian_rotations: Float[Tensor, "batch gaussian 4"] | None = None,
    mult: float = 0.5,
) -> Int[Tensor, "batch gaussian"]:
    """FastGS multi-view scoring render: rasterize each view with ``get_flag=True`` and a binary
    high-error ``metric_map`` (flattened [H*W]), returning per-Gaussian ``accum_metric_counts``
    (how many flagged pixels each Gaussian contributed to). No gradients are needed — call under
    no_grad. Mirrors the second ``render_fastgs`` call in FastGS ``compute_gaussian_score_fastgs``."""
    using_cov = gaussian_covariances is not None
    using_sr = gaussian_scales is not None and gaussian_rotations is not None
    assert using_cov ^ using_sr, "Provide either gaussian_covariances or (gaussian_scales+gaussian_rotations)."

    views = _fastgs_view_setup(extrinsics, intrinsics, near, far, gaussian_means,
                               gaussian_covariances, gaussian_scales, gaussian_sh_coefficients, scale_invariant)
    b, _, _ = extrinsics.shape
    n_gauss = gaussian_means.shape[1]

    all_counts = []
    for i in range(b):
        rasterizer = GaussianRasterizer(_fastgs_settings(
            views, i, background_color, mult, image_shape, True, metric_maps[i].int().contiguous()))
        # Screen tensor is unused here (no grad), but the rasterizer still expects an [N, 4] means2D.
        means2D = torch.zeros((n_gauss, 4), dtype=gaussian_means.dtype, device=gaussian_means.device)
        kw = dict(means3D=views.means[i], means2D=means2D, opacities=gaussian_opacities[i, ..., None])
        kw.update(_fastgs_appearance_kwargs(views, i, True, gaussian_rotations))

        _, _, accum_metric_counts = rasterizer(**kw)
        all_counts.append(accum_metric_counts)
    return torch.stack(all_counts)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda_fastgs(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
    mult: float = 0.5,
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    images, _ = render_cuda_fastgs(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
        mult=mult,
    )
    return images.mean(dim=1)
