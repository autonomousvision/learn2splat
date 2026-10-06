from inspect import signature
from math import isqrt
from typing import Literal

import torch

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError as e:
    raise ImportError(
        "The inria decoder requires diff_gaussian_rasterization, which is "
        "not installed. Install it with: "
        "pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"
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
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
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


def render_cuda(
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
    means2d_out: Float[Tensor, "batch gaussian 2"] | None = None,
) -> tuple[
    Float[Tensor, "batch 3 height width"],
    Int[Tensor, "batch gaussian"],
]:
    # means2d_out: a [B, N, 2] tensor whose slices become the rasterizer's screen-space means
    # input. The 2D screen-space gradient is then reachable via torch.autograd.grad(loss,
    # means2d_out) — no .backward()/.grad needed (see splatting_cuda_decoder). When None
    # (e.g. the depth pass) a throwaway buffer is used.
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    # Exactly one of (covariances) or (scales+rotations) must be supplied.
    using_cov = gaussian_covariances is not None
    using_sr = gaussian_scales is not None and gaussian_rotations is not None
    assert using_cov ^ using_sr, "Provide either gaussian_covariances or (gaussian_scales+gaussian_rotations)."

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        if using_cov:
            gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        else:
            gaussian_scales = gaussian_scales * scale[:, None, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    cxs = intrinsics[:, 0, 2] * w
    cys = intrinsics[:, 1, 2] * h

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    # Some forks' settings carry cx/cy/prepare_for_gsgn_backward; the original Inria
    # rasterizer does not.
    # The accelerated-3DGS build (graphdeco-inria@3dgs_accel) instead requires an `antialiasing` flag.
    _settings_fields = set(GaussianRasterizationSettings._fields)
    _fork_has_cxcy = "cx" in _settings_fields and "cy" in _settings_fields
    _fork_has_gsgn = "prepare_for_gsgn_backward" in _settings_fields
    _has_antialiasing = "antialiasing" in _settings_fields
    # The accelerated-3DGS build splits the SH DC term from the rest (separate `dc` arg);
    # the original Inria rasterizer and most forks take the full SH as a single `shs` arg.
    _wants_dc = "dc" in signature(GaussianRasterizer.forward).parameters

    n_gauss = gaussian_means.shape[1]
    if means2d_out is None:
        means2d_out = torch.zeros((b, n_gauss, 2), dtype=gaussian_means.dtype,
                                  device=gaussian_means.device, requires_grad=True)

    all_images = []
    all_radii = []
    for i in range(b):
        # Screen-space means input for the kernel: [means2d_out[i] | 0] so the rasterizer's
        # backward routes the 2D gradient into means2d_out (reachable via autograd.grad).
        # The Inria screenspace is [N, 3]; the extra column carries no useful gradient.
        pad = torch.zeros((n_gauss, 1), dtype=gaussian_means.dtype, device=gaussian_means.device)
        mean_gradients = torch.cat([means2d_out[i], pad], dim=1)  # [N, 3]

        settings_kwargs = dict(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            debug=False,
        )
        if _fork_has_cxcy:
            settings_kwargs["cx"] = float(cxs[i].item())
            settings_kwargs["cy"] = float(cys[i].item())
        if _fork_has_gsgn:
            settings_kwargs["prepare_for_gsgn_backward"] = False
        if _has_antialiasing:
            settings_kwargs["antialiasing"] = False
        settings = GaussianRasterizationSettings(**settings_kwargs)
        rasterizer = GaussianRasterizer(settings)

        raster_kwargs = dict(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
        )
        if use_sh and _wants_dc:
            # Accelerated build: DC term split out; the rest stays a tensor (empty [N,0,3]
            # at degree 0), not None — the kernel validates on `shs`.
            raster_kwargs["dc"] = shs[i, :, :1].contiguous()
            raster_kwargs["shs"] = shs[i, :, 1:].contiguous()
        else:
            raster_kwargs["shs"] = shs[i] if use_sh else None
        if using_cov:
            row, col = torch.triu_indices(3, 3)
            raster_kwargs["cov3D_precomp"] = gaussian_covariances[i, :, row, col]
        else:
            raster_kwargs["scales"] = gaussian_scales[i]
            raster_kwargs["rotations"] = gaussian_rotations[i]
        out = rasterizer(**raster_kwargs)
        # The original returns (image, radii); some forks return (image, radii, n_contrib, is_hit).
        image, radii = out[0], out[1]
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images), torch.stack(all_radii)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch: hand the transformed camera parameters back to the caller.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
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
    images, _ = render_cuda(
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
    )
    return images.mean(dim=1)
