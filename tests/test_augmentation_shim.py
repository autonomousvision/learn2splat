"""Tests for the horizontal-flip augmentation used by DatasetDL3DV.

The augmentation (``apply_augmentation_shim``) mirrors the scene about the
camera's vertical axis: it flips each image along its width and reflects the
camera extrinsics with R = diag(-1, 1, 1, 1). The two operations together must
leave the scene geometrically consistent, i.e. rendering the reflected cameras
reproduces the flipped images.
"""

import torch

from learn2splat.dataset.shims.augmentation_shim import (
    apply_augmentation_shim,
    reflect_extrinsics,
    reflect_views,
)

REFLECT = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))


def make_views(views=3, h=8, w=12):
    """A minimal AnyViews dict with random but valid extrinsics/intrinsics."""
    # Random rotations via QR so extrinsics are proper rigid transforms.
    extrinsics = torch.eye(4).repeat(views, 1, 1)
    for v in range(views):
        q, r = torch.linalg.qr(torch.randn(3, 3))
        q = q @ torch.diag(torch.sign(torch.diagonal(r)))  # make deterministic sign
        extrinsics[v, :3, :3] = q
        extrinsics[v, :3, 3] = torch.randn(3)

    # Normalized intrinsics with a centered principal point (cx = cy = 0.5),
    # which is what the un-modified intrinsics assume under the flip.
    intrinsics = torch.eye(3).repeat(views, 1, 1)
    intrinsics[:, 0, 0] = 0.8  # fx
    intrinsics[:, 1, 1] = 1.1  # fy
    intrinsics[:, 0, 2] = 0.5  # cx
    intrinsics[:, 1, 2] = 0.5  # cy

    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "image": torch.rand(views, 3, h, w),
        "near": torch.full((views,), 0.1),
        "far": torch.full((views,), 100.0),
        "index": torch.arange(views),
    }


def make_example(with_remain=False):
    example = {
        "context": make_views(),
        "target": make_views(),
        "scene": "test_scene",
    }
    if with_remain:
        example["context_remain"] = make_views()
    return example


def project(point_world, extrinsics, intrinsics):
    """Project a world-space point into normalized pixel coords (u, v)."""
    w2c = torch.linalg.inv(extrinsics)
    p_cam = w2c @ torch.cat([point_world, torch.ones(1)])
    uv = intrinsics @ (p_cam[:3] / p_cam[2])
    return uv[:2], p_cam[2]


# --------------------------------------------------------------------------- #
# reflect_extrinsics
# --------------------------------------------------------------------------- #

def test_reflect_extrinsics_matches_sandwich():
    e = make_views()["extrinsics"]
    torch.testing.assert_close(reflect_extrinsics(e), REFLECT @ e @ REFLECT)


def test_reflect_extrinsics_is_involution():
    e = make_views()["extrinsics"]
    torch.testing.assert_close(reflect_extrinsics(reflect_extrinsics(e)), e)


def test_reflect_extrinsics_preserves_rigidity():
    """Reflected extrinsics must remain a valid rigid transform."""
    e = make_views()["extrinsics"]
    refl = reflect_extrinsics(e)
    rot = refl[:, :3, :3]
    # Orthonormal rotation: R R^T = I.
    eye = torch.eye(3).expand_as(rot)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), eye, atol=1e-5, rtol=1e-5)
    # Bottom row unchanged.
    torch.testing.assert_close(refl[:, 3, :], e[:, 3, :])


# --------------------------------------------------------------------------- #
# reflect_views
# --------------------------------------------------------------------------- #

def test_reflect_views_flips_image_width():
    views = make_views()
    out = reflect_views(views)
    torch.testing.assert_close(out["image"], views["image"].flip(-1))


def test_reflect_views_sets_x_flipped():
    out = reflect_views(make_views())
    assert out["x_flipped"] is True


def test_reflect_views_preserves_unrelated_fields():
    views = make_views()
    out = reflect_views(views)
    torch.testing.assert_close(out["intrinsics"], views["intrinsics"])
    torch.testing.assert_close(out["near"], views["near"])
    torch.testing.assert_close(out["far"], views["far"])


# --------------------------------------------------------------------------- #
# Geometric consistency: the heart of the test.
# --------------------------------------------------------------------------- #

def test_flip_preserves_projection_geometry():
    """A world point and its mirror project to horizontally-mirrored pixels.

    Under the flip, image column u maps to (1 - u). For that to be consistent
    with the reflected camera, projecting the mirrored world point R*X through
    the reflected extrinsics must yield u' = 1 - u (with centered principal
    point) and the same v and depth.

    On failure, an annotated PNG is written under ``tests/_viz_out/`` showing
    where each probe point actually lands versus where the flip should put it.
    """
    torch.manual_seed(0)
    views = make_views()
    views["image"][0] = make_pattern_image(*views["image"].shape[-2:])
    refl_views = reflect_views(views)

    # A few probe points spanning the frame, all in front of the camera.
    points = torch.tensor([[0.3, -0.4, 2.0], [-0.5, 0.2, 3.0], [0.1, 0.6, 1.5]])

    failures = []
    projections = []  # (uv, uv_aug, ok) for view 0, for the diagnostic
    for v in range(views["extrinsics"].shape[0]):
        for point in points:
            point_mirror = REFLECT[:3, :3] @ point  # negate x
            uv, depth = project(point, views["extrinsics"][v], views["intrinsics"][v])
            uv_f, depth_f = project(
                point_mirror, refl_views["extrinsics"][v], refl_views["intrinsics"][v]
            )
            ok = (
                torch.allclose(uv_f[0], 1.0 - uv[0], atol=1e-5)
                and torch.allclose(uv_f[1], uv[1], atol=1e-5)
                and torch.allclose(depth_f, depth, atol=1e-5)
            )
            if v == 0:
                projections.append((uv.tolist(), uv_f.tolist(), ok))
            if not ok:
                failures.append((v, point.tolist(), uv.tolist(), uv_f.tolist()))

    if failures:
        path = save_flip_diagnostic(
            views["image"][0], refl_views["image"][0], projections,
            f"{VIZ_DIR}/projection_geometry_FAIL.png",
            title="flip projection consistency — FAIL (yellow X = expected, dot = actual)",
        )
        msg = "\n".join(
            f"  view {v} point {p}: u_aug={uf[0]:.4f} expected {1 - u[0]:.4f}"
            for v, p, u, uf in failures
        )
        raise AssertionError(
            f"{len(failures)} projection mismatch(es); diagnostic: {path}\n{msg}"
        )


# --------------------------------------------------------------------------- #
# apply_augmentation_shim
# --------------------------------------------------------------------------- #

def _is_augmented(out, original):
    return torch.equal(out["context"]["image"], original["context"]["image"].flip(-1))


def test_shim_respects_random_branch():
    """Flip iff the random draw is >= 0.5; either way x_flipped is recorded."""
    for seed in range(20):
        example = make_example()

        g = torch.Generator().manual_seed(seed)
        draw = torch.rand(tuple(), generator=g).item()

        g2 = torch.Generator().manual_seed(seed)
        out = apply_augmentation_shim(example, generator=g2)

        if draw >= 0.5:
            assert _is_augmented(out, example)
            assert out["context"]["x_flipped"] is True
            assert out["target"]["x_flipped"] is True
            torch.testing.assert_close(
                out["context"]["extrinsics"],
                reflect_extrinsics(example["context"]["extrinsics"]),
            )
            torch.testing.assert_close(
                out["target"]["extrinsics"],
                reflect_extrinsics(example["target"]["extrinsics"]),
            )
        else:
            # No-op branch: images/extrinsics unchanged, but x_flipped is now
            # always recorded (False) rather than absent.
            assert out["context"]["x_flipped"] is False
            assert out["target"]["x_flipped"] is False
            torch.testing.assert_close(out["context"]["image"], example["context"]["image"])
            torch.testing.assert_close(
                out["context"]["extrinsics"], example["context"]["extrinsics"]
            )


def test_shim_augments_context_and_target():
    """Find a seed that augments and check both views are flipped together."""
    seed = next(
        s for s in range(20)
        if torch.rand(tuple(), generator=torch.Generator().manual_seed(s)).item() >= 0.5
    )
    example = make_example()
    g = torch.Generator().manual_seed(seed)
    out = apply_augmentation_shim(example, generator=g)

    torch.testing.assert_close(out["context"]["image"], example["context"]["image"].flip(-1))
    torch.testing.assert_close(out["target"]["image"], example["target"]["image"].flip(-1))
    assert out["context"]["x_flipped"] is True
    assert out["target"]["x_flipped"] is True


def test_shim_handles_context_remain():
    """When context_remain is present it must be flipped alongside the rest."""
    seed = next(
        s for s in range(20)
        if torch.rand(tuple(), generator=torch.Generator().manual_seed(s)).item() >= 0.5
    )
    example = make_example(with_remain=True)
    g = torch.Generator().manual_seed(seed)
    out = apply_augmentation_shim(example, generator=g)

    assert "context_remain" in out
    torch.testing.assert_close(
        out["context_remain"]["image"], example["context_remain"]["image"].flip(-1)
    )
    torch.testing.assert_close(
        out["context_remain"]["extrinsics"],
        reflect_extrinsics(example["context_remain"]["extrinsics"]),
    )


# --------------------------------------------------------------------------- #
# Failure diagnostics — only invoked when the geometric check above fails.
#
# The PNG shows the original image (left) and the augmented image (right) with,
# for each probe point: a green/red dot at the point's actual projection, and a
# yellow X on the right marking where the flip *should* have landed it (the
# horizontal mirror 1 - u of the original). A visible gap between the right-hand
# dot and X is exactly the error the geometric assertion reports.
# --------------------------------------------------------------------------- #

VIZ_DIR = "tests/_viz_out"


def make_pattern_image(h=64, w=96):
    """An asymmetric RGB pattern so a horizontal flip is visually obvious.

    Left edge is bright red, a green vertical bar sits at ~1/4 width, and a blue
    ramp brightens toward the right. None of it is left-right symmetric.
    """
    img = torch.zeros(3, h, w)
    xs = torch.linspace(0, 1, w)
    img[2] = xs[None, :].expand(h, w)  # blue ramp brightening rightward
    img[0, :, : w // 8] = 1.0  # red block on the far left
    img[1, :, w // 4 : w // 4 + max(1, w // 24)] = 1.0  # green bar at 1/4 width
    # A diagonal white streak (breaks any residual symmetry).
    for y in range(h):
        x = int((y / h) * w)
        img[:, y, max(0, x - 1) : x + 1] = 1.0
    return img


def save_flip_diagnostic(image, image_aug, projections, out_path, title=""):
    """Render an annotated original-vs-augmented comparison.

    Args:
        image:      original view image, [3, H, W] in [0, 1].
        image_aug:  augmented (flipped) view image, [3, H, W].
        projections: list of (uv, uv_aug, ok) with uv/uv_aug normalized (u, v)
            and ok a bool for whether this probe satisfied u_aug == 1 - u.
        out_path:   where to write the PNG.
        title:      figure suptitle.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = image.shape[-2:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    axes[0].set_title("original")
    axes[1].imshow(image_aug.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
    axes[1].set_title("augmented = flip(image) + reflect(extrinsics)")

    # Mirror line at u = 0.5 on both panels for reference.
    for ax in axes:
        ax.axvline(0.5 * w, color="white", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

    for i, (uv, uv_aug, ok) in enumerate(projections):
        color = "lime" if ok else "red"
        # Actual projection in each image.
        axes[0].scatter(uv[0] * w, uv[1] * h, c=color, s=70, edgecolors="black", zorder=3)
        axes[0].annotate(str(i), (uv[0] * w, uv[1] * h), color="white", fontsize=8)
        axes[1].scatter(
            uv_aug[0] * w, uv_aug[1] * h, c=color, s=70, edgecolors="black", zorder=3,
            label="reflected-camera projection" if i == 0 else None,
        )
        # Where the flip *should* land it: horizontal mirror of the original.
        axes[1].scatter(
            (1 - uv[0]) * w, uv[1] * h, marker="x", c="yellow", s=110, lw=2.5, zorder=4,
            label="expected mirror (1 - u)" if i == 0 else None,
        )
    axes[1].legend(loc="lower center", fontsize=8, framealpha=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
