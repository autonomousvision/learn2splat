"""Convention bridges between inria 3DGS objects and learn2splat types.

* Gaussians: via the original-3DGS PLY schema. ``learn2splat/model/ply_export.py``
  (``load_gaussians_ply`` / ``save_gaussian_ply``) and inria
  ``GaussianModel.save_ply`` / ``load_ply`` write/read the *same* schema
  (scales log<->exp, opacity logit<->sigmoid, quat wxyz<->xyzw, SH dc/rest),
  so a PLY round-trip is convention-correct by construction.

* Cameras: inria stores world->camera ``R,T`` + FoV (COLMAP convention);
  learn2splat wants camera->world extrinsics and image-size-normalized intrinsics.
  We reuse inria's own ``getWorld2View2`` / ``fov2focal`` for the forward
  direction so the round trip through ``learn2splat.geometry.projection.get_fov`` is
  exact.

All inria imports are deferred into functions (the inria repo is only on
``sys.path`` when the caller runs from it).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from learn2splat.model.types import Gaussians


def _import_inria_graphics():
    try:
        from utils.graphics_utils import fov2focal, getWorld2View2  # type: ignore
    except Exception as e:  # ImportError or deeper
        from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

        raise Learn2SplatError(
            "could not import inria graphics utils (utils.graphics_utils). "
            "Run from your inria gaussian-splatting checkout (so it is on "
            "sys.path), or use Learn2Splat.initialize_from_ply / "
            "initialize_from_tensors. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e
    return getWorld2View2, fov2focal


# ---------------------------------------------------------------------------
# Gaussians
# ---------------------------------------------------------------------------

def learn2splat_gaussians_from_ply(
    ply_path: str | Path,
    *,
    sh_degree: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "Gaussians":
    """Load a 3DGS PLY into a learn2splat ``Gaussians`` (batch=1, post-activation)."""
    from learn2splat.model.ply_export import load_gaussians_ply
    from learn2splat.scene_trainer.common.gaussians import build_covariance

    g = load_gaussians_ply(str(ply_path), max_sh_degree=sh_degree)
    g = g.to(device=device, dtype=dtype)
    # Populate covariances so any depth / use_covariances path is safe (the
    # default color path recomputes from scales+rotations anyway).
    try:
        g.covariances = build_covariance(g.scales[0], g.rotations[0]).unsqueeze(0)
    except Exception:
        g.covariances = None
    return g


def learn2splat_gaussians_from_inria_model(
    gm: object,
    *,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "Gaussians":
    """Ingest an inria ``GaussianModel`` via a temp PLY round-trip."""
    sh_degree = int(getattr(gm, "max_sh_degree", 3))
    with tempfile.TemporaryDirectory(prefix="learn2splat_ingest_") as d:
        tmp = Path(d) / "gaussians.ply"
        gm.save_ply(str(tmp))  # inria writer (original-3DGS schema)
        return learn2splat_gaussians_from_ply(
            tmp, sh_degree=sh_degree, device=device, dtype=dtype
        )


def write_back_to_inria_model(gm: object, final: "Gaussians") -> None:
    """Replace an inria ``GaussianModel``'s params with refined Gaussians.

    Full-replacement semantics: the learned ADC may change the point count, so
    we reallocate every parameter (via inria ``load_ply``) and reset the inria
    Adam optimizer + densification accumulators. The caller must call
    ``gaussians.training_setup(...)`` again before resuming inria Adam.
    """
    import torch

    from learn2splat.model.ply_export import save_gaussian_ply

    with tempfile.TemporaryDirectory(prefix="learn2splat_writeback_") as d:
        tmp = Path(d) / "refined.ply"
        # save_gaussian_ply: B==1, xyzw->wxyz, re-inverts activations.
        save_gaussian_ply(final, save_path=tmp)
        gm.load_ply(str(tmp))  # reallocs _xyz/_features_*/_opacity/_scaling/_rotation

    n = gm._xyz.shape[0]
    dev = gm._xyz.device
    gm.optimizer = None
    gm.xyz_gradient_accum = torch.zeros((n, 1), device=dev)
    gm.denom = torch.zeros((n, 1), device=dev)
    gm.max_radii2D = torch.zeros((n,), device=dev)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def batched_views_from_cameras(
    cameras: Sequence[object],
    *,
    scene_scale: float,
    device: "torch.device",
    dtype: "torch.dtype",
    near: float = 0.01,
    far: float = 100.0,
):
    """Build a learn2splat ``BatchedViews`` (B=1) from inria-style cameras.

    ``near``/``far`` default to inria's hardcoded ``znear=0.01``/``zfar=100.0``
    (also the learn2splat colmap-dataset constants). All cameras must share one
    (H, W) — the decoder takes a single image shape.
    """
    import torch

    from learn2splat.dataset.data_types import BatchedViews

    getWorld2View2, fov2focal = _import_inria_graphics()

    if len(cameras) == 0:
        from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

        raise Learn2SplatError("no cameras provided.")

    exts, intrs, imgs = [], [], []
    H0 = W0 = None
    for cam in cameras:
        W = int(cam.image_width)
        H = int(cam.image_height)
        if H0 is None:
            H0, W0 = H, W
        elif (H, W) != (H0, W0):
            from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

            raise Learn2SplatError(
                f"all train cameras must share one (H, W); got {(H, W)} vs "
                f"{(H0, W0)}. Render to a single resolution before optimizing."
            )

        w2c = torch.tensor(
            getWorld2View2(cam.R, cam.T), dtype=torch.float32
        )  # [4,4] world->camera
        c2w = torch.inverse(w2c)  # learn2splat extrinsics convention

        fx = fov2focal(cam.FoVx, W)
        fy = fov2focal(cam.FoVy, H)
        cx = float(getattr(cam, "cx", W / 2.0))
        cy = float(getattr(cam, "cy", H / 2.0))
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = fx / W      # normalized focal
        K[1, 1] = fy / H
        K[0, 2] = cx / W      # normalized principal point
        K[1, 2] = cy / H

        img = cam.original_image
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img = img.float().clamp(0.0, 1.0)  # [3, H, W]

        exts.append(c2w)
        intrs.append(K)
        imgs.append(img)

    V = len(cameras)
    extrinsics = torch.stack(exts).unsqueeze(0).to(device=device, dtype=dtype)
    intrinsics = torch.stack(intrs).unsqueeze(0).to(device=device, dtype=dtype)
    image = torch.stack(imgs).unsqueeze(0).to(device=device, dtype=dtype)
    return BatchedViews.from_dict(
        {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "image": image,
            "near": torch.full((1, V), near, device=device, dtype=dtype),
            "far": torch.full((1, V), far, device=device, dtype=dtype),
            "index": torch.arange(V, device=device).unsqueeze(0),
            "scene_scale": torch.tensor([float(scene_scale)], device=device, dtype=dtype),
        }
    )
