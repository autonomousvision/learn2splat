"""Structural contracts the external scene must satisfy.

These ``runtime_checkable`` Protocols are written so that inria
gaussian-splatting's own ``Scene`` / ``GaussianModel`` / ``Camera`` classes
satisfy them **with no changes** (graphdeco-inria/gaussian-splatting and forks of
it). Non-inria codebases can either expose
the same attribute names or use the low-level ``Learn2Splat.initialize_from_ply`` /
``initialize_from_tensors`` entrypoints.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


class Learn2SplatError(RuntimeError):
    """Raised for all Learn2Splat API misuse / unsupported-checkpoint conditions."""


@runtime_checkable
class GaussiansLike(Protocol):
    """An inria ``GaussianModel`` (raw, pre-activation parameter storage)."""

    active_sh_degree: int
    max_sh_degree: int
    _xyz: Any
    _features_dc: Any
    _features_rest: Any
    _scaling: Any
    _rotation: Any
    _opacity: Any

    def save_ply(self, path: str) -> None: ...
    def load_ply(self, path: str) -> None: ...


@runtime_checkable
class CameraLike(Protocol):
    """An inria ``Camera`` (world->camera ``R,T`` + FoV, COLMAP convention)."""

    R: Any
    T: Any
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    original_image: Any  # [3, H, W] in [0, 1]


@runtime_checkable
class SceneLike(Protocol):
    """An inria ``Scene`` holding a ``GaussianModel`` and posed cameras."""

    cameras_extent: float
    gaussians: GaussiansLike

    def getTrainCameras(self, scale: float = 1.0) -> Sequence[CameraLike]: ...


# Explicit attribute lists drive precise error messages (clearer than a bare
# isinstance failure, which does not say *which* member is missing).
_SCENE_ATTRS = ("cameras_extent", "gaussians")
_SCENE_METHODS = ("getTrainCameras",)
_GAUSSIAN_ATTRS = (
    "active_sh_degree", "max_sh_degree",
    "_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity",
)
_GAUSSIAN_METHODS = ("save_ply", "load_ply")
_CAMERA_ATTRS = (
    "R", "T", "FoVx", "FoVy", "image_width", "image_height", "original_image",
)


def _missing(obj: object, attrs: tuple[str, ...], methods: tuple[str, ...] = ()) -> list[str]:
    miss = [a for a in attrs if not hasattr(obj, a)]
    miss += [f"{m}()" for m in methods if not callable(getattr(obj, m, None))]
    return miss


def assert_scene_protocol(scene: object) -> None:
    """Validate ``scene`` against :class:`SceneLike`; raise a precise error.

    Checked structurally (duck-typed) so inria's classes pass unchanged.
    """
    miss = _missing(scene, _SCENE_ATTRS, _SCENE_METHODS)
    if miss:
        raise Learn2SplatError(
            f"scene is missing required attribute(s)/method(s): {', '.join(miss)}. "
            f"Expected an inria-style Scene (cameras_extent, gaussians, "
            f"getTrainCameras()). Use Learn2Splat.initialize_from_ply/"
            f"initialize_from_tensors for non-inria codebases."
        )
    gm = scene.gaussians
    gmiss = _missing(gm, _GAUSSIAN_ATTRS, _GAUSSIAN_METHODS)
    if gmiss:
        raise Learn2SplatError(
            f"scene.gaussians is missing: {', '.join(gmiss)}. Expected an inria "
            f"GaussianModel (raw _xyz/_features_dc/_features_rest/_scaling/"
            f"_rotation/_opacity + save_ply/load_ply)."
        )
    cams = scene.getTrainCameras()
    if cams is None or len(cams) == 0:
        raise Learn2SplatError("scene.getTrainCameras() returned no cameras.")
    cmiss = _missing(cams[0], _CAMERA_ATTRS)
    if cmiss:
        raise Learn2SplatError(
            f"train camera is missing: {', '.join(cmiss)}. Expected an inria "
            f"Camera (R, T, FoVx, FoVy, image_width, image_height, "
            f"original_image)."
        )
