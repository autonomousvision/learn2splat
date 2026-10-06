"""Integration helpers for using learn2splat's learned optimizer in external
(inria-style) 3D Gaussian Splatting codebases.

Public surface is :class:`learn2splat.Learn2Splat`; the symbols here are the building
blocks (kept importable for advanced/low-level use and testing).
"""

from learn2splat.experimental.api.integration.scene_protocol import (
    Learn2SplatError,
    CameraLike,
    GaussiansLike,
    SceneLike,
    assert_scene_protocol,
)

__all__ = [
    "Learn2SplatError",
    "CameraLike",
    "GaussiansLike",
    "SceneLike",
    "assert_scene_protocol",
]
