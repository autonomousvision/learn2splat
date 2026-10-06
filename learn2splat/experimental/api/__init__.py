"""learn2splat experimental API — use the learned optimizer in external 3DGS codebases.

The public entry points are :class:`Learn2Splat` (the learned optimizer) and
:class:`Learn2SplatInitializer` (the learned feed-forward initializer). They are
exposed lazily via PEP 562 ``__getattr__`` so that ``import
learn2splat.experimental.api`` stays cheap (no torch/hydra import) until a symbol is
actually accessed.
"""

__all__ = ["Learn2Splat", "Learn2SplatInitializer", "Learn2SplatError"]


def __getattr__(name: str):
    if name == "Learn2Splat":
        from learn2splat.experimental.api.api import Learn2Splat

        return Learn2Splat
    if name == "Learn2SplatInitializer":
        from learn2splat.experimental.api.initializer import Learn2SplatInitializer

        return Learn2SplatInitializer
    if name == "Learn2SplatError":
        from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError

        return Learn2SplatError
    raise AttributeError(f"module 'learn2splat.experimental.api' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
