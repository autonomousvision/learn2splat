"""Resolve Hugging Face Hub checkpoint references to local cached paths.

Any `checkpointing.pretrained_*` config value may be given as a Hugging Face
reference instead of a local path:

    hf://<repo_id>/<path/in/repo.ckpt>            # latest revision on main
    hf://<repo_id>/<path/in/repo.ckpt>@<revision> # pinned branch/tag/commit

Example:

    checkpointing.pretrained_model=hf://autonomousvision/learn2splat/model.ckpt

The file is downloaded once into ``./checkpoints`` (``HF_CACHE_DIR`` below,
relative to the working directory), laid out by its in-repo path, and the
local path is returned, so all downstream ``torch.load`` calls keep working
unchanged.

Gated/private repos (e.g. ``autonomousvision/learn2splat``) require
authentication: run ``huggingface-cli login`` or set the ``HF_TOKEN``
environment variable.
"""

from __future__ import annotations

from .io import cyan

HF_PREFIX = "hf://"

# hf:// checkpoints (and their sibling config.yaml) are downloaded here on
# first access — relative to the working directory — as plain files laid out
# by their in-repo path (e.g. ./checkpoints/dense/checkpoints/model.ckpt),
# instead of the global HF cache's models--*/snapshots/<hash>/ structure.
# huggingface_hub still skips the download when the local copy is current.
HF_CACHE_DIR = "checkpoints"


def is_hf_ref(path: str | None) -> bool:
    return isinstance(path, str) and path.startswith(HF_PREFIX)


def resolve_hf_ref(ref: str) -> str:
    """Download an ``hf://`` reference and return the local cached file path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "huggingface_hub is required to load 'hf://' checkpoints. "
            "Install it with `pip install huggingface_hub`."
        ) from e

    body = ref[len(HF_PREFIX):]
    revision = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)

    parts = body.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid HF checkpoint reference {ref!r}. Expected "
            f"'hf://<org>/<repo>/<path/in/repo>[@<revision>]'."
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])

    print(cyan(f"Resolving HF checkpoint {ref} (repo={repo_id}, "
               f"file={filename}, revision={revision or 'main'})"))
    local_path = hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision,
        local_dir=HF_CACHE_DIR,
    )
    print(cyan(f"Downloaded to {local_path}"))
    return local_path


def maybe_resolve_hf_ref(path: str | None) -> str | None:
    """Resolve `path` if it is an `hf://` reference, otherwise return it as-is."""
    if is_hf_ref(path):
        return resolve_hf_ref(path)
    return path


def hf_sibling_config(ref: str) -> str | None:
    """Download the ``config.yaml`` that sits next to an ``hf://`` checkpoint.

    Released checkpoints are laid out as ``<tag>/checkpoints/<file>.ckpt`` with
    the training config at ``<tag>/config.yaml`` (the same `<ckpt>/../../`
    relation `_find_config_for_checkpoint` expects). ``hf_hub_download`` only
    fetches the requested file, so the sibling config must be fetched
    explicitly; pulling it into the same repo/revision snapshot makes it
    discoverable. Returns the local path, or ``None`` if ``ref`` is not an
    ``hf://`` reference / the sibling does not exist.
    """
    if not is_hf_ref(ref):
        return None
    from pathlib import PurePosixPath

    from huggingface_hub import hf_hub_download

    body = ref[len(HF_PREFIX):]
    revision = None
    if "@" in body:
        body, revision = body.rsplit("@", 1)
    parts = body.split("/")
    if len(parts) < 3:
        return None
    repo_id = "/".join(parts[:2])
    file_in_repo = "/".join(parts[2:])
    cfg_in_repo = str(PurePosixPath(file_in_repo).parent.parent / "config.yaml")
    try:
        local = hf_hub_download(
            repo_id=repo_id, filename=cfg_in_repo, revision=revision,
            local_dir=HF_CACHE_DIR,
        )
        print(cyan(f"Fetched sibling config {cfg_in_repo} -> {local}"))
        return local
    except Exception as e:  # sibling may not exist for non-standard layouts
        print(cyan(f"No sibling config.yaml for {ref} ({type(e).__name__}); "
                   f"will fall back to local config discovery."))
        return None
