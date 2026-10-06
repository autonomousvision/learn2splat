import os
import sys
from pathlib import Path

from learn2splat.misc.io import CustomPath

_PKG_DIR = CustomPath(__file__).resolve().parent          # .../learn2splat   (the package)
_REPO_ROOT = _PKG_DIR.parent                              # repo root (source) OR site-packages (installed)

# Source checkout iff a project marker sits next to the package dir.
_IS_SOURCE_CHECKOUT = (_REPO_ROOT / "pyproject.toml").exists() or (_REPO_ROOT / ".git").exists()

# Working root for run outputs (checkpoints/results/figures) and datasets.
#   source checkout : the repo root
#   installed       : $LEARN2SPLAT_HOME if set, else the current working directory
if _IS_SOURCE_CHECKOUT:
    PROJECT_DIR = CustomPath(str(_REPO_ROOT))
else:
    PROJECT_DIR = CustomPath(str(os.environ.get("LEARN2SPLAT_HOME", Path.cwd())))

SRC_DIR = CustomPath(str(_PKG_DIR))                       # importable package dir (always correct)
CKPT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
DATA_DIR = PROJECT_DIR / "datasets"

DL3DV_480P_DIR = DATA_DIR / "dl3dv-480p-chunks"
DL3DV_COLMAP_SfM_DIR = DATA_DIR / "dl3dv-colmap-sfm"

# Eval-index / font assets are NOT bundled in the wheel (they are data, like
# datasets). Resolution order: $LEARN2SPLAT_ASSETS, else <repo>/assets in a source
# checkout. See README / DATASETS.md.
ASSETS_DIR = (
    CustomPath(str(os.environ["LEARN2SPLAT_ASSETS"]))
    if os.environ.get("LEARN2SPLAT_ASSETS")
    else (PROJECT_DIR / "assets")
)


def asset_path(rel) -> CustomPath:
    """Resolve an asset file (eval-index JSON, font, ...).

    Absolute or already-existing paths pass through unchanged. A leading
    ``assets/`` is stripped so callers may pass either ``assets/x.json`` or
    ``x.json``. The base dir is ``$LEARN2SPLAT_ASSETS`` (recommended for installed
    use), else ``<repo>/assets`` in a source checkout.
    """
    p = Path(str(rel))
    if p.is_absolute() or p.exists():
        return CustomPath(str(p))
    s = str(rel)
    if s.startswith("assets/"):
        s = s[len("assets/"):]
    return CustomPath(str(ASSETS_DIR)) / s


# DEBUG enables debug-only code paths (shorter runs, extra logging, "DEBUG"
# run names). Force it on or off with LEARN2SPLAT_DEBUG (1/0, true/false); otherwise
# it auto-enables under any debugger that installs a trace function (PyCharm,
# pdb, debugpy).
_debug_env = os.environ.get("LEARN2SPLAT_DEBUG")
if _debug_env is not None:
    DEBUG = _debug_env.strip().lower() in ("1", "true", "yes", "on")
else:
    DEBUG = sys.gettrace() is not None

if DEBUG:
    print("Running in debug mode")
