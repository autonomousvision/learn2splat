"""End-to-end Learn2Splat demo on a COLMAP scene.

SfM-initialize Gaussians, refine them with the learned optimizer via the
``Learn2Splat`` API, and evaluate on held-out views, using only the
``learn2splat`` package:

    from learn2splat.experimental.api import Learn2Splat

    learn2splat = Learn2Splat(checkpoint="hf://org/repo/model.ckpt", device="cuda")
    learn2splat.initialize_from_tensors(gaussians, batched_views)
    refined = learn2splat.optimize()          # learned optimization

COLMAP loading uses ``learn2splat.dataset.colmap``; the SfM init builds a learn2splat
``Gaussians`` directly via ``points_to_gaussians``; evaluation renders with
the optimizer's own decoder.

The scene is refined two ways and compared on held-out views: the learned
optimizer (Learn2Splat) and a 3DGS Adam baseline (gsplat hyperparameters).
Both run through the same ``optimize()`` path with identical SfM init, view
minibatches and step budget. Each uses its checkpoint's gsplat renderer;
``--rasterize-mode`` / ``--eps2d`` pin one renderer across all runs.

Usage (run from the repo root, with ``learn2splat`` importable):

    python demo.py                    # headless: Learn2Splat vs an Adam baseline
    python demo.py --with-gui server  # interactive viser GUI (frames rendered by the decoder)
    python demo.py --with-gui client  # interactive viser GUI (viser's WebGL splat renderer)
    python demo.py --with-gui gradio  # interactive gradio GUI (streamed renders + Model3D splats)

The demo scene and the checkpoints are fetched from the Hugging Face Hub on
first run (cached under ./datasets and ./checkpoints). A CUDA device is required.
"""

import warnings

# Demo: silence third-party UserWarnings (xFormers/flash-attn not installed,
# Hydra's _self_ notice, pointops' deprecated tensor constructors) for clean output.
warnings.filterwarnings("ignore")

import glob
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from rich.console import Console
from rich.table import Table
from torch import Tensor

console = Console()

from learn2splat.dataset.colmap.utils import Dataset, Parser
from learn2splat.experimental.initializers_utils import knn, points_to_gaussians
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance

# Camera near/far planes — inria's znear/zfar (also the learn2splat colmap-dataset
# constants). Fixed; not a user setting.
NEAR_PLANE = 0.01
FAR_PLANE = 100.0

# Spherical-harmonics DC -> RGB (3DGS convention: rgb = 0.5 + C0 * dc). Colours
# the splats for viser's client-side renderer.
SH_C0 = 0.28209479177387814

# The demo scene is fetched from this Hugging Face repo on first run. The repo
# mirrors the local layout, so e.g. ``datasets/mip360/garden`` in the repo lands at
# ``./datasets/mip360/garden``.
DEMO_DATA_REPO = "autonomousvision/learn2splat"

# Learned-optimizer checkpoints on the Hugging Face Hub. hf:// refs are fetched
# and cached under ./checkpoints on first use (see learn2splat.misc.hf_ckpt).
CHECKPOINTS = {
    "joint": "hf://autonomousvision/learn2splat/joint/checkpoints/epoch_7-step_150000.ckpt",
}

# Dedicated learned feed-forward initializer checkpoint (the "resplat" init), distinct
# from the dense/sparse *optimizer* checkpoints above — it carries only the initializer
# weights. Used by the Learn2SplatInitializer facade for init_type="resplat", independent of
# which optimizer is then chosen to refine. Fetched + cached like the others on first use.
RESPLAT_INIT_CHECKPOINT = (
    "hf://autonomousvision/learn2splat/resplat_init/checkpoints/epoch_20-step_100000.ckpt"
)


def _discover_resplat_checkpoints() -> Dict[str, str]:
    """Scan local checkpoint dirs for available ReSplat initializer checkpoints.

    Returns an ordered dict of display-label → path for files that exist on disk.
    Only GS-capable checkpoints are included (depth-only variants are skipped).
    """
    from learn2splat.misc.checkpointing import extract_step

    ckpts: Dict[str, str] = {}
    official_dir = "checkpoints/resplat_official"
    if os.path.isdir(official_dir):
        for path in sorted(glob.glob(os.path.join(official_dir, "resplat-*.pth"))):
            fname = os.path.basename(path)
            if "-depth-" in fname:
                continue  # depth-only checkpoints lack a Gaussian regressor
            m = re.match(r"resplat-(\w+)-(\w+)-(\d+x\d+)-view(\d+)-[0-9a-f]+\.pth", fname)
            if m:
                size, dataset, res, views = m.groups()
                label = f"{size} · {dataset} {res} view{views}"
            else:
                label = fname[len("resplat-"):-4]
            ckpts[label] = path
    init_ckpt_dir = "checkpoints/resplat_init/checkpoints"
    if os.path.isdir(init_ckpt_dir):
        local_ckpts = glob.glob(os.path.join(init_ckpt_dir, "*.ckpt"))
        if local_ckpts:
            ckpts["local (resplat_init)"] = max(
                local_ckpts, key=lambda p: extract_step(os.path.basename(p))
            )
    return ckpts


# Label → path/hf-ref map for the ReSplat checkpoint dropdown (CLI and gradio).
# Populated at import time; grows automatically as checkpoints appear on disk.
RESPLAT_CHECKPOINTS: Dict[str, str] = _discover_resplat_checkpoints()

# Optimizer-dropdown label -> (instances key | None, Adam base-config | None),
# shared by both GUIs. The key picks the checkpoint pipeline ("joint", or "custom"
# when one is passed), None defaulting to "joint". The 2nd entry is None for that
# checkpoint's own learned optimizer, or a scene_optimizer config name that swaps in
# an Adam baseline (adam / adam_tuned / fastgs LRs) — checkpoint-agnostic, so its key is None.
OPTIMIZER_OPTIONS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "Learn2Splat": ("joint", None),
    "Adam": (None, "adam"),
    "Adam-tuned": (None, "adam_tuned"),
    "Adam (FastGS)": (None, "fastgs"),
}


def optimizer_slug(label: str) -> str:
    """CLI-friendly key for an OPTIMIZER_OPTIONS label, e.g. 'Adam (FastGS)' -> 'adam-fastgs'."""
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def resolve_optimizer(value: str) -> Optional[str]:
    """Map a CLI --optimizer value to an OPTIMIZER_OPTIONS label. Accepts the exact
    label ('Adam (FastGS)') or its slug ('adam-fastgs'). Returns None if no match."""
    if value in OPTIMIZER_OPTIONS:
        return value
    return {optimizer_slug(k): k for k in OPTIMIZER_OPTIONS}.get(optimizer_slug(value))


# Initialization methods, shared by the CLI (the Config.init_type Literal) and the
# gradio Initializer dropdown so the two can't drift. Maps the gradio display label ->
# the init_type value. "sfm"/"random" are built by sfm_initialization (parser-based);
# "resplat" runs the learned feed-forward initializer (RESPLAT_INIT_CHECKPOINT) via the
# Learn2SplatInitializer API facade. "depth" unprojects per-view RGB-D into a point cloud
# (InitializerDepth) — only valid on depth-carrying datasets.
INIT_TYPES: Dict[str, str] = {
    "SfM": "sfm", "Random": "random", "ReSplat": "resplat", "Depth (RGB-D)": "depth",
}

# Curated scenes for the gradio Scene dropdown — display label -> data_path. These are
# the HF-mirrored demo scenes ensure_data can fetch on demand. The CLI selects a scene
# directly via --data-path (any local COLMAP dir). Grows as more demo scenes are added.
SCENES: Dict[str, str] = {"Garden (mip360)": "datasets/mip360/garden"}
# Dropdown sentinel that reveals the custom-path textbox (load an external reconstruction).
CUSTOM_SCENE_LABEL = "Custom path…"


def ensure_data(data_path: str) -> None:
    """Make a scene available locally. For the HF-mirrored demo scenes (SCENES values)
    download from the Hub on first use; for any other (custom external) path just require
    it exists — never fire a bogus Hub download for a path that isn't mirrored there."""
    if os.path.isdir(data_path) and os.listdir(data_path):
        return
    if data_path not in SCENES.values():
        raise SystemExit(
            f"scene path {data_path!r} not found — pass an existing COLMAP directory "
            f"(images/ + sparse/0/) via --data-path."
        )
    from huggingface_hub import snapshot_download

    console.print(
        f"[yellow]{data_path}[/] not found — downloading from "
        f"[cyan]hf://{DEMO_DATA_REPO}[/] …"
    )
    snapshot_download(
        repo_id=DEMO_DATA_REPO,
        allow_patterns=[f"{data_path.rstrip('/')}/**"],
        local_dir=".",
    )
    console.print(f"[green]✓[/] scene ready at [yellow]{data_path}[/]")


@dataclass
class Config:
    # Dataset format / reader. Only "colmap" is wired up today; the Literal will grow
    # ("dl3dv", "megasynth", …) and the single dispatch point is build_parser(). tyro
    # rejects an unknown --data-format at parse time.
    data_format: Literal["colmap"] = "colmap"
    # Path to the scene (for colmap: a dir with images/ + sparse/0/).
    data_path: str = "datasets/mip360/garden"
    # Downsample factor for the dataset.
    data_factor: int = 4
    # Global multiplier on scene-size-related parameters.
    global_scale: float = 1.0
    # Normalize the world space.
    normalize_world_space: bool = True
    # Every N images is a test image, held out for evaluation.
    test_every: int = 8
    # Directory to save renders / stats / the refined PLY.
    result_dir: str = "results/demo"
    # Random seed.
    seed: int = 42

    # --- Interactive GUI ---
    # Launch an interactive GUI instead of the headless comparison. viser:
    # "server" renders frames with the learn2splat decoder, "client" uses viser's
    # built-in WebGL Gaussian-splat renderer. "gradio" runs a browser GUI
    # (decoder renders streamed live + an interactive Model3D splat viewer for
    # the result). Unset = headless run.
    with_gui: Optional[Literal["client", "server", "gradio"]] = None
    # Port for the GUI web server (--with-gui only).
    gui_port: int = 8080

    # --- Learn2Splat learned optimizer ---
    # Compute device (Learn2Splat requires CUDA).
    device: str = "cuda"
    # Number of learned refinement steps.
    max_steps: int = 100
    # Views the optimizer sees per refinement step (the view minibatch).
    opt_batch_size: int = 8
    # View-minibatch sampling strategy: "random", "sequential", or "fps"
    # (farthest-point sampling over camera positions).
    opt_batch_strategy: Literal["random", "sequential", "fps"] = "fps"
    # Headless only: which optimizer to run. Unset runs the full comparison sweep
    # (Learn2Splat dense + sparse + an Adam baseline). Set to one entry of
    # OPTIMIZER_OPTIONS — by label ("Adam (FastGS)") or slug ("adam-fastgs") — to run
    # only that method (e.g. to reproduce a single GUI configuration headless).
    optimizer: Optional[str] = None

    # --- Renderer ---
    # Rasterizer backend for rendering and (with the learned / Adam optimizers)
    # the optimization itself: "gsplat" (default, the trained renderer), "inria",
    # or "fastgs". inria / fastgs need their CUDA backend installed. Applies to the
    # headless comparison and seeds the GUI's Renderer dropdown.
    renderer: Literal["gsplat", "inria", "fastgs"] = "gsplat"
    # rasterize_mode / eps2d apply to the gsplat renderer ONLY — inria / fastgs
    # ignore them. When set (with --renderer gsplat), applied to every run (dense,
    # sparse, Adam), overriding each checkpoint's decoder config. Left unset, each
    # run uses its own checkpoint's value.
    rasterize_mode: Optional[Literal["classic", "antialiased"]] = None
    eps2d: Optional[float] = None

    # --- ADC (densification) ---
    # Adaptive Density Control strategy applied during optimization: "off" (default
    # — fixed Gaussian set, the like-for-like comparison), "vanilla" (3DGS clone/
    # split/prune), "edgs", "mcmc". The densification schedule is auto-scaled to
    # max_steps, so a short demo run densifies (the usual 500-step warm-up never
    # would). Auto-extends to "fastgs" once it's registered in the ADC dispatcher.
    adc: str = "off"

    # --- Initialization ---
    # Initialization strategy. tyro validates the value (rejects typos up front);
    # the choices match the gradio Initializer dropdown (see INIT_TYPES). "resplat"
    # runs the learned feed-forward initializer (RESPLAT_INIT_CHECKPOINT); "depth"
    # unprojects per-view RGB-D into a point cloud (needs a depth-carrying dataset —
    # COLMAP scenes have none, so it errors there).
    init_type: Literal["sfm", "random", "resplat", "depth"] = "sfm"
    # Initial number of GSs. Ignored when init_type="sfm".
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the scene extent (random init).
    init_extent: float = 3.0
    # Initial opacity / scale of each GS (sfm / random / depth — depth uses these as its
    # init_opacity / scaling_factor).
    init_opa: float = 0.1
    init_scale: float = 1.0
    # resplat only: how many input views the learned initializer is conditioned on
    # (trained on ~8; the Gaussian count and MVS cost volume scale ~linearly with it).
    init_num_context_views: int = 8
    # resplat only: how those context views are picked from the training set —
    # "fps" (farthest-point spread, best MVS coverage), "sequential", or "random".
    init_context_strategy: Literal["fps", "sequential", "random"] = "fps"
    # resplat only: longer image side fed to the MVS backbone (~training res). Full
    # COLMAP res is much larger (OOMs / out-of-distribution); the Gaussian count scales
    # with this. Higher = denser/sharper init but more VRAM.
    init_resplat_max_side: int = 512
    # resplat only: which initializer checkpoint to load. Accepts a local path
    # (e.g. checkpoints/resplat_official/resplat-base-dl3dv-256x448-view8-*.pth) or
    # an hf:// ref. Defaults to the first locally-discovered checkpoint if any exist,
    # falling back to the HF hub ref. The gradio GUI exposes all locally-discovered
    # checkpoints in a dropdown (see RESPLAT_CHECKPOINTS).
    resplat_checkpoint: str = next(iter(RESPLAT_CHECKPOINTS.values()), RESPLAT_INIT_CHECKPOINT)
    # Local path or hf:// ref. Adds "Learn2Splat (custom)" to the optimizer dropdown;
    # headless: pair with --optimizer learn2splat-custom.
    optimizer_checkpoint: Optional[str] = None
    # depth only: resize each context view so its longer side is this many pixels before
    # unprojecting (one point per pixel per view → controls point-cloud density). null =
    # native resolution.
    init_depth_longer_side: Optional[int] = 512


def scene_extent(parser: Parser, global_scale: float) -> float:
    """Scene-size scalar: parser extent x 1.1 x global_scale."""
    return parser.scene_scale * 1.1 * global_scale


def build_parser(cfg: "Config", normalize: bool, *, verbose: bool = False) -> Parser:
    """Build the scene reader for cfg.data_format."""
    if cfg.data_format == "colmap":
        return Parser(
            data_dir=cfg.data_path, factor=cfg.data_factor, normalize=normalize,
            verbose=verbose,
        )
    raise NotImplementedError(f"data_format {cfg.data_format!r} is not yet supported")


def sfm_initialization(
    parser: Parser, cfg: Config, sh_degree: int, device: torch.device, dtype: torch.dtype
) -> Gaussians:
    """SfM (or random) Gaussian init -> a learn2splat ``Gaussians`` (batch=1).

    Builds the parameter tensors with the same heuristics as 3DGS / the learn2splat
    COLMAP initializer, then assembles them through ``points_to_gaussians``.
    """
    if cfg.init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif cfg.init_type == "random":
        extent = scene_extent(parser, cfg.global_scale)
        points = cfg.init_extent * extent * (
            torch.rand((cfg.init_num_pts, 3)) * 2 - 1
        )
        rgbs = torch.rand((cfg.init_num_pts, 3))
    else:
        raise ValueError(f"unknown init_type: {cfg.init_type!r} (sfm | random)")

    # GS size = average distance to the 3 nearest neighbours ([:, 1:] drops self).
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    scales = (torch.sqrt(dist2_avg) * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    opacities = torch.full((points.shape[0],), cfg.init_opa)

    # points_to_gaussians returns pre-activation params (log scales, logit
    # opacity, sh0/shN, random quats).
    g = points_to_gaussians(
        {"xyz": points, "rgb": rgbs, "scales": scales, "opacities": opacities},
        sh_degree=sh_degree,
        device=device,
    )
    sh0, shN = g["sh0"], g["shN"]
    harmonics = torch.cat([sh0, shN], dim=1) if shN is not None else sh0  # [N, K, 3]
    harmonics = harmonics.permute(0, 2, 1)  # -> [N, 3, K]

    scales_act = torch.exp(g["scales_raw"])
    opacities_act = torch.sigmoid(g["opacities_raw"])
    rotations = F.normalize(g["rotations_unnorm"], dim=-1)  # identity — points_to_gaussians sets it
    covariances = build_covariance(scale=scales_act, rotation_xyzw=rotations)

    def _b(t: Tensor) -> Tensor:  # add the batch dimension and cast
        return t.unsqueeze(0).to(dtype)

    return Gaussians(
        means=_b(g["xyz"]),
        covariances=_b(covariances),
        harmonics=_b(harmonics),
        opacities=_b(opacities_act),
        scales=_b(scales_act),
        rotations=_b(rotations),
        rotations_unnorm=_b(g["rotations_unnorm"]),
    )


def get_resplat_initializer(cfg: Config, _cache: dict = {}):
    """Build the learned feed-forward initializer facade once and cache it per checkpoint.

    Loads ``cfg.resplat_checkpoint`` (a ~40M–2B-param MVS network) via the
    ``Learn2SplatInitializer`` API facade. Cached by checkpoint path so switching checkpoints
    in the gradio GUI builds a new facade without discarding previously-loaded ones.
    Only used when ``cfg.init_type == "resplat"``; independent of the chosen optimizer.
    """
    ckpt = cfg.resplat_checkpoint
    if ckpt not in _cache:
        from learn2splat.experimental.api import Learn2SplatInitializer

        _cache[ckpt] = Learn2SplatInitializer(checkpoint=ckpt, device=cfg.device)
    return _cache[ckpt]


def build_depth_initializer(cfg: Config, sh_degree: int):
    """Build the non-learned RGB-D ``InitializerDepth`` (no weights) at the demo's SH
    degree. Reuses cfg's shared opacity / scale settings and the depth-resolution setting;
    DDP-only subsampling is disabled (single-scene init)."""
    from learn2splat.scene_trainer.initializer import get_scene_initializer
    from learn2splat.scene_trainer.initializer.initializer_depth import InitializerDepthCfg

    init_cfg = InitializerDepthCfg(
        name="depth",
        per_pixel=False, per_view=False,
        train_min_gaussians_subsample=None, train_max_gaussians_subsample=None,
        eval_min_gaussians_subsample=None, eval_max_gaussians_subsample=None,
        train_fixed_gaussians_num=None, eval_fixed_gaussians_num=None,
        scaling_factor=cfg.init_scale,
        init_opacity=cfg.init_opa,
        sh_degree=sh_degree,
        init_longer_side=cfg.init_depth_longer_side,
        filter_invalid_depth=True,
    )
    return get_scene_initializer(init_cfg).eval()


def build_initialization(
    parser: Parser,
    cfg: Config,
    train_bv: dict,
    sh_degree: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Gaussians:
    """Dispatch ``cfg.init_type`` to its initial Gaussians (batch=1).

    ``sfm``/``random`` -> :func:`sfm_initialization` (parser-based, cheap). ``resplat``
    -> the learned feed-forward initializer (built + cached once via
    :func:`get_resplat_initializer`) run on the training views, at the configured image
    resolution; its depth sweep uses the facade's fixed near/far defaults — NOT the
    demo's render planes. ``depth`` -> :class:`InitializerDepth`, unprojecting per-view
    RGB-D into a point cloud — requires a depth-carrying scene (raises otherwise).
    """
    if cfg.init_type in ("sfm", "random"):
        return sfm_initialization(parser, cfg, sh_degree, device, dtype)
    if cfg.init_type == "resplat":
        facade = get_resplat_initializer(cfg)
        if facade.sh_degree != sh_degree:
            raise ValueError(
                f"resplat initializer SH degree ({facade.sh_degree}) does not match "
                f"the optimizer's ({sh_degree}); pick matching checkpoints."
            )
        return facade.initialize(
            train_bv,
            num_context_views=cfg.init_num_context_views,
            strategy=cfg.init_context_strategy,
            max_image_side=cfg.init_resplat_max_side,
        )
    if cfg.init_type == "depth":
        if "depth" not in train_bv:
            raise ValueError(
                f"the 'depth' initializer needs per-view depth, but the current scene "
                f"({cfg.data_path}, format={cfg.data_format}) provides none — COLMAP "
                f"scenes carry no depth. Use a depth-carrying dataset (e.g. MegaSynth)."
            )
        return build_depth_initializer(cfg, sh_degree).forward(train_bv, device=device).gaussians
    raise ValueError(f"unknown init_type: {cfg.init_type!r} (sfm | random | resplat | depth)")


def collect_cameras(
    dataset: Dataset, indices: List[int]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Stack the selected views into ``(camtoworlds, Ks, images)``.

    ``images`` is returned in [0, 1]. All views must share one (H, W) — the
    learn2splat renderer takes a single image shape.
    """
    c2ws, ks, imgs = [], [], []
    hw = None
    for i in indices:
        data = dataset[i]
        img = data["image"] / 255.0  # [H, W, 3], float
        if hw is None:
            hw = img.shape[:2]
        elif img.shape[:2] != hw:
            raise ValueError(
                f"all views must share one (H, W); got {tuple(img.shape[:2])} "
                f"vs {tuple(hw)}. Render the dataset at a single resolution."
            )
        c2ws.append(data["camtoworld"])
        ks.append(data["K"])
        imgs.append(img)
    return torch.stack(c2ws), torch.stack(ks), torch.stack(imgs)


def build_batched_views(
    camtoworlds: Tensor,
    Ks: Tensor,
    images: Tensor,
    scene_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """COLMAP cameras -> a learn2splat ``BatchedViews`` dict (batch=1).

    COLMAP ``camtoworld`` is already learn2splat's extrinsics convention (OpenCV
    camera->world). ``K`` is pixel-space; learn2splat wants it normalized by image
    width/height.
    """
    v, h, w = images.shape[0], images.shape[1], images.shape[2]

    Ks_norm = Ks.clone()
    Ks_norm[:, 0, :] /= w  # normalized focal / principal point
    Ks_norm[:, 1, :] /= h

    image = images.permute(0, 3, 1, 2)  # [V, 3, H, W]

    def _b(t: Tensor) -> Tensor:  # add the batch dimension and move to device
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    return {
        "extrinsics": _b(camtoworlds),
        "intrinsics": _b(Ks_norm),
        "image": _b(image),
        "near": torch.full((1, v), NEAR_PLANE, device=device, dtype=dtype),
        "far": torch.full((1, v), FAR_PLANE, device=device, dtype=dtype),
        "index": torch.arange(v, device=device).unsqueeze(0),
        "scene_scale": torch.tensor([scene_scale], device=device, dtype=dtype),
    }


@torch.no_grad()
def render_and_score(
    learn2splat,
    refined: Gaussians,
    val_bv: dict,
    val_images: Tensor,
    out_dir: str,
    device: torch.device,
) -> dict:
    """Render one optimizer's result on the held-out views; report mean PSNR.

    Saves a ``gt | pred`` strip per view under ``out_dir/renders``.
    """
    render_dir = os.path.join(out_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    h, w = val_images.shape[1], val_images.shape[2]

    out = learn2splat.decoder.forward(
        refined, val_bv["extrinsics"], val_bv["intrinsics"],
        val_bv["near"], val_bv["far"], image_shape=(h, w),
    )
    colors = out.color[0].clamp(0.0, 1.0)  # [V, 3, H, W]

    psnrs = []
    for i in range(colors.shape[0]):
        gt = val_images[i].to(device)  # [H, W, 3]
        pred = colors[i].permute(1, 2, 0)
        psnrs.append(-10.0 * torch.log10(torch.mean((pred - gt) ** 2)).item())

        canvas = torch.cat([gt, pred], dim=1).cpu().numpy()  # gt | pred
        imageio.imwrite(
            os.path.join(render_dir, f"val_{i:04d}.png"),
            (canvas * 255).astype(np.uint8),
        )

    return {"psnr": float(np.mean(psnrs)), "num_views": int(colors.shape[0])}


@torch.no_grad()
def render_view(
    learn2splat, gaussians: Gaussians, camera, height: int,
    device: torch.device, dtype: torch.dtype,
) -> np.ndarray:
    """Render ``gaussians`` from a viser camera into an ``[H, W, 3]`` uint8 image.

    viser cameras follow OpenCV conventions, so ``(wxyz, position)`` is directly
    the camera-to-world transform the learn2splat decoder expects — no axis flip.
    """
    import viser.transforms as vtf

    from learn2splat.misc.image_io import prep_image

    h = int(height)
    w = max(1, round(h * camera.aspect))  # camera.aspect = width / height

    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, :3] = torch.tensor(
        vtf.SO3(camera.wxyz).as_matrix(), device=device, dtype=dtype
    )
    c2w[:3, 3] = torch.tensor(camera.position, device=device, dtype=dtype)

    # Normalized intrinsics from the vertical fov; the decoder un-normalizes by
    # the image width/height.
    fy = (h / 2.0) / float(np.tan(camera.fov / 2.0))
    K = torch.eye(3, device=device, dtype=dtype)
    K[0, 0] = fy / w
    K[1, 1] = fy / h
    K[0, 2] = 0.5
    K[1, 2] = 0.5

    near = torch.full((1, 1), NEAR_PLANE, device=device, dtype=dtype)
    far = torch.full((1, 1), FAR_PLANE, device=device, dtype=dtype)
    out = learn2splat.decoder.forward(
        gaussians, c2w[None, None], K[None, None], near, far, image_shape=(h, w),
    )
    return prep_image(out.color[0, 0])  # [H, W, 3] uint8


def gaussians_to_splat_data(gaussians: Gaussians) -> dict:
    """A learn2splat ``Gaussians`` (batch=1) -> numpy arrays for viser's splat viewer.

    Covariances are recomputed from scale/rotation (the optimizer updates those
    but may leave the optional ``Gaussians.covariances`` field stale); colours
    come from the SH DC term (degree 0 — viser's renderer is not view-dependent).
    """
    scales = gaussians.scales[0]
    opacities = gaussians.opacities[0]
    if not gaussians.stores_activated:
        scales = torch.exp(scales)
        opacities = torch.sigmoid(opacities)
    rotations = F.normalize(gaussians.rotations_unnorm[0], dim=-1)
    covariances = build_covariance(scale=scales, rotation_xyzw=rotations)
    rgbs = (0.5 + SH_C0 * gaussians.harmonics[0, :, :, 0]).clamp(0.0, 1.0)

    def _np(t: Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32)

    return {
        "centers": _np(gaussians.means[0]),          # (N, 3)
        "covariances": _np(covariances),             # (N, 3, 3)
        "rgbs": _np(rgbs),                           # (N, 3)
        "opacities": _np(opacities.reshape(-1, 1)),  # (N, 1)
    }


def run_gui(
    instances: dict,
    gaussians: Gaussians,
    train_bv: dict,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Interactive viser GUI: watch the optimization, pick an optimizer, reset.

    The initialization is shown first; the user picks an optimizer — the
    Learn2Splat learned optimizer (dense or sparse checkpoint) or a 3DGS Adam
    baseline — and clicks Start; every optimizer step is rendered and displayed;
    Reset restores the initialization. ``cfg.with_gui`` chooses the renderer —
    "server" (learn2splat decoder, frames streamed as images) or "client" (viser's
    WebGL splats).

    ``instances`` maps each checkpoint name to its initialized ``Learn2Splat``.
    """
    import threading

    import viser
    import viser.transforms as vtf

    from learn2splat.experimental.api.integration.config_bridge import build_adam_baseline

    mode = cfg.with_gui  # "server" | "client"
    server = viser.ViserServer(port=cfg.gui_port)

    optimizer_dd = server.gui.add_dropdown("Optimizer", tuple(OPTIMIZER_OPTIONS))

    # Optimization controls — applied to the picked Learn2Splat at Start; frozen
    # while optimizing, unfrozen by Reset. opt_batch_size is capped at the
    # number of training views (the per-step view minibatch can't exceed them).
    n_train_views = int(train_bv["image"].shape[1])
    max_steps_input = server.gui.add_number(
        "Max steps", min=1, max=1000, step=1, initial_value=cfg.max_steps
    )
    batch_size_input = server.gui.add_number(
        "Opt batch size", min=1, max=n_train_views, step=1,
        initial_value=min(cfg.opt_batch_size, n_train_views),
    )
    strategy_dd = server.gui.add_dropdown(
        "Opt batch strategy", ("random", "sequential", "fps"),
        initial_value=cfg.opt_batch_strategy,
    )
    opt_controls = (max_steps_input, batch_size_input, strategy_dd)

    start_btn = server.gui.add_button("Start optimization")
    reset_btn = server.gui.add_button("Reset to initialization")
    status = server.gui.add_markdown("**initialized** — pick an optimizer, then Start")
    res_slider = (
        server.gui.add_slider(
            "Render height", min=240, max=1080, step=60, initial_value=540
        )
        if mode == "server"
        else None
    )

    init_gaussians = gaussians.clone()  # pristine copy, for Reset
    current = init_gaussians            # Gaussians currently displayed
    active = instances["joint"]         # Learn2Splat used to render + to optimize next
    gen = None                          # optimize_iter generator while running
    last_cam_ts: dict = {}              # client id -> last-rendered camera stamp
    lock = threading.Lock()
    state = {
        "mode": "init",                 # "init" | "optimizing" | "done"
        "step": 0,
        "start": False,
        "reset": False,
        "rerender": False,              # a GUI control changed -> re-render once
        "selected": next(iter(OPTIMIZER_OPTIONS)),
    }

    @start_btn.on_click
    def _(_) -> None:
        with lock:
            if state["mode"] in ("init", "done"):
                state["selected"] = optimizer_dd.value
                state["start"] = True

    @reset_btn.on_click
    def _(_) -> None:
        with lock:
            state["reset"] = True

    # The render-height slider only affects server-rendered frames; re-render
    # on change so the new resolution takes effect without a camera move.
    if res_slider is not None:

        @res_slider.on_update
        def _(_) -> None:
            with lock:
                state["rerender"] = True

    # Frame newly-connected clients on the first training camera (viser and
    # learn2splat share the OpenCV camera-to-world convention).
    cam_extr = train_bv["extrinsics"][0, 0].detach().cpu().numpy()

    @server.on_client_connect
    def _(client) -> None:
        try:
            client.camera.position = cam_extr[:3, 3]
            client.camera.wxyz = vtf.SO3.from_matrix(cam_extr[:3, :3]).wxyz
        except Exception:
            pass

    if mode == "client":  # show the initialization immediately
        # Black backdrop for the WebGL splat renderer (viser's canvas is not
        # black by default); on server.scene so late-joining clients get it.
        server.scene.set_background_image(np.zeros((8, 8, 3), dtype=np.uint8))
        server.scene.add_gaussian_splats(
            "/learn2splat/splats", **gaussians_to_splat_data(current)
        )

    console.print(
        f"[green]✓[/] viser GUI ([cyan]{mode}[/]) on port [cyan]{cfg.gui_port}[/]"
        f" — forward the port over SSH and open the printed URL"
    )

    try:
        while True:
            changed = False

            with lock:
                do_reset, do_start = state["reset"], state["start"]
                do_rerender = state["rerender"]
                state["reset"] = state["start"] = state["rerender"] = False
                selected = state["selected"]

            if do_rerender:
                changed = True  # server mode re-renders every connected client

            if do_reset:
                if gen is not None:
                    gen.close()  # runs optimize_iter's finally -> on_scene_end()
                    gen = None
                current = init_gaussians
                with lock:
                    state["mode"], state["step"] = "init", 0
                optimizer_dd.disabled = start_btn.disabled = False
                for c in opt_controls:
                    c.disabled = False
                changed = True

            if do_start and gen is None:
                name, base = OPTIMIZER_OPTIONS[selected]
                active = instances[name or "joint"]  # None -> the joint checkpoint by default
                # Apply the GUI optimization controls before the run starts.
                active.num_refine = int(max_steps_input.value)
                active.opt_batch_size = int(batch_size_input.value)
                active.opt_batch_strategy = strategy_dd.value
                opt = (
                    build_adam_baseline(active.num_refine, base=base).to(device)
                    if base
                    else None
                )
                gen = active.optimize_iter(optimizer=opt)
                with lock:
                    state["mode"], state["step"] = "optimizing", 0
                optimizer_dd.disabled = start_btn.disabled = True
                for c in opt_controls:
                    c.disabled = True

            if gen is not None:
                try:
                    step, current = next(gen)
                    changed = True
                    with lock:
                        state["step"] = step + 1
                except StopIteration:
                    gen = None
                    with lock:
                        state["mode"] = "done"
                    optimizer_dd.disabled = start_btn.disabled = False

            if mode == "server":
                for cid, client in server.get_clients().items():
                    try:
                        cam_ts = client.camera.update_timestamp
                        if last_cam_ts.get(cid) != cam_ts or changed:
                            last_cam_ts[cid] = cam_ts
                            image = render_view(
                                active, current, client.camera,
                                res_slider.value, device, dtype,
                            )
                            client.scene.set_background_image(image, format="jpeg")
                    except Exception:
                        continue  # no camera message from this client yet
            elif changed:  # client mode — re-push splats when the Gaussians change
                server.scene.add_gaussian_splats(
                    "/learn2splat/splats", **gaussians_to_splat_data(current)
                )

            with lock:
                status.content = (
                    f"**{state['mode']}** — step "
                    f"{state['step']}/{active.num_refine} — "
                    f"{current.means.shape[1]} Gaussians"
                )

            if gen is None:
                time.sleep(1 / 30)  # idle: poll cameras at ~30 Hz
    except KeyboardInterrupt:
        if gen is not None:
            gen.close()
        console.print("\n[yellow]GUI stopped.[/]")


def adc_strategies() -> tuple[list[str], dict[str, str]]:
    """(labels, label -> raw refiner name) for the ADC dropdown / CLI.

    Derived from the refiner registry (``BaseStrategyCfg.name``) so a newly
    registered strategy (e.g. ``fastgs``) appears automatically. ``off`` is first;
    friendly labels rename ``none``->``off`` and ``default``->``vanilla``.
    """
    from typing import get_args

    from learn2splat.scene_trainer.adc.base import BaseStrategyCfg

    friendly = {"none": "off", "default": "vanilla"}
    raw = list(get_args(BaseStrategyCfg.__annotations__["name"]))
    labels = ["off"] + [friendly.get(n, n) for n in raw if n != "none"]
    to_name = {"off": "none", **{friendly.get(n, n): n for n in raw if n != "none"}}
    return labels, to_name


def build_renderer_decoder(renderer: str, device) -> object:
    """Build a fresh decoder for an inria / fastgs ``renderer``, independent of any
    checkpoint, to override a Learn2Splat instance's decoder. gsplat is handled by the
    caller (it keeps the checkpoint's own decoder, which carries its rasterize_mode
    / eps2d). Assumes ``renderer`` is installed — main() validates that up front."""
    from types import SimpleNamespace

    from learn2splat.model.decoder import DECODER_CFGS, get_decoder

    Cfg = DECODER_CFGS[renderer]
    return get_decoder(
        Cfg(name=renderer, scale_invariant=False),
        SimpleNamespace(background_color=[0.0, 0.0, 0.0]),
    ).to(device)


def build_scene_bundle(
    cfg: "Config", normalize: bool, instances: dict, device, dtype
) -> dict:
    """(Re)load the COLMAP scene with world-space normalization on/off, build the SfM
    init, and re-initialize each Learn2Splat instance. Returns the GUI scene bundle. Mirrors
    the scene-build in :func:`main`; the gradio normalize toggle calls it to swap scenes.

    ``normalize=False`` keeps raw COLMAP scale (scene extent ~ camera radius, as FastGS
    uses); ``True`` rescales the world to ~unit. View count / resolution are unchanged."""
    parser = build_parser(cfg, normalize)
    dataset = Dataset(parser)
    val_idx = [i for i in range(len(dataset)) if i % cfg.test_every == 0]
    train_idx = [i for i in range(len(dataset)) if i % cfg.test_every != 0]
    scene_scale = scene_extent(parser, cfg.global_scale)
    train_bv = build_batched_views(
        *collect_cameras(dataset, train_idx), scene_scale, device, dtype
    )
    val_c2w, val_Ks, val_images = collect_cameras(dataset, val_idx)
    val_bv = build_batched_views(val_c2w, val_Ks, val_images, scene_scale, device, dtype)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gaussians = build_initialization(
        parser, cfg, train_bv, instances["joint"].sh_degree, device, dtype
    )
    for inst in instances.values():
        inst.initialize_from_tensors(gaussians, train_bv)
    return {
        "gaussians": gaussians, "train_bv": train_bv, "val_bv": val_bv,
        "val_images": val_images, "scene_scale": scene_scale,
    }


def run_gradio_gui(
    instances: dict,
    gaussians: Gaussians,
    train_bv: dict,
    val_bv: dict,
    val_images: Tensor,
    cfg: Config,
    device: torch.device,
) -> None:
    """Interactive gradio GUI — a browser port of :func:`run_gui` (viser).

    gradio can't stream the camera back to Python, so there is no free-camera
    server rendering. Instead the optimization is *watched* as a streamed
    decoder render from a chosen training view (``gr.Image``, refreshed every
    step), and the finished scene is handed to an interactive ``gr.Model3D``
    splat viewer (orbit / zoom in the browser). The controls mirror the viser
    GUI: pick the optimizer (Learn2Splat dense/sparse or a 3DGS Adam baseline),
    set the step budget / view-minibatch size / sampling strategy, Start, Reset.
    When a run finishes, Evaluate scores the result on the held-out ``val_bv``
    views (mean novel-view PSNR / SSIM / LPIPS + a ``gt | pred`` strip).

    ``instances`` maps each checkpoint name to its initialized ``Learn2Splat``.
    """
    import gc

    import gradio as gr

    from learn2splat.experimental.api.integration.config_bridge import build_adam_baseline
    from learn2splat.misc.image_io import prep_image
    from learn2splat.model.ply_export import save_gaussian_ply

    n_train_views = int(train_bv["image"].shape[1])
    h_full, w_full = train_bv["image"].shape[3], train_bv["image"].shape[4]
    init_gaussians = gaussians.clone()  # pristine copy; each Start re-inits from it

    # A counter for unique PLY filenames (so the Model3D reloads each result
    # rather than serving a stale cache). Single-GPU, single-session demo.
    # "result"/"decoder" stash the last finished run so Evaluate can score it.
    holder: Dict[str, object] = {"ply": 0, "result": None, "decoder": None}
    runs: List[dict] = []  # one row per finished run, for the past-runs table

    n_val_views = int(val_bv["image"].shape[1])
    ply_dir = os.path.join(cfg.result_dir, "gradio")
    eval_dir = os.path.join(cfg.result_dir, "gradio", "eval")
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Decoder renderers the user can pick. gsplat is the checkpoint-trained
    # default; inria / fastgs are optional backends, offered only when their CUDA
    # extension is installed (DECODER_CFGS holds exactly the importable ones). The
    # chosen renderer drives both the in-loop optimization and the displayed renders.
    from learn2splat.model.decoder import DECODER_CFGS
    RENDERERS = list(DECODER_CFGS)  # gsplat first, then any installed inria / fastgs
    orig_decoder = {k: v.decoder for k, v in instances.items()}  # checkpoint-matched gsplat
    alt_decoder: Dict[str, object] = {}  # name -> built inria/fastgs decoder (lazy)

    # ADC (densification) strategies for the dropdown; "off" first, auto-extends to
    # "fastgs" once registered. adc_to_name maps the label back to a raw refiner name.
    ADC_STRATEGIES, adc_to_name = adc_strategies()

    # Initializer strategies (INIT_TYPES is module-level — the same choices the CLI
    # --init-type exposes). SfM / Random map to sfm_initialization's init_type; ReSplat
    # runs the learned feed-forward initializer (get_resplat_initializer). The dropdown
    # toggles each strategy's headline settings (see on_init_type_select); they
    # are applied — and the init (re)run — only when the Initialize button is pressed.
    INIT_CHOICES = list(INIT_TYPES)

    def init_label_for(init_type: str) -> str:
        """The dropdown label for an init_type (SfM if unknown)."""
        return next((k for k, v in INIT_TYPES.items() if v == init_type), "SfM")

    def scene_label_for(data_path: str) -> str:
        """The Scene-dropdown label for a data_path: the curated name if it's one of
        SCENES, else the 'Custom path…' sentinel (an external reconstruction)."""
        return next((k for k, v in SCENES.items() if v == data_path), CUSTOM_SCENE_LABEL)

    def decoder_for(renderer: str):
        """Decoder for a renderer name. gsplat reuses the checkpoint's decoder;
        inria / fastgs are built once on first use."""
        if renderer == "gsplat":
            return orig_decoder["joint"]
        if renderer not in alt_decoder:
            alt_decoder[renderer] = build_renderer_decoder(renderer, device)
        return alt_decoder[renderer]

    @torch.no_grad()
    def render(g: Gaussians, view_idx: float, height: float, renderer: str) -> np.ndarray:
        """Render Gaussians ``g`` from training view ``view_idx`` with ``renderer``.

        Normalized intrinsics make the render resolution-independent; the width
        is derived from ``height`` at the training views' aspect ratio.
        """
        h = int(height)
        w = max(1, round(h * w_full / h_full))
        sl = slice(int(view_idx), int(view_idx) + 1)
        out = decoder_for(renderer).forward(
            g,
            train_bv["extrinsics"][:, sl],
            train_bv["intrinsics"][:, sl],
            train_bv["near"][:, sl],
            train_bv["far"][:, sl],
            image_shape=(h, w),
        )
        return prep_image(out.color[0, 0])  # [H, W, 3] uint8

    def overlay_iter(img: np.ndarray, it: int) -> np.ndarray:
        """Draw 'iter {it}' top-left on a rendered [H, W, 3] uint8 frame — white
        text with a black outline (mirrors image_io's video overlay) so it stays
        legible on any background; font scales with the preview height."""
        import cv2

        img = np.ascontiguousarray(img)
        scale = max(0.4, img.shape[0] / 480.0)
        thick = max(1, round(scale * 2))
        org = (round(10 * scale), round(28 * scale))
        for color, t in (((0, 0, 0), thick + 2), ((255, 255, 255), thick)):
            cv2.putText(img, f"iter {it}", org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, t, cv2.LINE_AA)
        return img

    def reorient_for_viewer(g: Gaussians) -> Gaussians:
        """Reorient a copy of ``g`` from the COLMAP world (this scene is Z-up)
        into the Y-up frame the gradio Model3D shows upright.

        gradio/Babylon flips the loaded splats' Y (``scaling.y *= -1``), so
        exporting through the reflection E(p)=(x,-z,-y) makes the *displayed*
        scene N(p)=(x,z,-y) — the world's Z-up mapped onto Babylon's Y-up. E's
        point-reflection part leaves the covariance unchanged; its proper-rotation
        part R_p=-E rotates the splat orientations to match.
        """
        from scipy.spatial.transform import Rotation as Rsp

        g = g.clone()
        m = g.means[0]
        g.means = torch.stack([m[:, 0], -m[:, 2], -m[:, 1]], dim=1)[None]  # E
        q = F.normalize(g.rotations_unnorm[0], dim=-1).detach().cpu().numpy()  # xyzw
        R_p = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        q = (Rsp.from_matrix(R_p) * Rsp.from_quat(q)).as_quat()  # R_p @ R, xyzw
        g.rotations_unnorm = torch.from_numpy(q).to(g.means)[None]
        return g

    def export_ply(g: Gaussians) -> str:
        """Write ``g`` (reoriented to Y-up) to a fresh PLY path for the viewer."""
        from pathlib import Path

        holder["ply"] += 1
        path = os.path.join(ply_dir, f"result_{holder['ply']}.ply")
        save_gaussian_ply(reorient_for_viewer(g), save_path=Path(path))
        return path

    g0 = int(init_gaussians.means.shape[1])  # initial Gaussian count, for the stats delta

    def msg(text: str, err: bool = False) -> str:
        """One-line status message for the panel (idle / starting / error)."""
        return f"<div class='l2s-msg{' err' if err else ''}'>{text}</div>"

    def status_only(status: str, n: int) -> tuple:
        """An ``n``-wide outputs tuple that updates only the status field (gui_outputs[1])
        and no-ops the rest — the shape both load/init error paths return so the current
        scene/viewer stays put. ``n`` is the handler's output width (``gui_outputs`` ± extras)."""
        out = [gr.update()] * n
        out[1] = status
        return tuple(out)

    def card(value: str, label: str) -> str:
        """One metric card (big value + small label) for a stats panel."""
        return f"<div class='l2s-stat'><b>{value}</b><span>{label}</span></div>"

    def fmt_status(phase: str, step: int, n_steps: int, g_now: int, t0: float) -> str:
        """Live optimization stats as an HTML metric-card panel. ``phase``: 'run' | 'done'.

        Cards: elapsed time, per-step time (+ step/s when done), Gaussian count (with
        the delta from initialization — densification grows it), and peak VRAM.
        """
        el = time.perf_counter() - t0
        done = max(1, step)
        msps = el / done * 1000.0
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        dg = g_now - g0
        if phase == "done":
            head = f"done · {n_steps} steps"
            t_lbl = "total"
            spd_lbl = f"ms/step · {done / el:.1f}/s" if el > 0 else "ms / step"
        else:
            head = f"optimizing · step {step} / {n_steps}"
            t_lbl = "elapsed"
            spd_lbl = "ms / step"
        g_lbl = f"gaussians · +{dg:,}" if dg > 0 else "gaussians"

        cards = (
            card(f"{el:.1f}s", t_lbl)
            + card(f"{msps:.0f}", spd_lbl)
            + card(f"{g_now:,}", g_lbl)
            + card(f"{peak:.1f} GB", "peak vram")
        )
        return f"<div class='l2s-head'>{head}</div><div class='l2s-stats'>{cards}</div>"

    def preview_stride(step: int) -> int:
        """Live-preview cadence for a 0-based step: dense early, sparse late, so
        the per-step preview render + GPU sync + gradio round-trip don't dominate
        the timed loop. Every iter for the first 10, then every 10 for the next
        200, every 100 for the next 1000, then every 1000 until the end."""
        if step < 10:
            return 1
        if step < 210:
            return 10
        if step < 1210:
            return 100
        return 1000

    def start(optimizer_label, max_steps, batch_size, strategy, view_idx, height, renderer, adc):
        """Generator: run the picked optimizer with the chosen renderer, streaming
        a render each step, then load the finished splats into the Model3D viewer.

        Outputs (per yield) match ``gui_outputs``: image, status, Model3D, Start button,
        Reset button, Evaluate button, eval panel, eval image. Start and Reset are
        mutually exclusive — Start shows while idle/running, then hides and hands
        off to Reset + Evaluate when the run finishes. The Model3D keeps showing the
        initialization during the run, then switches to the refined result.
        """
        name, base = OPTIMIZER_OPTIONS[optimizer_label]
        name = name or "joint"  # None -> the joint checkpoint by default
        inst = instances[name]
        opt = None
        try:
            # FastGS densification's real signals (Abs-GS split gradient, multi-view
            # importance/prune scores) come only from the FastGS rasterizer; on any
            # other backend it silently degrades to vanilla ADC. Refuse that combo.
            if adc_to_name[adc] == "fastgs" and renderer != "fastgs":
                raise ValueError(
                    "FastGS densification requires the FastGS renderer — set "
                    "Renderer to 'fastgs', or pick a different Densification."
                )
            # gsplat: the instance's own checkpoint-matched decoder; else swap in
            # the chosen backend (may raise if its CUDA extension is missing).
            inst.decoder = (
                orig_decoder[name] if renderer == "gsplat" else decoder_for(renderer)
            )
            # Re-init from the pristine copy so repeated Starts share one start point.
            inst.initialize_from_tensors(init_gaussians.clone(), train_bv)
            inst.num_refine = int(max_steps)
            inst.opt_batch_size = min(int(batch_size), n_train_views)
            inst.opt_batch_strategy = strategy
            adc_name = adc_to_name[adc]  # label -> raw refiner name
            # Densification strategy. Learned path: set it on the instance's
            # optimizer. Adam path: the AdamOptimizer below carries it (and overrides
            # the instance for the run). "off" -> fixed Gaussian set.
            inst.configure_adc(adc_name)
            opt = (
                build_adam_baseline(inst.num_refine, adc=adc_name, base=base).to(device)
                if base else None
            )

            # A fresh run invalidates any prior result the Evaluate button scored.
            holder["result"] = holder["decoder"] = None

            # Running: Start disabled, Reset/Evaluate hidden; the 3D viewer keeps showing
            # the initialization until the refined result is ready.
            yield (
                overlay_iter(render(init_gaussians, view_idx, height, renderer), 0),
                f"<div class='l2s-head'>optimizing · step 0 / {inst.num_refine}</div>"
                + msg(f"{g0:,} Gaussians · starting…"),
                gr.update(),                           # Model3D — keep showing the init
                gr.update(interactive=False),          # Start disabled
                gr.update(visible=False),              # Reset hidden
                gr.update(visible=False),              # Evaluate hidden
                gr.update(visible=False, value=None),  # eval panel hidden
                gr.update(visible=False, value=None),  # eval image hidden
            )

            # Time the optimization loop (excludes setup); isolate this run's peak VRAM.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            g = init_gaussians
            for step, g in inst.optimize_iter(optimizer=opt):
                # Stats refresh every step; the live preview render (+ its forced
                # GPU sync) fires only on cadence — off-cadence leaves the image as-is.
                img = (
                    overlay_iter(render(g, view_idx, height, renderer), step + 1)
                    if step % preview_stride(step) == 0
                    else gr.update()
                )
                yield (
                    img,
                    fmt_status("run", step + 1, inst.num_refine, g.means.shape[1], t0),
                    gr.update(), gr.update(), gr.update(),   # Model3D / Start / Reset — no change
                    gr.update(), gr.update(), gr.update(),   # Evaluate / eval panel / eval image
                )

            # Stash the finished result + its decoder so Evaluate can score it.
            holder["result"], holder["decoder"] = g, inst.decoder

            # Append this run as a row in the past-runs table — it's always
            # runs[-1] until the next run finishes, so Evaluate can fill in its
            # metric columns. Peak VRAM is this run's, isolated above.
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            runs.append({
                "method": optimizer_label,
                "renderer": renderer,
                "adc": adc,
                "batch": inst.opt_batch_size,
                "strategy": strategy,
                "steps": inst.num_refine,
                "time": elapsed,
                "ms_step": elapsed / max(1, inst.num_refine) * 1000.0,
                "gaussians": int(g.means.shape[1]),
                "peak_vram": peak,
                "psnr": None, "ssim": None, "lpips": None,
            })

            # Done: show the refined splats in the viewer, reveal Reset + Evaluate, hide Start.
            yield (
                overlay_iter(render(g, view_idx, height, renderer), inst.num_refine),
                fmt_status("done", inst.num_refine, inst.num_refine, g.means.shape[1], t0),
                gr.update(value=export_ply(g)),             # Model3D — refined result
                gr.update(visible=False),                   # Start hidden
                gr.update(visible=True, interactive=True),  # Reset shown
                gr.update(visible=True, interactive=True),  # Evaluate shown
                gr.update(visible=False, value=None),       # eval panel reset
                gr.update(visible=False, value=None),       # eval image reset
            )
        except Exception as e:  # e.g. the inria/fastgs backend isn't installed
            yield (
                gr.update(),  # leave the last image
                msg(f"error — {type(e).__name__}: {str(e)[:200]}", err=True),
                gr.update(),                                # Model3D — leave showing the init
                gr.update(visible=True, interactive=True),  # Start back
                gr.update(visible=False),                   # Reset hidden
                gr.update(visible=False),                   # Evaluate hidden
                gr.update(visible=False, value=None),       # eval panel hidden
                gr.update(visible=False, value=None),       # eval image hidden
            )
        finally:
            # Free the run's CUDA work (also runs if the user hits Stop mid-run),
            # so GPU memory doesn't accumulate across repeated runs.
            opt = None
            gc.collect()
            torch.cuda.empty_cache()

    def init_state_outputs(status_html, view_idx, height, renderer):
        """The full ``gui_outputs`` tuple for the freshly-initialized state: the current
        init shown (live render + 3D viewer), Start ready, Reset/Evaluate hidden. Shared
        by :func:`reset` and :func:`on_initialize` (a fresh init invalidates any run)."""
        holder["result"] = holder["decoder"] = None
        return (
            render(init_gaussians, view_idx, height, renderer),
            status_html,
            gr.update(value=export_ply(init_gaussians)),  # Model3D — the init
            gr.update(visible=True, interactive=True),    # Start shown
            gr.update(visible=False),                     # Reset hidden
            gr.update(visible=False),                     # Evaluate hidden
            gr.update(visible=False, value=None),         # eval panel hidden
            gr.update(visible=False, value=None),         # eval image hidden
        )

    def reset(view_idx, height, renderer):
        """Restore the initialization: re-render it and show it in the 3D viewer, show Start.

        No CUDA cleanup here — ``start``'s ``finally`` already frees each run's work.
        The past-runs table persists — Reset only clears the current result.
        """
        return init_state_outputs(
            msg("Initialized — pick a method, then <b>Start</b>."), view_idx, height, renderer
        )

    @torch.no_grad()
    def evaluate():
        """Score the last finished run on the held-out ``val_bv`` views.

        Renders each val view with the run's own decoder, streaming progress,
        then reports mean novel-view PSNR / SSIM / LPIPS as metric cards and a
        ``gt | pred`` strip (the worst-PSNR views, where artefacts show up).
        """
        from learn2splat.evaluation.metrics import (
            compute_lpips, compute_psnr, compute_ssim,
        )

        g, decoder = holder["result"], holder["decoder"]
        if g is None or decoder is None:
            yield gr.update(), gr.update(visible=True, value=msg(
                "Nothing to evaluate — finish a run first.", err=True)), gr.update()
            return

        h, w = int(val_images.shape[1]), int(val_images.shape[2])
        ext, intr = val_bv["extrinsics"], val_bv["intrinsics"]
        near, far = val_bv["near"], val_bv["far"]

        # Render view-by-view so progress can be streamed; metrics need [B,C,H,W].
        psnrs, ssims, lpipss, strips = [], [], [], []
        for i in range(n_val_views):
            yield (
                gr.update(interactive=False),
                gr.update(visible=True, value=(
                    f"<div class='l2s-head'>evaluating · view {i + 1} / {n_val_views}</div>"
                    + msg("rendering and scoring held-out views…"))),
                gr.update(visible=False, value=None),
            )
            sl = slice(i, i + 1)
            out = decoder.forward(
                g, ext[:, sl], intr[:, sl], near[:, sl], far[:, sl],
                image_shape=(h, w),
            )
            pred = out.color[0, 0].clamp(0.0, 1.0)            # [3, H, W]
            gt = val_images[i].to(device).permute(2, 0, 1)    # [3, H, W]
            psnrs.append(compute_psnr(gt[None], pred[None]).item())
            ssims.append(compute_ssim(gt[None], pred[None]).item())
            lpipss.append(compute_lpips(gt[None], pred[None])[1].item())  # vgg
            strip = torch.cat([gt, pred], dim=2)              # [3, H, 2W] gt | pred
            strips.append(strip)
            imageio.imwrite(
                os.path.join(eval_dir, f"val_{i:04d}.png"),
                (strip.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            )

        # Fill in this run's metric columns — it's runs[-1] (the early guard
        # above means a finished run, hence a row, exists).
        runs[-1].update(
            psnr=float(np.mean(psnrs)),
            ssim=float(np.mean(ssims)),
            lpips=float(np.mean(lpipss)),
        )

        worst = sorted(range(len(psnrs)), key=lambda i: psnrs[i])[:3]
        montage = prep_image(  # stack the worst strips vertically -> [H', 2W, 3]
            torch.cat([strips[i] for i in worst], dim=1)
        )

        panel = (
            "<div class='l2s-head'>evaluation · held-out views</div>"
            "<div class='l2s-stats'>"
            + card(f"{np.mean(psnrs):.2f}", "psnr · db")
            + card(f"{np.mean(ssims):.3f}", "ssim")
            + card(f"{np.mean(lpipss):.3f}", "lpips · vgg")
            + card(f"{n_val_views}", "held-out views")
            + "</div>"
        )
        yield (
            gr.update(interactive=True),
            gr.update(visible=True, value=panel),
            gr.update(visible=True, value=montage),
        )

    def rerender(view_idx, height, renderer):
        """Preview always shows the initialization, from the chosen view + renderer."""
        return render(init_gaussians, view_idx, height, renderer)

    def reload_scene(normalize) -> dict:
        """Rebuild the scene bundle (current cfg.data_path / init_type, normalization
        on/off) and rebind everything the render / start / evaluate closures read —
        Python closures see the rebind. Also refreshes the scene-derived scalars
        (view counts + full-res aspect), which change when the scene changes (the
        normalize/Initialize paths leave them untouched). Returns the bundle."""
        nonlocal init_gaussians, train_bv, val_bv, val_images, g0
        nonlocal n_train_views, n_val_views, h_full, w_full
        bundle = build_scene_bundle(cfg, bool(normalize), instances, device, torch.float32)
        init_gaussians = bundle["gaussians"].clone()
        train_bv, val_bv, val_images = bundle["train_bv"], bundle["val_bv"], bundle["val_images"]
        g0 = int(init_gaussians.means.shape[1])
        n_train_views = int(train_bv["image"].shape[1])
        n_val_views = int(val_bv["image"].shape[1])
        h_full, w_full = train_bv["image"].shape[3], train_bv["image"].shape[4]
        return bundle

    def on_scene_select(scene_label):
        """Reveal the custom-path textbox only for the 'Custom path…' entry."""
        return gr.update(visible=scene_label == CUSTOM_SCENE_LABEL)

    def on_load_scene(scene_label, custom_path, normalize, view_idx, height, renderer,
                      ncv, max_side, batch):
        """Load the selected scene (a curated SCENES entry or an external COLMAP path),
        re-initialize, show it in the 3D viewer, and retarget the view-count-dependent
        widgets (preview view / # context views / opt batch size) to the new scene. The
        past-runs table is cleared (cross-scene metrics aren't comparable). A bad path
        leaves the current scene + ranges intact and reports the error. Returns
        ``gui_outputs + [view_slider, ncv_input, batch_size_input, est_md]``."""
        curated = scene_label in SCENES
        data_path = SCENES[scene_label] if curated else custom_path.strip()
        try:
            ensure_data(data_path)            # named → fetch from HF; custom → existence check
            cfg.data_path = data_path
            reload_scene(normalize)           # refreshes n_train_views / n_val_views / h_full / w_full
        except (Exception, SystemExit) as e:  # bad path / unreadable COLMAP — keep current scene
            return status_only(
                msg(f"could not load {data_path!r} — "
                    f"{type(e).__name__}: {str(e)[:160]}", err=True),
                len(gui_outputs) + 4,  # gui_outputs + the 4 trailing widget/estimate updates
            )
        runs.clear()                          # stale: prior rows are from the old scene
        n = n_train_views
        base = init_state_outputs(
            msg(f"Loaded <b>{scene_label if curated else data_path}</b> · "
                f"{g0:,} Gaussians · {n} train / {n_val_views} val views. "
                f"Pick a method, then <b>Start</b>."),
            view_idx, height, renderer,
        )
        est = resplat_estimate_text(ncv, max_side) if cfg.init_type == "resplat" else gr.update()
        return (
            *base,
            gr.update(maximum=n - 1, value=min(int(view_idx), n - 1)),  # view_slider
            gr.update(maximum=n, value=min(int(ncv), n)),               # ncv_input
            gr.update(maximum=n, value=min(int(batch), n)),             # batch_size_input
            est,                                                        # est_md
        )

    def on_init_type_select(init_label, ncv, max_side):
        """Reveal the selected strategy's headline settings (Random → # points;
        ReSplat → # context views / strategy / resolution; SfM → none) and, for ReSplat,
        refresh the Gaussian-count estimate. Does NOT re-initialize — settings are applied
        with the Initialize button."""
        t = INIT_TYPES[init_label]
        est = resplat_estimate_text(ncv, max_side) if t == "resplat" else gr.update()
        return gr.update(visible=t == "random"), gr.update(visible=t == "resplat"), est

    def on_initialize(init_label, normalize, num_pts, ncv, ctx_strategy, max_side,
                      view_idx, height, renderer):
        """Apply the Initializer-panel settings to cfg, (re)run the initialization, show
        it in the 3D viewer, and reset the run controls to the freshly-initialized state
        (Start shown, Reset/Evaluate hidden). Triggered by the Initialize button — not on
        every setting change. ReSplat's first run builds + caches the learned initializer (a
        few seconds while its checkpoint downloads). Returns ``gui_outputs``."""
        cfg.init_type = INIT_TYPES[init_label]
        cfg.init_num_pts = int(num_pts)
        cfg.init_num_context_views = int(ncv)
        cfg.init_context_strategy = ctx_strategy
        cfg.init_resplat_max_side = int(max_side)
        try:
            bundle = reload_scene(normalize)
        except Exception as e:  # e.g. 'depth' init on a scene with no per-view depth
            # reload_scene raises before rebinding state, so the current init stays put.
            return status_only(
                msg(f"{init_label} init failed — "
                    f"{type(e).__name__}: {str(e)[:180]}", err=True),
                len(gui_outputs),
            )
        return init_state_outputs(
            msg(f"Initialized · <b>{init_label}</b> · {g0:,} Gaussians · normalize "
                f"<b>{'on' if normalize else 'off'}</b> · extent {bundle['scene_scale']:.3f}. "
                f"Pick a method, then <b>Start</b>."),
            view_idx, height, renderer,
        )

    def resplat_estimate_text(ncv, max_side) -> str:
        """Live (config-only) estimate of the resplat Gaussian count at these settings,
        for the chosen checkpoint, #context-views and image resolution. The exact count
        is reported once Initialize runs."""
        from learn2splat.experimental.api import Learn2SplatInitializer

        # All resplat GS checkpoints share the same geometry, so any checkpoint with a
        # reachable config (e.g. the HF hub ref) can proxy for local .pth files whose
        # config.yaml lives on the remote side.
        for ckpt in (cfg.resplat_checkpoint, RESPLAT_INIT_CHECKPOINT):
            try:
                n = Learn2SplatInitializer.estimate_num_gaussians(
                    ckpt, h_full, w_full,
                    num_context_views=int(ncv), max_image_side=int(max_side),
                )
                return msg(f"≈ <b>{n:,}</b> Gaussians  ·  {int(ncv)} views @ ≤{int(max_side)} px")
            except Exception:
                continue
        return msg("estimate unavailable", err=True)

    def on_resplat_ckpt_select(ckpt_label, ncv, max_side):
        """Update cfg.resplat_checkpoint and refresh the Gaussian-count estimate."""
        if ckpt_label not in RESPLAT_CHECKPOINTS:
            return ""
        cfg.resplat_checkpoint = RESPLAT_CHECKPOINTS[ckpt_label]
        return resplat_estimate_text(ncv, max_side)

    def render_runs_table():
        """The past-runs table as HTML; hidden until the first run finishes.

        Chained via ``.then`` after Start / Evaluate, so it re-reads ``runs``
        once each completes — the eval columns show ``—`` until Evaluate fills
        them in for that row.
        """
        if not runs:
            return gr.update(visible=False, value=None)

        def met(r: dict, key: str, fmt: str) -> str:  # eval cell, — until scored
            return "—" if r[key] is None else format(r[key], fmt)

        # (header, value fn, css class) per column. "idx"/"cfg" left-align; "num"
        # right-aligns numbers; "ev" are the eval metrics (highlighted, — if unrun).
        cols = [
            ("#",         lambda r: r["_i"],                          "idx"),
            ("Method",    lambda r: r["method"],                      "cfg"),
            ("Renderer",  lambda r: r["renderer"],                    "cfg"),
            ("Densify",   lambda r: r["adc"],                         "cfg"),
            ("Batch",     lambda r: f"{r['batch']} · {r['strategy']}", "cfg"),
            ("Steps",     lambda r: f"{r['steps']}",                  "num"),
            ("Time",      lambda r: f"{r['time']:.1f}s",              "num"),
            ("ms/step",   lambda r: f"{r['ms_step']:.0f}",            "num"),
            ("Gaussians", lambda r: f"{r['gaussians']:,}",            "num"),
            ("Peak VRAM", lambda r: f"{r['peak_vram']:.1f} GB",       "num"),
            ("PSNR",      lambda r: met(r, "psnr", ".2f"),            "ev"),
            ("SSIM",      lambda r: met(r, "ssim", ".3f"),            "ev"),
            ("LPIPS",     lambda r: met(r, "lpips", ".3f"),           "ev"),
        ]

        def row(r: dict, i: int) -> str:
            r = {**r, "_i": i}
            cells = "".join(f"<td class='{c}'>{fn(r)}</td>" for _, fn, c in cols)
            return f"<tr>{cells}</tr>"

        head = "".join(f"<th class='{c}'>{h}</th>" for h, _, c in cols)
        body = "".join(row(r, i) for i, r in enumerate(runs, 1))
        table = (
            "<div class='l2s-head'>past runs</div>"
            f"<table class='l2s-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )
        return gr.update(visible=True, value=table)

    initial_img = render(init_gaussians, 0, 540, "gsplat")

    # Open the interactive viewer on the same vantage as the live render's
    # default preview (view 0). The viewer shows splats in the reoriented frame
    # N(p)=(x,z,-y) (see reorient_for_viewer); the orbit camera sits at the
    # training view's position and looks at the scene centroid. babylon_camera
    # maps the world camera into N and inverts Babylon's ArcRotateCamera position
    # formula rel=(r·cosα·sinβ, r·cosβ, r·sinα·sinβ) into (alpha°, beta°, radius).
    def babylon_camera(c2w: np.ndarray, centroid: np.ndarray) -> tuple:
        p = c2w[:3, 3] - centroid
        rel = np.array([p[0], p[2], -p[1]], dtype=np.float64)  # N applied to (cam - centroid)
        radius = float(np.linalg.norm(rel)) or 1e-3
        beta = float(np.degrees(np.arccos(np.clip(rel[1] / radius, -1.0, 1.0))))
        alpha = float(np.degrees(np.arctan2(rel[2], rel[0])))
        return (alpha, beta, radius)

    means0 = init_gaussians.means[0].detach().float().cpu().numpy()
    centroid0 = (means0.min(0) + means0.max(0)) / 2.0
    cam0_c2w = train_bv["extrinsics"][0, 0].detach().float().cpu().numpy()
    init_camera = babylon_camera(cam0_c2w, centroid0)

    # --- Visual style: lifted from the Learn2Splat project page
    # (https://autonomousvision.github.io/learn2splat/) — plum accent (#B04080) with an
    # indigo hover, a light slate canvas with white cards, Source Serif 4
    # headings / Inter body / JetBrains Mono labels. ---
    plum = gr.themes.Color(
        c50="#faf0f6", c100="#f5e6ef", c200="#eccadd", c300="#dda3c2",
        c400="#c66ba0", c500="#b04080", c600="#9b3570", c700="#7f2a5b",
        c800="#6a2550", c900="#581f43", c950="#350e26", name="plum",
    )
    slate = gr.themes.Color(
        c50="#f7f8fc", c100="#eef0f8", c200="#e2e5ef", c300="#c8cde0",
        c400="#8890b0", c500="#5a6080", c600="#454b6b", c700="#343a56",
        c800="#262b42", c900="#1a1d2e", c950="#0f1120", name="slate",
    )
    theme = gr.themes.Soft(
        primary_hue=plum,
        neutral_hue=slate,
        font=["Inter", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#f7f8fc",
        body_text_color="#1a1d2e",
        block_background_fill="#ffffff",
        block_border_color="#e2e5ef",
        block_radius="12px",
        block_label_text_color="#5a6080",
        block_title_text_color="#1a1d2e",
        border_color_primary="#e2e5ef",
        input_background_fill="#ffffff",
        button_primary_background_fill="#b04080",
        button_primary_background_fill_hover="#3d50c0",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_border_color="#c8cde0",
        button_secondary_text_color="#1a1d2e",
        slider_color="#b04080",
        link_text_color="#b04080",
    )
    # Source Serif 4 / Inter / JetBrains Mono, loaded into <head> like the page.
    fonts_head = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&"
        "family=Inter:wght@300;400;500;600&"
        'family=JetBrains+Mono:wght@400;500&display=swap">'
    )
    css = """
    .gradio-container { max-width: 1580px !important; margin: 0 auto !important; }
    #l2s-hero { background: linear-gradient(180deg,#f3f4fb 0%,#f7f8fc 100%);
        border:1px solid #e2e5ef; border-radius:14px; padding:26px 30px; margin-bottom:6px; }
    #l2s-hero .eyebrow { font-family:'JetBrains Mono',monospace; font-size:12px;
        letter-spacing:.16em; text-transform:uppercase; color:#b04080; font-weight:500;
        display:inline-flex; align-items:center; gap:8px; }
    #l2s-hero .eyebrow .dot { width:6px; height:6px; border-radius:50%; background:#b04080; }
    #l2s-hero h1 { font-family:'Source Serif 4',Georgia,serif; font-weight:600; color:#1a1d2e;
        font-size:clamp(1.55rem,3vw,2.25rem); line-height:1.2; margin:13px 0 10px; }
    #l2s-hero p { color:#5a6080; font-size:15px; line-height:1.6; margin:0; max-width:820px; }
    #l2s-hero a { color:#b04080; text-decoration:none; border-bottom:1px solid #e6cdd9;
        white-space:nowrap; }
    .l2s-eyebrow span { font-family:'JetBrains Mono',monospace; font-size:11px;
        letter-spacing:.15em; text-transform:uppercase; color:#8890b0; font-weight:500; }
    #l2s-start button, #l2s-reset button, #l2s-eval-btn button, #l2s-init-btn button {
        font-family:'JetBrains Mono',monospace; letter-spacing:.03em; font-weight:500; }
    #l2s-status, #l2s-eval, #l2s-runs { background:#f7f8fc; border:1px solid #e2e5ef;
        border-left:3px solid #b04080; border-radius:9px; padding:11px 14px; }
    #l2s-status .l2s-head, #l2s-eval .l2s-head, #l2s-runs .l2s-head { font-family:'JetBrains Mono',monospace; font-size:11px;
        letter-spacing:.12em; text-transform:uppercase; color:#b04080; font-weight:500; margin:2px 1px 11px; }
    #l2s-status .l2s-stats, #l2s-eval .l2s-stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    #l2s-status .l2s-stat, #l2s-eval .l2s-stat { background:#fff; border:1px solid #e6e8f1; border-radius:8px; padding:8px 11px; }
    #l2s-status .l2s-stat b, #l2s-eval .l2s-stat b { display:block; font-family:'Source Serif 4',Georgia,serif;
        font-size:19px; font-weight:600; color:#1a1d2e; line-height:1.15; }
    #l2s-status .l2s-stat span, #l2s-eval .l2s-stat span { display:block; font-family:'JetBrains Mono',monospace; font-size:9.5px;
        letter-spacing:.04em; text-transform:uppercase; color:#8890b0; margin-top:3px; }
    #l2s-status .l2s-msg, #l2s-eval .l2s-msg { color:#5a6080; font-size:14px; margin:6px 1px; }
    #l2s-status .l2s-msg.err, #l2s-eval .l2s-msg.err { color:#c0392b; word-break:break-word; }
    #l2s-status .l2s-msg b, #l2s-eval .l2s-msg b { color:#1a1d2e; }
    #l2s-runs { overflow-x:auto; }
    #l2s-runs .l2s-table { width:100%; border-collapse:collapse; font-size:12.5px;
        font-family:'JetBrains Mono',monospace; }
    #l2s-runs thead th { font-size:9.5px; font-weight:500; letter-spacing:.06em;
        text-transform:uppercase; color:#8890b0; padding:7px 13px; white-space:nowrap;
        border-bottom:1px solid #dfe3ef; }
    #l2s-runs tbody td { color:#1a1d2e; padding:8px 13px; white-space:nowrap;
        border-bottom:1px solid #eef0f8; }
    #l2s-runs tbody tr:nth-child(even) td { background:#fafbff; }
    #l2s-runs tbody tr:hover td { background:#faf0f6; }
    #l2s-runs tbody tr:last-child td { border-bottom:none; }
    #l2s-runs .idx, #l2s-runs .cfg { text-align:left; }
    #l2s-runs .idx { color:#8890b0; }
    #l2s-runs .num, #l2s-runs .ev { text-align:right; font-variant-numeric:tabular-nums; }
    #l2s-runs .ev { color:#b04080; font-weight:500; }
    footer { display:none !important; }
    """
    hero_html = (
        "<div class='eyebrow'><span class='dot'></span>Learn2Splat · Interactive demo</div>"
        "<h1>Extending the Horizon of Learned 3DGS Optimization</h1>"
        "<p>SfM-initialize a COLMAP scene, then refine the Gaussians with the "
        "meta-learned optimizer — pick a method, press <b>Start</b>, and watch the "
        "decoder render converge. The finished splats load in the interactive 3D "
        "viewer. <a href='https://autonomousvision.github.io/learn2splat/' target='_blank' "
        "rel='noopener'>Project page&nbsp;↗</a></p>"
    )

    with gr.Blocks(
        title="Learn2Splat — Demo", theme=theme, css=css, head=fonts_head,
        analytics_enabled=False,
    ) as ui:
        gr.HTML(hero_html, elem_id="l2s-hero")
        with gr.Row(equal_height=False):
            # Column 1 — controls.
            with gr.Column(scale=3, min_width=300):
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Scene</span></div>")
                    scene_dd = gr.Dropdown(
                        list(SCENES) + [CUSTOM_SCENE_LABEL],
                        value=scene_label_for(cfg.data_path), label="Scene",
                        info="Pick a scene, then press Load scene.",
                    )
                    custom_path_tb = gr.Textbox(
                        value="", label="Custom COLMAP path",
                        info="A directory with images/ + sparse/0/.",
                        visible=scene_label_for(cfg.data_path) == CUSTOM_SCENE_LABEL,
                    )
                    load_scene_btn = gr.Button(
                        "Load scene", variant="primary", elem_id="l2s-scene-btn",
                    )
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Initializer</span></div>")
                    init_dd = gr.Dropdown(
                        INIT_CHOICES, value=init_label_for(cfg.init_type), label="Strategy",
                        elem_id="l2s-init",
                        info="Set the options, then press Initialize.",
                    )
                    # Per-init headline settings — only the selected strategy's group is
                    # shown (toggled by on_init_type_select). SfM has none.
                    with gr.Group(visible=cfg.init_type == "random") as random_group:
                        num_pts_input = gr.Number(
                            value=cfg.init_num_pts, minimum=1, step=1, precision=0,
                            label="Random · # points",
                        )
                    with gr.Group(visible=cfg.init_type == "resplat") as resplat_group:
                        _avail_ckpts = list(RESPLAT_CHECKPOINTS)
                        _init_ckpt_label = next(
                            (k for k, v in RESPLAT_CHECKPOINTS.items()
                             if v == cfg.resplat_checkpoint),
                            _avail_ckpts[0] if _avail_ckpts else None,
                        )
                        resplat_ckpt_dd = gr.Dropdown(
                            _avail_ckpts or ["(no checkpoints found)"],
                            value=_init_ckpt_label or "(no checkpoints found)",
                            label="ReSplat · checkpoint",
                            info="Download checkpoints to checkpoints/resplat_official/ to populate this list.",
                        )
                        ncv_input = gr.Number(
                            value=min(cfg.init_num_context_views, n_train_views),
                            minimum=1, maximum=n_train_views, step=1, precision=0,
                            label="ReSplat · # context views",
                        )
                        ctx_strategy_dd = gr.Dropdown(
                            ["fps", "sequential", "random"],
                            value=cfg.init_context_strategy, label="ReSplat · context strategy",
                        )
                        max_side_input = gr.Number(
                            value=cfg.init_resplat_max_side, minimum=64, step=64, precision=0,
                            label="ReSplat · image resolution (longer side)",
                        )
                        # Live Gaussian-count estimate for the current checkpoint/views/resolution
                        # (config-only, no init run); the actual count appears on Initialize.
                        est_md = gr.Markdown(
                            resplat_estimate_text(cfg.init_num_context_views, cfg.init_resplat_max_side)
                            if cfg.init_type == "resplat" else ""
                        )
                    normalize_chk = gr.Checkbox(
                        value=cfg.normalize_world_space,
                        label="Normalize scene (world space)",
                        info="Off = raw COLMAP scale (matches FastGS). Applied on Initialize.",
                    )
                    initialize_btn = gr.Button(
                        "Initialize", variant="primary", elem_id="l2s-init-btn",
                    )
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Optimizer</span></div>")
                    optimizer_dd = gr.Dropdown(
                        list(OPTIMIZER_OPTIONS), value=next(iter(OPTIMIZER_OPTIONS)), label="Method"
                    )
                    with gr.Row():
                        max_steps_input = gr.Number(
                            value=cfg.max_steps, minimum=1, step=1,
                            precision=0, label="Max steps",
                        )
                        batch_size_input = gr.Number(
                            value=min(cfg.opt_batch_size, n_train_views),
                            minimum=1, maximum=n_train_views, step=1, precision=0,
                            label="Opt batch size",
                        )
                    strategy_dd = gr.Dropdown(
                        ["random", "sequential", "fps"],
                        value=cfg.opt_batch_strategy, label="Batch strategy",
                    )
                    renderer_dd = gr.Dropdown(
                        RENDERERS, value=cfg.renderer, label="Renderer",
                    )
                    adc_dd = gr.Dropdown(
                        ADC_STRATEGIES, value=cfg.adc, label="Densification (ADC)",
                        info=("fastgs needs the FastGS renderer"
                              if "fastgs" in ADC_STRATEGIES else None),
                    )
                with gr.Group():
                    gr.HTML("<div class='l2s-eyebrow'><span>Preview</span></div>")
                    view_slider = gr.Slider(
                        0, n_train_views - 1, value=0, step=1, label="Preview view"
                    )
                    height_slider = gr.Slider(
                        240, 1080, value=540, step=60, label="Render height"
                    )
                with gr.Row():
                    start_btn = gr.Button(
                        "Start optimization", variant="primary",
                        elem_id="l2s-start", scale=2,
                    )
                    reset_btn = gr.Button(
                        "Reset", variant="secondary", elem_id="l2s-reset",
                        scale=1, visible=False,  # appears only when a run finishes
                    )
                evaluate_btn = gr.Button(
                    "Evaluate on held-out views", variant="primary",
                    elem_id="l2s-eval-btn",
                    visible=False,  # appears only when a run finishes
                )
                status_md = gr.HTML(
                    msg("Initialized — pick a method, then <b>Start</b>."),
                    elem_id="l2s-status",
                )
                eval_md = gr.HTML(visible=False, elem_id="l2s-eval")
            # Column 2 — live decoder render (streamed during optimization).
            with gr.Column(scale=5, min_width=380):
                image_out = gr.Image(
                    value=initial_img, label="Optimizer · live",
                    height=540, format="jpeg", interactive=False,
                )
            # Column 3 — interactive splats: the current scene (the initialization, then
            # the refined result after a run). Always visible.
            with gr.Column(scale=5, min_width=380):
                model3d_out = gr.Model3D(
                    value=export_ply(init_gaussians),
                    label="Splats · interactive", height=540,
                    camera_position=init_camera,
                )
                eval_img = gr.Image(
                    label="Held-out views · gt | pred", format="jpeg",
                    interactive=False, visible=False,
                )

        # Full-width past-runs table — one row per finished run, appears below
        # the 3-up layout once the first run completes.
        runs_table = gr.HTML(visible=False, elem_id="l2s-runs")

        start_inputs = [
            optimizer_dd, max_steps_input, batch_size_input, strategy_dd,
            view_slider, height_slider, renderer_dd, adc_dd,
        ]
        preview_inputs = [view_slider, height_slider, renderer_dd]
        gui_outputs = [
            image_out, status_md, model3d_out, start_btn, reset_btn,
            evaluate_btn, eval_md, eval_img,
        ]
        # One shared GPU lane (concurrency_id) so Start / Reset / Evaluate / preview
        # re-renders never run on the GPU at the same time — overlapping runs were
        # the path to runaway VRAM growth.
        # Start / Evaluate refresh the past-runs table once they finish (.then):
        # Start adds the run's row, Evaluate fills in that row's metric columns.
        start_btn.click(
            start, inputs=start_inputs, outputs=gui_outputs, concurrency_id="gpu"
        ).then(render_runs_table, None, runs_table)
        reset_btn.click(
            reset, inputs=preview_inputs, outputs=gui_outputs, concurrency_id="gpu"
        )
        evaluate_btn.click(
            evaluate, inputs=None,
            outputs=[evaluate_btn, eval_md, eval_img], concurrency_id="gpu",
        ).then(render_runs_table, None, runs_table)
        # Preview re-renders the initialization on slider release (not change, to
        # render once when the user lets go) or when the renderer changes.
        view_slider.release(rerender, preview_inputs, image_out, concurrency_id="gpu")
        height_slider.release(rerender, preview_inputs, image_out, concurrency_id="gpu")
        renderer_dd.change(rerender, preview_inputs, image_out, concurrency_id="gpu")
        # Init settings do NOT auto-reload — the user applies them with Initialize.
        # The dropdown only toggles which settings are shown (+ refreshes the ReSplat
        # estimate); the ReSplat settings only refresh the (config-only) count estimate.
        # These handlers touch no GPU, so they stay off the "gpu" lane.
        init_dd.change(
            on_init_type_select, [init_dd, ncv_input, max_side_input],
            [random_group, resplat_group, est_md],
        )
        resplat_ckpt_dd.change(
            on_resplat_ckpt_select, [resplat_ckpt_dd, ncv_input, max_side_input], est_md
        )
        ncv_input.change(resplat_estimate_text, [ncv_input, max_side_input], est_md)
        max_side_input.change(resplat_estimate_text, [ncv_input, max_side_input], est_md)
        # Scene: the dropdown only toggles the custom-path textbox (no GPU); Load scene
        # swaps the scene, re-inits, retargets the view-count-dependent widgets, and clears
        # the now-stale past-runs table (.then hides it, mirroring Start).
        scene_dd.change(on_scene_select, scene_dd, custom_path_tb)
        load_scene_btn.click(
            on_load_scene,
            [scene_dd, custom_path_tb, normalize_chk, view_slider, height_slider,
             renderer_dd, ncv_input, max_side_input, batch_size_input],
            gui_outputs + [view_slider, ncv_input, batch_size_input, est_md],
            concurrency_id="gpu",
        ).then(render_runs_table, None, runs_table)
        # Initialize: apply everything in the Initializer panel (init type, its settings, resolution,
        # normalize) and (re)run the initialization, then refresh the preview.
        initialize_btn.click(
            on_initialize,
            [init_dd, normalize_chk, num_pts_input, ncv_input, ctx_strategy_dd,
             max_side_input, view_slider, height_slider, renderer_dd],
            gui_outputs, concurrency_id="gpu",
        )

    console.print(
        f"[green]✓[/] gradio GUI on port [cyan]{cfg.gui_port}[/]"
        f" — forward the port over SSH and open the printed URL"
    )
    ui.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=cfg.gui_port, share=False,
        show_error=True,
    )


def main(cfg: Config) -> None:
    # Fetch the demo scene on first run, before anything else touches it.
    ensure_data(cfg.data_path)

    from learn2splat.experimental.api import Learn2Splat, Learn2SplatError
    from learn2splat.experimental.api.integration.config_bridge import build_adam_baseline
    from learn2splat.model.decoder import DECODER_CFGS

    if cfg.optimizer_checkpoint is not None:
        CHECKPOINTS["custom"] = cfg.optimizer_checkpoint
        OPTIMIZER_OPTIONS["Learn2Splat (custom)"] = ("custom", None)

    if cfg.renderer not in DECODER_CFGS:
        console.print(
            f"[bold red]renderer {cfg.renderer!r} is not available[/] — its CUDA "
            f"backend isn't installed. Installed: {', '.join(DECODER_CFGS)}. "
            f"Build the optional ones with: WITH_OPTIONAL_RASTERIZERS=1 bash setup.sh"
        )
        raise SystemExit(1)
    if cfg.renderer != "gsplat" and (cfg.rasterize_mode is not None or cfg.eps2d is not None):
        console.print(
            f"[yellow]Note:[/] --rasterize-mode / --eps2d apply to the gsplat "
            f"renderer only; ignored for {cfg.renderer!r}."
        )
    adc_labels, adc_to_name = adc_strategies()
    if cfg.adc not in adc_labels:
        console.print(
            f"[bold red]ADC strategy {cfg.adc!r} is not available[/] — "
            f"choose one of: {', '.join(adc_labels)}."
        )
        raise SystemExit(1)
    adc_name = adc_to_name[cfg.adc]  # raw refiner name for the API
    if adc_name == "fastgs" and cfg.renderer != "fastgs":
        console.print(
            "[bold red]FastGS densification requires the FastGS renderer[/] — "
            "pass --renderer fastgs, or choose a different --adc."
        )
        raise SystemExit(1)
    opt_label = resolve_optimizer(cfg.optimizer) if cfg.optimizer is not None else None
    if cfg.optimizer is not None and opt_label is None:
        console.print(
            f"[bold red]optimizer {cfg.optimizer!r} is not available[/] — choose one of: "
            f"{', '.join(OPTIMIZER_OPTIONS)} (or a slug like 'adam-fastgs')."
        )
        raise SystemExit(1)

    os.makedirs(cfg.result_dir, exist_ok=True)
    device = torch.device(cfg.device)
    dtype = torch.float32

    console.rule("[bold cyan]Learn2Splat demo[/]  ·  Learn2Splat vs Adam")

    # --- COLMAP scene, train/val split ---
    parser = build_parser(cfg, cfg.normalize_world_space)
    dataset = Dataset(parser)
    val_idx = [i for i in range(len(dataset)) if i % cfg.test_every == 0]
    train_idx = [i for i in range(len(dataset)) if i % cfg.test_every != 0]
    scene_scale = scene_extent(parser, cfg.global_scale)
    console.print(
        f"scene scale [cyan]{scene_scale:.4f}[/]  ·  "
        f"train [cyan]{len(train_idx)}[/]  ·  val [cyan]{len(val_idx)}[/]"
    )
    train_bv = build_batched_views(
        *collect_cameras(dataset, train_idx), scene_scale, device, dtype
    )

    # --- Interactive GUI: build both learned-optimizer checkpoints (dense and
    # sparse), initialize each, and open the viser GUI instead of the
    # headless comparison. The GUI's Optimizer dropdown picks between them. ---
    if cfg.with_gui is not None:
        instances = {}
        for name in ("joint", *(("custom",) if cfg.optimizer_checkpoint else ())):
            try:
                instances[name] = Learn2Splat(
                    checkpoint=CHECKPOINTS[name],
                    device=cfg.device,
                    num_refine=cfg.max_steps,
                    opt_batch_size=cfg.opt_batch_size,
                    opt_batch_strategy=cfg.opt_batch_strategy,
                    rasterize_mode=cfg.rasterize_mode,
                    eps2d=cfg.eps2d,
                )
            except Learn2SplatError as e:
                console.print(f"[bold red]Learn2Splat error ({name}):[/] {e}")
                raise SystemExit(1)

        # One init (cfg.init_type) shared by both checkpoints: dense and sparse get
        # an identical starting point, and the GUI shows a single initialization
        # regardless of which optimizer is picked.
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        gaussians = build_initialization(
            parser, cfg, train_bv, instances["joint"].sh_degree, device, dtype
        )
        for inst in instances.values():
            inst.initialize_from_tensors(gaussians, train_bv)

        if cfg.with_gui == "gradio":
            # Held-out views for the GUI's Evaluate button (PSNR / SSIM / LPIPS).
            val_c2w, val_Ks, val_images = collect_cameras(dataset, val_idx)
            val_bv = build_batched_views(
                val_c2w, val_Ks, val_images, scene_scale, device, dtype
            )
            run_gradio_gui(
                instances, gaussians, train_bv, val_bv, val_images, cfg, device
            )
        else:
            run_gui(instances, gaussians, train_bv, cfg, device, dtype)
        return

    val_c2w, val_Ks, val_images = collect_cameras(dataset, val_idx)
    val_bv = build_batched_views(val_c2w, val_Ks, val_images, scene_scale, device, dtype)

    results: dict = {}

    def finish(learn2splat, refined, name: str, elapsed: float) -> None:
        """Persist + evaluate one run's result under results/demo/<name>/."""
        out_dir = os.path.join(cfg.result_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        learn2splat.export_ply(os.path.join(out_dir, "point_cloud.ply"))
        ev = render_and_score(learn2splat, refined, val_bv, val_images, out_dir, device)
        results[name] = {
            "psnr": ev["psnr"], "time": elapsed,
            "num_views": ev["num_views"], "num_GS": int(refined.means.shape[1]),
        }
        console.print(
            f"[green]✓[/] [bold]{name}[/] — PSNR [cyan]{ev['psnr']:.3f}[/]  ·  "
            f"[cyan]{elapsed:.1f}s[/]  → [yellow]{out_dir}[/]"
        )

    def run_one(ckpt_key: str, base: Optional[str], name: str) -> "Learn2Splat":
        """Build a Learn2Splat on a checkpoint, init (cfg.init_type), optimize (the learned optimizer when
        ``base`` is None, else an Adam baseline configured by ``base``), and evaluate. Returns the Learn2Splat."""
        try:
            learn2splat = Learn2Splat(
                checkpoint=CHECKPOINTS[ckpt_key], device=cfg.device, num_refine=cfg.max_steps,
                opt_batch_size=cfg.opt_batch_size, opt_batch_strategy=cfg.opt_batch_strategy,
                rasterize_mode=cfg.rasterize_mode, eps2d=cfg.eps2d,
            )
        except Learn2SplatError as e:
            console.print(f"[bold red]Learn2Splat error ({name}):[/] {e}")
            raise SystemExit(1)
        # gsplat keeps the checkpoint's own decoder (with its rasterize_mode / eps2d); others swap in.
        if cfg.renderer != "gsplat":
            learn2splat.decoder = build_renderer_decoder(cfg.renderer, device)
        learn2splat.configure_adc(adc_name)  # densification strategy (off = fixed set)
        torch.manual_seed(cfg.seed)  # seed after construction so every run gets an identical init
        np.random.seed(cfg.seed)
        gaussians = build_initialization(
            parser, cfg, train_bv, learn2splat.sh_degree, device, dtype
        )
        learn2splat.initialize_from_tensors(gaussians, train_bv)
        opt = None if base is None else build_adam_baseline(learn2splat.num_refine, adc=adc_name, base=base).to(device)
        torch.cuda.synchronize()  # drain setup GPU work so it isn't timed
        tic = time.time()
        refined = learn2splat.optimize(optimizer=opt)
        torch.cuda.synchronize()
        finish(learn2splat, refined, name, time.time() - tic)
        return learn2splat

    # --- Single selected optimizer (--optimizer): run just that method, headless. ---
    if opt_label is not None:
        key, base = OPTIMIZER_OPTIONS[opt_label]
        console.print(
            f"[bold]optimizer:[/] {opt_label}  ·  batch [cyan]{cfg.opt_batch_size}[/]/"
            f"{cfg.opt_batch_strategy}  ·  [cyan]{cfg.max_steps}[/] steps"
        )
        run_one(key or "joint", base, optimizer_slug(opt_label))
        with open(os.path.join(cfg.result_dir, "stats.json"), "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]✓[/] results written to [yellow]{cfg.result_dir}[/]")
        return

    # --- Comparison sweep: Learn2Splat dense + sparse, then a fair Adam baseline ---
    learn2splat = None
    for name in ("joint",):
        learn2splat = None  # free the previous instance before building the next
        torch.cuda.empty_cache()
        learn2splat = run_one(name, None, name)

    # Adam baseline: reuse the last instance — same SfM init / views / budget / renderer; only the
    # update rule differs (optimize() re-inits from the stored SfM init, keeping the comparison fair).
    adam = build_adam_baseline(learn2splat.num_refine, adc=adc_name).to(device)
    torch.cuda.synchronize()  # drain setup GPU work so it isn't timed
    tic = time.time()
    refined_adam = learn2splat.optimize(optimizer=adam)
    torch.cuda.synchronize()
    finish(learn2splat, refined_adam, "adam", time.time() - tic)

    # --- Comparison table ---
    table = Table(
        title=(
            f"Novel-view PSNR  ·  {results['joint']['num_views']} held-out "
            f"views  ·  {cfg.max_steps} steps  ·  "
            f"{results['joint']['num_GS']} Gaussians"
        ),
        title_style="bold",
        caption=(
            f"{cfg.renderer} renderer  ·  adc={cfg.adc}  ·  "
            f"rasterize_mode={cfg.rasterize_mode or 'per-checkpoint'}  ·  "
            f"eps2d={cfg.eps2d if cfg.eps2d is not None else 'per-checkpoint'}"
        ),
    )
    table.add_column("Optimizer")
    table.add_column("PSNR (dB)", justify="right")
    table.add_column("Time (s)", justify="right")
    best = max(results, key=lambda k: results[k]["psnr"])
    for key, label in (
        ("joint", "Learn2Splat"),
        ("adam", "Adam"),
    ):
        table.add_row(
            label,
            f"{results[key]['psnr']:.3f}",
            f"{results[key]['time']:.1f}",
            style="bold green" if key == best else None,
        )
    console.print(table)

    with open(os.path.join(cfg.result_dir, "stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓[/] results written to [yellow]{cfg.result_dir}[/]")


if __name__ == "__main__":
    main(tyro.cli(Config))
