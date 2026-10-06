from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from learn2splat.dataset.data_types import BatchedViews
from learn2splat.experimental.initializers_utils import colored_points_to_gaussians
from learn2splat.geometry.projection import get_world_rays, sample_image_grid
from learn2splat.scene_trainer.initializer.initializer import (
    InitializerOutput,
    NonlearnedInitializer,
    NonlearnedInitializerCfg,
)


@dataclass
class InitializerDepthCfg(NonlearnedInitializerCfg):
    name: Literal["depth"]
    scaling_factor: float
    init_opacity: float
    sh_degree: int
    # Resize the context image+depth so the longer side equals this many pixels BEFORE
    # unprojecting; controls point-cloud density (one point per pixel per view). null = native
    # resolution. Mirrors ReSplat's "choose the unprojection resolution" behaviour. Normalized
    # intrinsics are resolution-independent, so no intrinsics adjustment is needed when resizing.
    init_longer_side: Optional[int]
    # Drop pixels whose depth is <= 0 (the converter's background/invalid sentinel).
    filter_invalid_depth: bool
    # Train-only per-pixel depth noise (relative std): multiplies each pixel's GT depth by
    # (1 + N(0, std)) before unprojecting, so the near-perfect GT-depth init has geometry for the
    # optimizer to fix. 0.0 disables it. Applied only in training (eval uses the clean init).
    train_depth_noise_std: float

    def get_sh_d(self):
        return (self.sh_degree + 1) ** 2


class InitializerDepth(NonlearnedInitializer[InitializerDepthCfg]):
    """Initialize Gaussians by unprojecting context RGB-D into a colored point cloud.

    Only usable on datasets that provide per-view depth (e.g. MegaSynth with
    `dataset.load_depth=true`). Each context view's depth is back-projected to world
    space using its (normalized) intrinsics and camera-to-world extrinsics, colored by
    the RGB image, and the per-view clouds are concatenated. The colored point cloud is
    subsampled (`subsample_points`), then built into Gaussians by `colored_points_to_gaussians`
    (knn scales, constant opacity, RGB->SH, pad to the fixed DDP count).
    """

    def __init__(self, cfg: InitializerDepthCfg) -> None:
        super().__init__(cfg)

    def _unproject_context(
        self,
        context: BatchedViews,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (xyz [N,3], rgb [N,3] in [0,1]) world-space colored points from all context views."""
        image = context["image"][0].to(device)          # [V, 3, H, W] in [0,1]
        depth = context["depth"][0].to(device).float()  # [V, H, W]
        intrinsics = context["intrinsics"][0].to(device)  # [V, 3, 3] normalized
        extrinsics = context["extrinsics"][0].to(device)  # [V, 4, 4] camera-to-world (OpenCV)
        v, _, h, w = image.shape

        # Defensive: align depth to the image's spatial size (e.g. if a patch shim cropped the image).
        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)

        # Optionally downscale so the longer side hits the requested resolution.
        if self.cfg.init_longer_side is not None and max(h, w) != self.cfg.init_longer_side:
            s = self.cfg.init_longer_side / max(h, w)
            h, w = max(1, round(h * s)), max(1, round(w * s))
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
            depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)

        # Per-pixel depth noise (train only; see train_depth_noise_std) -- also gives the render_depth
        # loss a real gap to close, not just position refinement.
        if self.training and self.cfg.train_depth_noise_std > 0:
            depth = depth * (1.0 + torch.randn_like(depth) * self.cfg.train_depth_noise_std)

        # Normalized pixel grid (0..1), shared across views; intrinsics are normalized too.
        coords, _ = sample_image_grid((h, w), device=device)  # [H, W, 2]
        coords = rearrange(coords, "h w xy -> (h w) xy")       # [P, 2]
        p = coords.shape[0]

        xyz_list, rgb_list = [], []
        for vi in range(v):
            origins, directions = get_world_rays(
                coords, extrinsics[vi].expand(p, 4, 4), intrinsics[vi].expand(p, 3, 3)
            )  # directions normalized to camera-space z=1, so origins + directions*z-depth is correct
            d = depth[vi].reshape(p)                            # [P] z-depth
            pts = origins + directions * d[:, None]            # [P, 3] world points
            cols = image[vi].reshape(3, p).permute(1, 0)       # [P, 3] in [0,1]
            if self.cfg.filter_invalid_depth:
                valid = d > 0
                pts, cols = pts[valid], cols[valid]
            xyz_list.append(pts)
            rgb_list.append(cols)

        return torch.cat(xyz_list, dim=0), torch.cat(rgb_list, dim=0)

    def forward(
        self,
        context: BatchedViews,
        visualization_dump: Optional[dict] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> InitializerOutput:
        if "depth" not in context:
            raise ValueError(
                "The 'depth' initializer requires per-view depth; set dataset.load_depth=true."
            )
        assert context["image"].shape[0] == 1, "Only single-scene initialization is supported."
        if device is None:
            device = context["image"].device

        xyz, rgbs = self._unproject_context(context, device)
        # Subsample the point cloud ("sample Gaussians") before building the Gaussians.
        xyz, rgbs = self.subsample_points(xyz, rgbs)
        gaussians = colored_points_to_gaussians(
            xyz,
            rgbs,
            sh_degree=self.cfg.sh_degree,
            scaling_factor=self.cfg.scaling_factor,
            init_opacity=self.cfg.init_opacity,
            fixed_num=self.fixed_gaussians_num,
            device=device,
        )
        return InitializerOutput(gaussians=gaussians, features=None, depths=None)
