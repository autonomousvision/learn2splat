from dataclasses import dataclass
from typing import Literal, Optional

from learn2splat.dataset.data_types import BatchedViews
import torch
import math
import torch.nn.functional as F
from pathlib import Path
from learn2splat.experimental.edgs.init import init_gaussians_with_corr
from learn2splat.experimental.initializers_utils import points_to_gaussians
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.scene_trainer.initializer.initializer import InitializerOutput, NonlearnedInitializer, NonlearnedInitializerCfg


@dataclass
class InitializerEdgsCfg(NonlearnedInitializerCfg):
    name: Literal["edgs"]
    sh_degree: int
    init_opacity: float
    scaling_factor: float
    roma_model_type: str

    sample_init_gaussians: int  # if >0, randomly sample this many gaussians from the initialized set
    cache_dir: str  # root dir for the per-scene points_dict cache (built once per scene by RoMa)

    def get_sh_d(self):
        sh_d = (self.sh_degree + 1) ** 2
        return sh_d


class InitializerEdgs(NonlearnedInitializer[InitializerEdgsCfg]):
    def __init__(self, cfg: InitializerEdgsCfg) -> None:
        super().__init__(cfg)

    def forward(
            self,
            context: BatchedViews,
            visualization_dump: Optional[dict] = None,
            scene: Optional[list[str]] = None,
            **kwargs
    ) -> InitializerOutput:

        # Per-scene cache for the (expensive RoMa-correspondence) points_dict.
        # NOTE: keyed only by scene name, not by the selected context views — re-evaluating
        # the same scene with a different context set will hit a stale cache.
        cached_data_path = Path(self.cfg.cache_dir) / scene[0] if scene is not None else None

        device = context["extrinsics"].device

        # unpack context (batch_dim = 1)
        viewpoints_img = context["image"].squeeze(0)  # [N, 3, H, W]
        h, w = viewpoints_img.shape[2], viewpoints_img.shape[3]

        # poses
        viewpoints_c2w = context["extrinsics"].squeeze(0).clone()  # [N, 4, 4]
        camera_centers = viewpoints_c2w[..., :3, 3]
        viewpoints_w2c = torch.inverse(viewpoints_c2w)  # [N, 4, 4]

        # convert to column-major
        viewpoints_w2c = viewpoints_w2c.permute(0, 2, 1)
        
        # intrinsics
        viewpoints_intrinsics = context["intrinsics"].squeeze(0).clone()  # [N, 3, 3]
        # un-normalize intrinsics by multiplying by image size
        viewpoints_intrinsics[:, 0, :] *= w
        viewpoints_intrinsics[:, 1, :] *= h

        def getProjectionMatrix(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P

        def focal2fov(focal, pixels):
            return 2 * math.atan(pixels / (2 * focal))

        viewpoints_proj = []
        for idx, intrinsic in enumerate(viewpoints_intrinsics):
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            znear = 0.01
            zfar = 100.0
            fovY = focal2fov(fy, h)
            fovX = focal2fov(fx, w)
            proj = getProjectionMatrix(
                znear=znear, zfar=zfar, fovX=fovX, fovY=fovY
            ).transpose(0, 1).cuda()
            viewpoints_proj.append(proj)
        viewpoints_proj = torch.stack(viewpoints_proj, dim=0)  # [N, 4, 4]

        # compute full projection matrices
        viewpoints_full_proj = (viewpoints_w2c.bmm(viewpoints_proj))  # [N, 4, 4]

        # check if points_dict is stored on disk already (cached)
        found_cached = False
        if cached_data_path is not None:
            print("Checking for cached points_dict at:", str(cached_data_path))
            cache_path = cached_data_path / "points_dict.pt"
            if cache_path.exists():
                points_dict = torch.load(cache_path)
                print("Loaded cached points_dict from:", str(cache_path))
                found_cached = True
            else:
                print("No cached points_dict found at:", str(cache_path))

        if not found_cached:
            # recompute points_dict
            _, _, points_dict = init_gaussians_with_corr(
                viewpoints_img=viewpoints_img,  # [N, 3, H, W]
                viewpoints_w2c=viewpoints_w2c,  # [N, 4, 4]
                viewpoints_proj=viewpoints_full_proj,  # [N, 4, 4]
                camera_centers=camera_centers,  # [N, 3]
                init_opacity=self.cfg.init_opacity,
                roma_model_type=self.cfg.roma_model_type,
                verbose=False
            )
            if cached_data_path is not None:
                print("Saving points_dict to cache at:", str(cache_path))
                cached_data_path.mkdir(parents=True, exist_ok=True)
                torch.save(points_dict, cache_path)

        points_dict["scales"] *= self.cfg.scaling_factor
        
        # printing some stats
        for k, v in points_dict.items():
            print(f"points_dict[{k}]: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
        
        # downsample if needed
        if self.cfg.sample_init_gaussians > 0:
            # randomly sample a subset of gaussians
            total_points = points_dict["xyz"].shape[0]
            sample_num = min(self.cfg.sample_init_gaussians, total_points)
            sampled_indices = torch.randperm(total_points)[:sample_num]
            points_dict = {k: v[sampled_indices] for k, v in points_dict.items()}
            print("Nr points after sampling:", points_dict["xyz"].shape[0])

        
        # pre-activation values on device
        gaussians_dict = points_to_gaussians(points_dict, sh_degree=self.cfg.sh_degree, device=device)

        means = gaussians_dict["xyz"]
        sh0 = gaussians_dict["sh0"]
        shN = gaussians_dict["shN"]
        harmonics = torch.cat([sh0, shN], dim=1)  # [N, sh_d, 3]
        harmonics = harmonics.permute(0, 2, 1)  # [N, 3, sh_d]
        rotations_unnorm = gaussians_dict["rotations_unnorm"]

        # post-activation values
        opacities = torch.sigmoid(gaussians_dict["opacities_raw"])
        scales = torch.exp(gaussians_dict["scales_raw"])
        rotations = F.normalize(gaussians_dict["rotations_unnorm"], dim=-1)
        covariances = build_covariance(scale=scales, rotation_xyzw=rotations)
        
        print("Nr gaussians initialized:", means.shape[0])

        gaussians = Gaussians(
            means=means.unsqueeze(0),
            covariances=covariances.unsqueeze(0),
            harmonics=harmonics.unsqueeze(0),  # [1, N, 3, sh_d]
            opacities=opacities.unsqueeze(0),
            scales=scales.unsqueeze(0),
            rotations=rotations.unsqueeze(0),
            rotations_unnorm=rotations_unnorm.unsqueeze(0),
        )

        return InitializerOutput(
            gaussians=gaussians,
            features=None,
            depths=None
        )
