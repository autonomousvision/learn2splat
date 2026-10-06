from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn.functional as F

from learn2splat.dataset.data_types import BatchedViews
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.model.types import Gaussians
from learn2splat.experimental.initializers_utils import nn_dist2_for_scales, points_to_gaussians
from learn2splat.scene_trainer.initializer.initializer import NonlearnedInitializer, InitializerOutput, NonlearnedInitializerCfg
from learn2splat.dataset.camera_datasets.camera import get_scene_scale


@dataclass
class InitializerRandomCfg(NonlearnedInitializerCfg):
    name: Literal["random"]
    init_num_pts: int
    init_extent: float
    scaling_factor: float
    init_opacity: float
    sh_degree: int
    
    def get_sh_d(self):
        sh_d = (self.sh_degree + 1) ** 2
        return sh_d
    

class InitializerRandom(NonlearnedInitializer[InitializerRandomCfg]):
    def __init__(self, cfg: InitializerRandomCfg) -> None:
        super().__init__(cfg)
        
    def forward(
        self,
        context: BatchedViews,
        **kwargs
    ) -> InitializerOutput:
        
        device = context["extrinsics"].device
        init_num_pts = self.cfg.init_num_pts
        init_extent = self.cfg.init_extent
        
        # calculate scene scale from context
        camtoworlds = context["extrinsics"].cpu().numpy()  # [B, 4, 4]
        assert camtoworlds.shape[0] == 1, "Batch size > 1 not supported in random initializer"
        camtoworlds = camtoworlds.squeeze(0)
        scene_scale = get_scene_scale(camtoworlds)
        
        xyz = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
        
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = nn_dist2_for_scales(xyz)  # [N]
        dist_avg = torch.sqrt(dist2_avg)
        scales = dist_avg.unsqueeze(-1).repeat(1, 3)  # [N, 3]
        
        points_dict = {
            "xyz": xyz,
            "rgb": rgbs,
            "scales": scales,
            "opacities": torch.full((xyz.shape[0],), self.cfg.init_opacity),
        }
        
        points_dict["scales"] *= self.cfg.scaling_factor
        
        # pre-activation values on device
        gaussians_dict = points_to_gaussians(points_dict, sh_degree=self.cfg.sh_degree, device=device)

        means = gaussians_dict["xyz"]
        sh0 = gaussians_dict["sh0"]
        shN = gaussians_dict["shN"]
        harmonics = torch.cat([sh0, shN], dim=1) # [N, sh_d, 3]
        harmonics = harmonics.permute(0, 2, 1)  # [N, 3, sh_d]
        rotations_unnorm = gaussians_dict["rotations_unnorm"]
        
        # post-activation values
        opacities = torch.sigmoid(gaussians_dict["opacities_raw"])
        scales = torch.exp(gaussians_dict["scales_raw"])
        rotations = F.normalize(gaussians_dict["rotations_unnorm"], dim=-1)
        covariances = build_covariance(scale=scales, rotation_xyzw=rotations)
        
        gaussians = Gaussians(
            means=means.unsqueeze(0),
            covariances=covariances.unsqueeze(0),
            harmonics=harmonics.unsqueeze(0),  # [1, N, C, sh_d]
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