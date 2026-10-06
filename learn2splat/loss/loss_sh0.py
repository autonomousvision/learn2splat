from dataclasses import dataclass

import torch
import torch.nn.functional
from einops import rearrange
from gsplat.exporter import sh2rgb

from learn2splat.loss import Loss
from learn2splat.model.decoder.decoder import DecoderOutput


@dataclass
class LossSh0Cfg:
    weight: float

@dataclass
class LossSh0CfgWrapper:
    sh0: LossSh0Cfg

class LossSh0(Loss[LossSh0Cfg, LossSh0CfgWrapper]):
    """Supervises each Gaussian's SH degree-0 (base) color against the ground-truth image colour
    sampled at the Gaussian's projected 2D location, averaged over the views where it is visible."""

    def forward(
        self,
        prediction: DecoderOutput,
        gaussians,
        global_step: int,
        gt_rgb: torch.Tensor,
        pred_rgb: torch.Tensor,
        valid_depth_mask,
        gt_image: torch.Tensor,  # full-res image [B, V, C, H, W], all views un-subsampled
        **kwargs,
    ):
        sh0_pred = gaussians.harmonics[..., 0]  # [B, G, 3]
        # Convert SH0 to RGB
        rgb_pred = sh2rgb(sh0_pred)  # [B, G, 3]

        rgb = gt_image  # [B, V, C, H, W]
        h, w = rgb.shape[-2:]
        means2d = prediction.means2d.detach().clone()  # [B, V, G, 2]
        means2d[..., 0] = (means2d[..., 0] / (w - 1)) * 2 - 1
        means2d[..., 1] = (means2d[..., 1] / (h - 1)) * 2 - 1
        rgb_gt = torch.nn.functional.grid_sample(rearrange(rgb, "b v c h w -> (b v) c h w"),
                                                     rearrange(means2d, "b v g c -> (b v) 1 g c"),
                                                     align_corners=False,
                                                     padding_mode="border")  # [(B V), 3, 1, G]
        rgb_gt = rearrange(rgb_gt, "(b v) c 1 g -> b v g c", b=rgb.shape[0], v=rgb.shape[1])  # [B, V, G, 3]
        # Calculate mean over views, exclude invalid pixels
        # Calculate only for valid intersection of the gaussians and the views
        radii = prediction.radii.detach()  # [B, V, G, 2]
        # Gaussian didn't contribute to this view
        # For these gaussians, means2d is (0,0), so we want to exclude them from the computation
        valid = (radii > 0).all(-1, keepdim=True)  # [B, V, G, 1]
        valid_counts = valid.sum(1)  # [B, G, 1]
        denom = valid_counts + (valid_counts == 0).float()  # avoid division by zero
        rgb_gt_avg = rgb_gt * valid  # [B, V, G, 3]
        rgb_gt_avg = rgb_gt_avg.sum(1) / denom  # [B, G, 3]

        error = rgb_pred - rgb_gt_avg
        error = error[(valid_counts > 0)[..., 0]]
        loss = (error ** 2).abs().mean()
        return loss * self.cfg.weight

