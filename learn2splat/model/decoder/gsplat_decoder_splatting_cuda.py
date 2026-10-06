from dataclasses import dataclass
from typing import Literal

import math
import torch
# import torch.nn.functional as F
from gsplat.rendering import rasterization
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from learn2splat.scene_trainer.gaussian_module import GaussiansModule
from .decoder import Decoder, DecoderOutput
from .decoder import DepthRenderingMode
from ..types import Gaussians
from ...dataset import DatasetCfg


@dataclass
class GSplatDecoderSplattingCUDACfg:
    name: Literal["gsplat"]
    use_covariances: bool
    rasterize_mode: Literal["antialiased", "classic"]
    eps2d: float


class GSplatDecoderSplattingCUDA(Decoder[GSplatDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GSplatDecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def means2d_grad_to_ndc(self, grad, image_shape):
        # gsplat's means2d is in pixel space, so its gradient scales with image size. Convert to
        # the resolution-independent NDC convention by * (w/2, h/2). (Inria/FastGS already emit
        # NDC and inherit the identity default — see Decoder.means2d_grad_to_ndc.)
        h, w = image_shape
        return grad * grad.new_tensor([w / 2.0, h / 2.0])

    def forward(
        self,
        gaussians: Gaussians | GaussiansModule,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,  # always render depth
        return_radii: bool = False,  # always return radii
        iter_batch_size: int = -1,  # -1 to render all views at once
        use_covariances: bool = False,  # override cfg
        to_cpu: bool = False,  # move outputs to cpu
    ) -> DecoderOutput:
        
        _use_covariances = self.cfg.use_covariances if use_covariances is None else use_covariances
        
        height, width = image_shape
        
        if isinstance(gaussians, GaussiansModule):
            
            # nb: no batch dimension
            
            means = gaussians.means
            quats = gaussians.rotations  # [N, 4] in xyzw (scalar last)
            quats = quats[:, [3, 0, 1, 2]]  # [N, 4] in wxyz (scalar first)
            quats = quats  # [1, N, 4]
            scales = gaussians.scales  # post-activation
            opacities = gaussians.opacities  # post-activation
            colors = gaussians.harmonics.permute(0, 2, 1)  # [1, N, d_sh, 3]
            if _use_covariances:
                covars = gaussians.covariances
            else:
                covars = None
                
            # add batch dimension
            means = means.unsqueeze(0)  # [1, N, 3]
            quats = quats.unsqueeze(0)  # [1, N, 4]
            scales = scales.unsqueeze(0)  # [1, N, 3]
            opacities = opacities.unsqueeze(0)  # [1, N, 1]
            colors = colors.unsqueeze(0)  # [1, N, d_sh, 3]
            if covars is not None:
                covars = covars.unsqueeze(0)  # [1, N, 3, 3]
            
        elif isinstance(gaussians, Gaussians):
            
            means = gaussians.means  # [B, N, 3]
            quats = gaussians.rotations_unnorm  # [B, N, 4] in wxyz (scalar first), rasterization normalizes internally
            quats = quats[:, :, [3, 0, 1, 2]]  # [B, N, 4] in wxyz (scalar first)
            scales = gaussians.scales  # [B, N, 3]
            opacities = gaussians.opacities  # [B, G]
            colors = gaussians.harmonics.permute(0, 1, 3, 2)  # [B, N, d_sh, 3]
            if _use_covariances:
                covars = gaussians.covariances  # [B, N, 3, 3]
                if covars is None:  
                    raise ValueError("Covariances are set to be used, but gaussians.covariances is None.")
            else:
                covars = None
                
            if gaussians.stores_activated:
                # already activated
                pass
            else:
                # activate
                scales = torch.exp(scales)  # [B, N, 3]
                opacities = torch.sigmoid(opacities)  # [B, N]
            
        else:
            raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")
        
        # prepare inputs for rasterization
        sh_degree = int(math.sqrt(colors.shape[-2])) - 1  # d_sh = (degree + 1) ** 2
        viewmats = extrinsics.inverse()  # [B, V, 4, 4]

        # scale intrinsics to image shape (avoid clone by creating scaled version directly)
        intrinsics_scaled = intrinsics * intrinsics.new_tensor([[[width], [height], [1]]])  # [B, V, 3, 3]

        def _render(viewmats, Ks):

            # rasterize
            render_colors, render_alphas, meta = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                sh_degree=sh_degree,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                # near_plane=near[0, 0].item(),  # use default
                # far_plane=far[0, 0].item(),  # use default
                eps2d=self.cfg.eps2d,
                rasterize_mode=self.cfg.rasterize_mode,
                packed=False,
                # absgrad=False, # use default
                # sparse_grad=False, # use default
                render_mode="RGB+ED",
                # with_ut=False, # use default
                # with_eval3d=False, # use default
                # covars=covars, # use default
            )
    
            # unpack outputs
            color = render_colors[..., :3].permute(0, 1, 4, 2, 3)  # [B, V, 3, H, W]
            depth = render_colors[..., -1]  # [B, V, H, W]
            means2d = meta["means2d"]  # [B, V, N, 2]
            radii = meta["radii"]  # [B, V, N, 2]
            visibility_filter = torch.all(radii > 0, dim=-1)  # [B, V, N]
            
            return color, depth, render_alphas, means2d, visibility_filter, radii
        
        # split into chunks to save memory
        nr_views = extrinsics.shape[1]
        if iter_batch_size < 0:
            # render all views at once
            color, depth, render_alphas, means2d, visibility_filter, radii = _render(viewmats, intrinsics_scaled)
            if to_cpu:
                color = color.detach().cpu()
                depth = depth.detach().cpu()
                render_alphas = render_alphas.detach().cpu()
                means2d = means2d.detach().cpu()
                visibility_filter = visibility_filter.detach().cpu()
                radii = radii.detach().cpu()
        else:
            # split into chunks
            chunk_outputs = []
            for i in tqdm(range(0, nr_views, iter_batch_size), desc="Rendering in batches"):
                if i + iter_batch_size > nr_views:
                    bs = nr_views - i
                else:
                    bs = iter_batch_size
                iter_viewmats = viewmats[:, i : i + bs]  # [B, v, 4, 4]
                iter_intrinsics = intrinsics_scaled[:, i : i + bs]  # [B, v, 3, 3]
                color, depth, render_alphas, means2d, visibility_filter, radii = _render(iter_viewmats, iter_intrinsics)
                if to_cpu:
                    color = color.detach().cpu()
                    depth = depth.detach().cpu()
                    render_alphas = render_alphas.detach().cpu()
                    means2d = means2d.detach().cpu()
                    visibility_filter = visibility_filter.detach().cpu()
                    radii = radii.detach().cpu()
                chunk_outputs.append((color, depth, render_alphas, means2d, visibility_filter, radii))
            
            # concatenate all chunks
            color = torch.cat([o[0] for o in chunk_outputs], dim=1)  # [B, V, 3, H, W]
            depth = torch.cat([o[1] for o in chunk_outputs], dim=1)  # [B, V, H, W]
            render_alphas = torch.cat([o[2] for o in chunk_outputs], dim=1)  # [B, V, H, W, 1]
            means2d = torch.cat([o[3] for o in chunk_outputs], dim=1)  # [B, V, N, 2]
            visibility_filter = torch.cat([o[4] for o in chunk_outputs], dim=1)  # [B, V, N]
            radii = torch.cat([o[5] for o in chunk_outputs], dim=1)  # [B, V, N, 2]
        
        return DecoderOutput(
            color,
            depth=depth,
            accumulated_alpha=render_alphas.squeeze(-1),  # [B, V, H, W]
            means2d=means2d,
            visibility_filter=visibility_filter,
            radii=radii,
        )


