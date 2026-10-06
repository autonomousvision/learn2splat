from dataclasses import dataclass
from typing import Literal, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import nn

from learn2splat.dataset.data_types import BatchedExample, DataShim, BatchedViews
from learn2splat.dataset.shims.patch_shim import apply_patch_shim
from learn2splat.geometry.projection import sample_image_grid, get_world_rays
from learn2splat.misc.general_utils import rotate_quats
from learn2splat.misc.io import FrequencyScheduler
from learn2splat.model.backbones.layer import BasicBlock
from learn2splat.model.backbones.unimatch.dpt_head import DPTHead
from learn2splat.model.backbones.unimatch.feature_upsampler import ResizeConvFeatureUpsampler
from learn2splat.model.backbones.unimatch.ldm_unet.unet import UNetModel
from learn2splat.model.backbones.unimatch.mv_unimatch import MultiViewUniMatch
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, build_covariance, RGB2SH
from learn2splat.scene_trainer.common.gaussians import quaternion_multiply_xyzw, rotation_matrix_to_quaternion_xyzw
from learn2splat.scene_trainer.initializer.initializer import InitializerOutput, LearnedInitializer, PerPixelInitializerCfg

try:
    from learn2splat.model.backbones.point_transformer.layer import PlainPointTransformer
except:
    pass


@dataclass
class ResplatInitializerCfg(PerPixelInitializerCfg):
    name: Literal["resplat_v1", "resplat_v2"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int | List[int]
    multiview_trans_attn_split: int

    deform_sample_depth: bool  # non-pixel aligned Gaussians with learned offsets
    deform_sample_depth_debug: bool  # check depth sampling

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int

    # point transformer
    pt_head: bool
    init_pt_with_mv_attn: bool
    init_pt_with_mv_attn_lowres: bool
    pt_head_conv: bool
    pt_head_concat_img: bool
    pt_head_channels: int | None
    attn_proj_channels: int | None
    knn_samples: int
    post_norm: bool
    no_rpe: bool
    no_knn_attn: bool
    num_blocks: int
    add_pt_residual: bool
    latent_dpt_upsampler: bool
    latent_dpt_upsampler_no_concat: bool
    light_dpt_feature: bool

    # freeze depth
    freeze_depth: bool
    use_gt_depth: bool

    # separate depth and color branches
    separate_depth_color: bool
    separate_depth_type: str
    separate_depth_gaussian_scale: bool

    # unet gaussian regressor
    unet_gaussian_regressor: bool
    resnet_gaussian_regressor: bool

    sample_log_depth: bool
    bilinear_upsample_depth: bool
    no_upsample_depth: bool
    return_lowres_depth: bool

    # latent gaussian instead of pixel-aligned gaussian
    fixed_latent_size: bool  # same channels for both downsample 4 and 8
    latent_gs_img_interp: str
    dpt_head_depth: bool  # downsample the full resolution depth to low resolution
    avgpool_depth: bool
    nearest_down_depth: bool

    # predict scene scale and use point distance to normalize the scene
    predict_scale: bool
    norm_by_points: bool
    no_pred_depth_range: bool

    # init gaussian scale with point cloud distance
    point_dist_init_gaussian_scale: bool

    # feature upsampler
    resizeconv_upsampler: bool

    # rotate_quat_to_world: bool  # rotate the quaternion to world space
    # rotations_unnorm = c2w * raw camera-frame quaternion (keeps its norm/sign) for the released ReSplat optimizer
    camera_frame_quats: bool
    latent_new_reshape: bool  # debug

    # amp
    use_amp: bool
    pt_head_amp: bool

    use_checkpointing: bool
    init_use_checkpointing: bool  # init model uses checkpointing
    no_pixel_offset: bool

    pt_heads: int
    init_gaussian_multiple: int
    depth_pred_half_res: bool

    def get_feature_upsampler_channels(self):
        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        vit_type = self.monodepth_vit_type
        in_channels = model_configs[vit_type]['in_channels']

        if self.latent_gs and not self.latent_dpt_upsampler:
            if self.latent_downsample == 2:
                feature_num = in_channels // 64 * 4 + 128 // 4 + 64 + 96 + 128 // 4
            elif self.latent_downsample == 4:
                feature_num = in_channels // 4 + 128 + 64 + 96 + 128
            elif self.latent_downsample == 8:
                if self.fixed_latent_size:
                    feature_num = in_channels // 4 + 128 + 64 + 96 + 128
                else:
                    feature_num = in_channels + 128 + 64 + 96 + 128
            else:
                raise NotImplementedError(f"Unsupported latent_downsample value: {self.cfg.latent_downsample}")
        elif self.resizeconv_upsampler:
            feature_num = self.feature_upsampler_channels
        else:
            if self.light_dpt_feature:
                for config in model_configs.values():
                    config['out_channels'] = [c // 2 for c in config['out_channels']]
            features = model_configs[vit_type]["features"]
            if self.latent_gs and not self.latent_dpt_upsampler_no_concat:
                features *= 4
            feature_num = features

        return feature_num, model_configs

    def get_pt_in_channels(self):
        feature_upsampler_channels, _ = self.get_feature_upsampler_channels()
        in_channels = 3 + feature_upsampler_channels + self.gaussian_regressor_channels + 1
        if self.latent_gs:
            # image unshuffle
            if self.fixed_latent_size:
                in_channels = in_channels - 3 + 3 * (4 ** 2)
            else:
                in_channels = in_channels - 3 + 3 * (self.latent_downsample ** 2)
        return in_channels

    def get_gaussian_param_num(self):
        # predict gaussian parameters: scale, q, sh, offset, opacity
        # d_in: (scale, q, sh)
        # base params (scale, q, sh, opacity) plus the 2D pixel offset (via get_position_param_num)
        init_gaussian_param_num = super().get_gaussian_param_num()
        if self.no_pixel_offset:
            init_gaussian_param_num -= 2
        # multiple gaussians per latent
        if self.init_gaussian_multiple > 1:
            # we use the point cloud unprojected from higher resolution depth map as center
            # assert self.cfg.gaussian_adapter.init_rotation_identity
            assert self.latent_gs
            init_gaussian_param_num *= self.init_gaussian_multiple
        return init_gaussian_param_num

    def get_position_param_num(self) -> int:
        # resplat encodes position as a 2D pixel offset added to a per-pixel depth, not an explicit 3D point.
        return 2

    def get_init_gaussian_multiple(self) -> int:
        return self.init_gaussian_multiple

    def get_sh_d(self):
        sh_d = (self.gaussian_adapter.sh_degree + 1) ** 2
        return sh_d


class ResplatInitializer(LearnedInitializer[ResplatInitializerCfg]):
    def __init__(self, cfg: ResplatInitializerCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = self._get_depth_predictor(cfg)

        if self.cfg.train_depth_only:
            return

        feature_upsampler_channels, model_configs = self.cfg.get_feature_upsampler_channels()

        if self.cfg.latent_gs and not self.cfg.latent_dpt_upsampler:
            # No need to create a module — this config only computes channels
            pass
        elif self.cfg.resizeconv_upsampler:
            self.feature_upsampler = ResizeConvFeatureUpsampler(
                num_scales=cfg.num_scales,
                lowest_feature_resolution=cfg.lowest_feature_resolution,
                out_channels=self.cfg.feature_upsampler_channels,
                vit_type=self.cfg.monodepth_vit_type,
            )

        else:
            self.feature_upsampler = DPTHead(
                **model_configs[cfg.monodepth_vit_type],
                downsample_factor=cfg.upsample_factor,
                return_feature=True,
                num_scales=cfg.num_scales,
                latent_downsample=self.cfg.latent_downsample if self.cfg.latent_gs else None,
                latent_feature_no_concat=self.cfg.latent_dpt_upsampler_no_concat,
            )

        # gaussians adapter (can be removed)
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        if self.cfg.latent_gs:
            # image unshuffle
            if self.cfg.fixed_latent_size:
                # fixed patch size 4
                in_channels = in_channels - 3 + 3 * (4 ** 2)
            else:
                in_channels = in_channels - 3 + 3 * (self.cfg.latent_downsample ** 2)

        # unet gaussian regressor
        if self.cfg.unet_gaussian_regressor:
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            if self.cfg.color_large_unet:
                unet_channel_mult = [1, 2, 4, 4, 4]
            else:
                unet_channel_mult = [1, 1, 1, 1, 1]
            unet_attn_resolutions = [16]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,  # self.unet_per_scale_blocks,
                    # attention_resolutions=[8, 4, 2],
                    attention_resolutions=unet_attn_resolutions,
                    # channel_mult=[1, 1, 1, 1],
                    channel_mult=unet_channel_mult,
                    num_head_channels=32 if self.cfg.gaussian_regressor_channels >= 32 else 16,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        elif self.cfg.resnet_gaussian_regressor:
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                BasicBlock(channels, channels),
                BasicBlock(channels, channels),
            ]

        else:
            # conv regressor
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, 1, 1),
            ]

        self.gaussian_regressor = nn.Sequential(*modules)

        init_gaussian_param_num = self.cfg.get_gaussian_param_num()

        # gaussian head input channels
        # concat(img, features, regressor_out, match_prob)
        in_channels = self.cfg.get_pt_in_channels()

        if self.cfg.pt_head:
            channels = self.cfg.gaussian_regressor_channels
            if self.cfg.pt_head_channels is not None:
                channels = self.cfg.pt_head_channels
            self.proj = nn.Linear(in_channels, channels)

            self.pt = PlainPointTransformer(channels, self.cfg.knn_samples,
                                            post_norm=self.cfg.post_norm,
                                            no_rpe=self.cfg.no_rpe,
                                            no_attn=self.cfg.no_knn_attn,
                                            num_blocks=self.cfg.num_blocks,
                                            num_heads=self.cfg.pt_heads,
                                            attn_proj_channels=self.cfg.attn_proj_channels,
                                            use_checkpointing=self.cfg.use_checkpointing,
                                            init_use_checkpointing=self.cfg.init_use_checkpointing,
                                            with_mv_attn=self.cfg.init_pt_with_mv_attn,
                                            with_mv_attn_lowres=self.cfg.init_pt_with_mv_attn_lowres,
                                            )

            out_channels = channels


            if self.cfg.pt_head_concat_img:
                # concat to the initial image and features
                out_channels = out_channels + 3

                if self.cfg.latent_gs:
                    # pixel unshuffle the full image to the latent resolution
                    out_channels = out_channels - 3 + 3 * (self.cfg.latent_downsample ** 2)

            self.gaussian_head = nn.Sequential(
                nn.Linear(out_channels, init_gaussian_param_num),
                nn.GELU(),
                nn.Linear(init_gaussian_param_num, init_gaussian_param_num)
            )

            # random initialize rotations: first part
            # 4
            num_rotation_params = 4 * self.cfg.init_gaussian_multiple

            # zero init other remaining params
            # scale, opacity, offset, sh
            # 4 + 1 + 1 + 3 * 16 = 54
            nn.init.zeros_(self.gaussian_head[-1].weight[num_rotation_params:])
            nn.init.zeros_(self.gaussian_head[-1].bias[num_rotation_params:])

        else:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, init_gaussian_param_num,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(init_gaussian_param_num,
                          init_gaussian_param_num, 3, 1, 1, padding_mode='replicate')
            )

            # random initialize rotations: first part
            # 4
            num_rotation_params = 4 * self.cfg.init_gaussian_multiple

            # zero init other remaining params
            # scale, opacity, offset, sh
            # 3 + 1 + 2 + 3 * 16 = 54
            nn.init.zeros_(self.gaussian_head[-1].weight[num_rotation_params:])
            nn.init.zeros_(self.gaussian_head[-1].bias[num_rotation_params:])

        self.test_save_every: FrequencyScheduler | None = None  # a class to save intermediate results during testing, will be set by the ModelWrraper

    def _get_depth_predictor(self, cfg):
        return MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            num_depth_candidates=cfg.num_depth_candidates,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
            sample_log_depth=self.cfg.sample_log_depth,
            bilinear_upsample_depth=self.cfg.bilinear_upsample_depth,
            no_upsample_depth=self.cfg.no_upsample_depth,
            use_amp=self.cfg.use_amp,
            return_raw_mono_features=not self.cfg.latent_dpt_upsampler,
            use_checkpointing=self.cfg.use_checkpointing,
        )

    def forward(
            self,
            context: BatchedViews,
            visualization_dump: Optional[dict] = None,
            **kwargs
    ) -> InitializerOutput:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        if self.cfg.depth_pred_half_res:
            half_img = rearrange(context["image"], "b v c h w -> (b v) c h w")
            half_img = F.interpolate(half_img, scale_factor=0.5, mode='bilinear', align_corners=True)
            half_img = rearrange(half_img, "(b v) c h w -> b v c h w", b=b, v=v)

            results_dict = self.depth_predictor(
                half_img,
                attn_splits_list=[2],
                min_depth=1. / context["far"],
                max_depth=1. / context["near"],
                intrinsics=context["intrinsics"],
                extrinsics=context["extrinsics"],
                nn_matrix=cameras_dist_index,
            )

            # upsample depth to the original resolution
            for key in results_dict.keys():
                # NOTE: no need to upsample depth since depth later is in the low resolution
                if key != 'depth_preds':
                    for i in range(len(results_dict[key])):
                        results_dict[key][i] = F.interpolate(results_dict[key][i], scale_factor=2, mode='bilinear',
                                                             align_corners=True)

            # depthsplat: upsample depth to the original resolution
            if not self.cfg.latent_gs:
                for i in range(len(results_dict['depth_preds'])):
                    results_dict['depth_preds'][i] = F.interpolate(results_dict['depth_preds'][i], scale_factor=2,
                                                                   mode='bilinear', align_corners=True)

        else:
            results_dict = self.depth_predictor(
                context["image"],
                attn_splits_list=[2],
                min_depth=1. / context["far"],
                max_depth=1. / context["near"],
                intrinsics=context["intrinsics"],
                extrinsics=context["extrinsics"],
                nn_matrix=cameras_dist_index,
            )

        if self.cfg.use_gt_depth:
            # directly use gt depth as gaussian centers instead of learning them
            # to understand the bottleneck of the model
            assert 'depth' in context
            depth_preds = [context['depth']]
        else:
            # list of [B, V, H, W], with all the intermediate depths
            depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        gaussian_scale_depth = None

        # features [BV, C, H, W]
        if self.cfg.latent_gs and not self.cfg.latent_dpt_upsampler:
            # concat all features
            assert self.cfg.num_scales == 1

            # use pixelshuffle and pixelunshuffle to align all feature resolutions
            # first resize the mono features to 1/16
            mono_features = [F.interpolate(x, size=(h // 16, w // 16), mode='bilinear', align_corners=True) for x in
                             results_dict['raw_mono_features']]
            if self.cfg.fixed_latent_size:
                scale_factor = 4
                mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 16 * 4

                if self.cfg.latent_downsample == 8:
                    mono_features = F.interpolate(mono_features, scale_factor=0.5, mode='bilinear', align_corners=True)
            else:
                if self.cfg.latent_downsample == 4:
                    scale_factor = 4
                    mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                    mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 16 * 4
                elif self.cfg.latent_downsample == 2:
                    scale_factor = 8
                    mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                    mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 64 * 4
                elif self.cfg.latent_downsample == 8:
                    scale_factor = 2
                    mono_features = [F.pixel_shuffle(x, upscale_factor=scale_factor) for x in mono_features]
                    mono_features = torch.cat(mono_features, dim=1)  # channel: 384 / 4 * 4
                else:
                    raise NotImplementedError

            cnn_features = results_dict["features_cnn_all_scales"][::-1]

            if self.cfg.latent_downsample == 2:
                # use pixel shuffle to save channels
                # 1/2, 1/2, 1/4
                cnn_features[2] = F.pixel_shuffle(cnn_features[2], upscale_factor=2)
                # 64 + 96 + 128 // 4
                cnn_features = torch.cat(cnn_features, dim=1)

                # 128 // 4
                mv_features = results_dict["features_mv"][0]
                mv_features = F.pixel_shuffle(mv_features, upscale_factor=2)
            else:
                # resize all cnn features to the latent resolution
                target_h, target_w = h // self.cfg.latent_downsample, w // self.cfg.latent_downsample
                for i in range(len(cnn_features)):
                    cnn_features[i] = F.interpolate(cnn_features[i], size=(target_h, target_w), mode='bilinear',
                                                    align_corners=True)
                cnn_features = torch.cat(cnn_features, dim=1)

                mv_features = results_dict["features_mv"][0]

                if mv_features.shape[-2] != target_h or mv_features.shape[-1] != target_w:
                    mv_features = F.interpolate(mv_features, size=(target_h, target_w), mode='bilinear',
                                                align_corners=True)

            features = torch.cat((mono_features, cnn_features, mv_features), dim=1)
        elif self.cfg.resizeconv_upsampler:
            features = self.feature_upsampler(results_dict["features_cnn"],
                                              results_dict["features_mv"],
                                              results_dict["features_mono"],
                                              )

        else:
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
                features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                                  cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                                  mv_features=results_dict["features_mv"][
                                                      0] if self.cfg.num_scales == 1 else results_dict["features_mv"][
                                                      ::-1]
                                                  )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]

        if not self.cfg.latent_gs:
            match_prob = F.interpolate(
                match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        if self.cfg.latent_gs:
            img_unshuffle = rearrange(context["image"], "b v c h w -> (b v) c h w")
            if self.cfg.fixed_latent_size:
                if self.cfg.latent_downsample == 8:
                    img_unshuffle = F.interpolate(img_unshuffle, scale_factor=0.5, mode='area')

                img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=4)
            else:
                img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=self.cfg.latent_downsample)
            # depth is in the full resolution, downsample to latent depth
            if self.cfg.depth_pred_half_res:
                latent_depth = F.interpolate(depth, scale_factor=1. / (self.cfg.latent_downsample // 2),
                                             mode='bilinear', align_corners=True)
            else:
                if self.cfg.no_upsample_depth:
                    assert self.cfg.latent_downsample == 8 or self.cfg.latent_downsample == 4
                    if self.cfg.latent_downsample == 8:
                        latent_depth = depth
                    else:
                        # 1/8 depth to 1/4
                        latent_depth = F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    if self.cfg.avgpool_depth:
                        latent_depth = F.avg_pool2d(depth, kernel_size=self.cfg.latent_downsample,
                                                    stride=self.cfg.latent_downsample)
                    elif self.cfg.nearest_down_depth:
                        latent_depth = F.interpolate(depth, scale_factor=1. / self.cfg.latent_downsample,
                                                     mode='nearest')
                    else:
                        latent_depth = F.interpolate(depth, scale_factor=1. / self.cfg.latent_downsample,
                                                     mode='bilinear', align_corners=True)

            if match_prob.shape[-2:] != latent_depth.shape[-2]:
                match_prob = F.interpolate(
                    match_prob, size=latent_depth.shape[-2:], mode='nearest')

            concat = torch.cat((
                img_unshuffle,
                rearrange(latent_depth, "b v h w -> (b v) () h w"),
                match_prob,
                features,
            ), dim=1)
        else:
            concat = torch.cat((
                rearrange(context["image"], "b v c h w -> (b v) c h w"),
                rearrange(depth, "b v h w -> (b v) () h w"),
                match_prob,
                features,
            ), dim=1)

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            out = self.gaussian_regressor(concat)

        if self.cfg.latent_gs:
            concat = [out, img_unshuffle, features, match_prob]
        else:
            concat = [out,
                      rearrange(context["image"],
                                "b v c h w -> (b v) c h w"),
                      features,
                      match_prob]

        out = torch.cat(concat, dim=1)

        # [BV, C, H, W]
        condition_features = out

        init_scales = None

        if self.cfg.pt_head:
            if self.cfg.latent_gs:
                h, w = latent_depth.shape[-2:]
            else:
                h, w = depth.shape[-2:]
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
                tmp_feature = self.proj(rearrange(out, "bv c h w -> (bv h w) c"))
            # get point cloud
            xy_ray, _ = sample_image_grid((h, w), out.device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

            # [B, V, H*W, 1, 2]
            tmp_coords = xy_ray.unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1, 1)

            # [B, V, H*W, 1, 1]
            if self.cfg.latent_gs:
                tmp_depth = rearrange(latent_depth, "b v h w -> b v (h w) () ()")
            else:
                tmp_depth = rearrange(depth, "b v h w -> b v (h w) () ()")

            # [B, V, 1, 1, 4, 4]
            tmp_extrinsics = context["extrinsics"].unsqueeze(2).unsqueeze(2)
            # [B, V, 1, 1, 3, 3]
            tmp_intrinsics = context["intrinsics"].unsqueeze(2).unsqueeze(2)

            # [B, V, H*W, 1, 3]
            origins, directions = get_world_rays(tmp_coords, tmp_extrinsics, tmp_intrinsics)
            point_cloud = origins + directions * tmp_depth

            # Create offset directly on device to avoid CPU-GPU transfer
            offset = torch.arange(1, b + 1, device=depth.device, dtype=torch.long) * (v * h * w)

            point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")

            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
                if self.cfg.add_pt_residual:
                    out = tmp_feature + self.pt((point_cloud, tmp_feature, offset), b=b, v=v, h=h, w=w)
                else:
                    out = self.pt((point_cloud, tmp_feature, offset), b=b, v=v, h=h, w=w)

                condition_features = rearrange(out, "(bv h w) c -> bv c h w", h=h, w=w)


            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_head_amp, dtype=torch.bfloat16):
                if self.cfg.pt_head_concat_img:
                    if self.cfg.latent_gs:
                        # pixel unshuffle image
                        img_unshuffle = rearrange(context["image"], "b v c h w -> (b v) c h w")
                        img_unshuffle = F.pixel_unshuffle(img_unshuffle, downscale_factor=self.cfg.latent_downsample)
                        img_unshuffle = rearrange(img_unshuffle, "(b v) c h w -> (b v h w) c", b=b, v=v)

                        out = torch.cat((out, img_unshuffle), dim=-1)

                if self.cfg.pt_head_conv:
                    out = rearrange(out, "(b v h w) c -> (b v) c h w", b=b, v=v, h=h, w=w)

                out = self.gaussian_head(out)

                if self.cfg.pt_head_conv:
                    out = rearrange(out, "(b v) c h w -> (b v h w) c", b=b, v=v)

            point_cloud = rearrange(point_cloud, "(b v h w) c -> b v (h w) () () c", b=b, v=v, h=h, w=w)

            gaussians = rearrange(out, "(b v h w) c -> (b v) c h w", b=b, h=h, w=w)

        else:
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
                gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        # [BV, C, H, W]
        gaussians = gaussians.float()

        if self.cfg.latent_gs:
            if self.cfg.init_gaussian_multiple > 1:
                # hard coded for now
                if self.cfg.init_gaussian_multiple == 4:
                    if self.cfg.latent_downsample == 4:
                        # resize full resolution depth
                        depths = F.interpolate(depth, scale_factor=0.5, mode='bilinear', align_corners=True)
                    elif self.cfg.latent_downsample == 8:
                        depths = F.interpolate(depth, scale_factor=0.25, mode='bilinear', align_corners=True)
                    elif self.cfg.latent_downsample == 2:
                        depths = depth
                    else:
                        raise NotImplementedError
                elif self.cfg.init_gaussian_multiple == 16:
                    if self.cfg.latent_downsample == 4:
                        depths = depth
                    elif self.cfg.latent_downsample == 8:
                        depths = F.interpolate(depth, scale_factor=0.5, mode='bilinear', align_corners=True)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError

                depths = rearrange(depths, "b v h w -> b v (h w) () ()")
            else:
                depths = rearrange(latent_depth, "b v h w -> b v (h w) () ()")
        else:
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        assert len(depth_preds) == 1, "num_scales must be 1; multi-scale depth supervision is not supported"

        # [B, V, H*W, C]
        repeat = self.cfg.init_gaussian_multiple
        num_sh = self.gaussian_adapter.d_sh

        if self.cfg.no_pixel_offset:
            rotations_unnorm, scales, opacities_raw, sh = raw_gaussians.split(
                [4 * repeat, 3 * repeat, 1 * repeat, 3 * num_sh * repeat],
                dim=-1,
            )
        else:
            rotations_unnorm, scales, opacities_raw, offset, sh = raw_gaussians.split(
                [4 * repeat, 3 * repeat, 1 * repeat, 2 * repeat, 3 * num_sh * repeat],
                dim=-1,
            )

        latent_h, latent_w = gaussians.shape[-2:]

        if repeat > 1:
            # reshape all the gaussian parameters
            if True or self.cfg.latent_new_reshape:
                # this works
                r = int(np.sqrt(repeat))
                rotations_unnorm = rearrange(rotations_unnorm, "b v (h w) (c x y) -> b v (h x w y) c",
                                             h=latent_h, w=latent_w, x=r, y=r)
                scales = rearrange(scales, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r,
                                   y=r)
                opacities_raw = rearrange(opacities_raw, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h,
                                          w=latent_w, x=r, y=r)
                offset = rearrange(offset, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r,
                                   y=r)
                sh = rearrange(sh, "b v (h w) (c x y) -> b v (h x w y) c", h=latent_h, w=latent_w, x=r, y=r)
            else:
                # doesn't work
                rotations_unnorm = rearrange(rotations_unnorm, "b v hw (k c) -> b v (hw k) c", k=repeat)
                scales = rearrange(scales, "b v hw (k c) -> b v (hw k) c", k=repeat)
                opacities_raw = rearrange(opacities_raw, "b v hw (k c) -> b v (hw k) c", k=repeat)
                offset = rearrange(offset, "b v hw (k c) -> b v (hw k) c", k=repeat)
                sh = rearrange(sh, "b v hw (k c) -> b v (hw k) c", k=repeat)

        opacities = opacities_raw.sigmoid()  # [B, V, H*W*K, 1]

        if self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 2 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 16:
            scale_factor = 4
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 4:
            scale_factor = 2
        elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 16:
            scale_factor = 4
        else:
            scale_factor = 1

        h, w = latent_h * scale_factor, latent_w * scale_factor

        # unproject depth
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

        if self.cfg.no_pixel_offset:
            offset_xy = torch.ones_like(raw_gaussians[..., :2]).unsqueeze(-2).to(
                raw_gaussians.device) * 0.5  # [B, V, H*W, 1, 2]
        else:
            offset_xy = offset.sigmoid().unsqueeze(-2)  # [B, V, H*W, 1, 2]

        pixel_size = 1 / \
                     torch.tensor((w, h), dtype=torch.float32, device=device)
        # [H*W, 1, 2]
        if self.cfg.deform_sample_depth and not self.cfg.deform_sample_depth_debug:
            # (offset_xy - 0.5) in -0.5 to 0.5, without multiplying by pixel size such that the points can move in the image space
            xy_ray = (xy_ray + (offset_xy - 0.5)).clamp(min=0., max=1.)
        else:
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        if self.cfg.deform_sample_depth:
            # use low-res xy_ray to sample full-res depth

            sample_grid = rearrange(xy_ray, "b v (h w) c xy -> (b v) h w (c xy)", h=h, w=w)  # in [0, 1]
            # to [-1, 1]
            sample_grid = 2 * (sample_grid - 0.5)  # [BV, h, w, 2]

            fullres_depth = rearrange(depth, "b v h w -> (b v) () h w")  # [BV, 1, H, W]
            sampled_depth = F.grid_sample(fullres_depth, sample_grid, mode='bilinear', align_corners=True,
                                          padding_mode="border")  # [BV, 1, h, w]
            # reshape
            depths = rearrange(sampled_depth, "(b v) () h w -> b v (h w) () ()", b=b, v=v, h=h, w=w)

        if self.cfg.latent_gs:
            sh_input_images = rearrange(context["image"], "b v c h w -> (b v) c h w")
            if self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 4:
                sh_input_images = F.interpolate(sh_input_images, scale_factor=0.5, mode='area')
            elif self.cfg.latent_downsample == 4 and self.cfg.init_gaussian_multiple == 16:
                pass
            elif self.cfg.latent_downsample == 2 and self.cfg.init_gaussian_multiple == 4:
                pass
            elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 4:
                sh_input_images = F.interpolate(sh_input_images, scale_factor=0.25, mode='area')
            elif self.cfg.latent_downsample == 8 and self.cfg.init_gaussian_multiple == 16:
                sh_input_images = F.interpolate(sh_input_images, scale_factor=0.5, mode='area')
            else:
                sh_input_images = F.interpolate(sh_input_images, scale_factor=1. / self.cfg.latent_downsample,
                                                mode='area')

            sh_input_images = rearrange(sh_input_images, "(b v) c h w -> b v c h w", b=b, v=v)

        else:
            sh_input_images = context["image"]

        assert len(depth_preds) == 1, "num_scales must be 1; multi-scale depth supervision is not supported"

        # build gaussians
        # scale
        scales = torch.clamp(F.softplus(scales - self.cfg.gaussian_adapter.exp_scale_bias),
                             min=self.cfg.gaussian_adapter.clamp_min_scale,
                             max=self.cfg.gaussian_adapter.gaussian_scale_max
                             )

        # Convert rotations to world-space
        c2w_rotations = context["extrinsics"][..., :3, :3].unsqueeze(2)  # [B, V, 1, 3, 3]
        # Here quaternions follow the xyzw format (scalar last)
        rotations = rotate_quats(c2w_rotations, rotations_unnorm)
        if self.cfg.camera_frame_quats:
            # q_c2w * q_raw: world-frame, but keeps the raw camera-frame norm and sign, so the ReSplat
            # optimizer can recover its camera-frame input exactly (the rasterizer normalizes anyway).
            q_c2w = rotation_matrix_to_quaternion_xyzw(c2w_rotations)  # [B, V, 1, 4]
            rotations_unnorm = quaternion_multiply_xyzw(q_c2w, rotations_unnorm)
        else:
            rotations_unnorm = rotations.clone()
        # Create world-space covariance matrices.
        covariances = build_covariance(scale=scales, rotation_xyzw=rotations)  # [B, V, H*W, 3, 3]

        # means
        # [B, V, H*W, 1, 2]
        origins, directions = get_world_rays(xy_ray,
                                             context["extrinsics"].unsqueeze(2).unsqueeze(2),
                                             context["intrinsics"].unsqueeze(2).unsqueeze(2))
        means = origins + directions * depths

        # sh: [B, V, HW, 3, SH]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3).clone()

        # [B, V, H*W, 3]
        sh_input_images = rearrange(sh_input_images, "b v c h w -> b v (h w) c")
        # init sh with input images
        sh[..., 0] = sh[..., 0] + RGB2SH(sh_input_images)

        gaussians = Gaussians(
            means=rearrange(means, "b v r spp xyz -> b (v r spp) xyz"),
            covariances=rearrange(covariances, "b v r i j -> b (v r) i j"),
            harmonics=rearrange(sh, "b v r c d_sh -> b (v r) c d_sh"),
            opacities=rearrange(opacities, "b v r spp -> b (v r spp)"),
            scales=rearrange(scales, "b v r xyz -> b (v r) xyz"),
            rotations=rearrange(rotations, "b v r wxyz -> b (v r) wxyz"),
            rotations_unnorm=rearrange(rotations_unnorm, "b v r wxyz -> b (v r) wxyz")
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = depth_preds[-1]

            if self.cfg.return_lowres_depth:
                assert latent_depth is not None
                depths = latent_depth
            else:
                if depths.shape[-2:] != context["image"].shape[-2:]:
                    # depths can be at lower resolution since we predict latent
                    depths = F.interpolate(
                        depths, size=context["image"].shape[-2:], mode='bilinear', align_corners=True)

            return InitializerOutput(
                gaussians=gaussians,
                depths=depths,
                features=condition_features
            )
        else:

            return InitializerOutput(
                gaussians=gaussians,
                features=condition_features
            )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            patch_size = self.cfg.shim_patch_size
            if isinstance(self.cfg.shim_patch_size, int):
                patch_size = patch_size * self.cfg.downscale_factor
            else:
                patch_size = [p * self.cfg.downscale_factor for p in patch_size]
            batch = apply_patch_shim(
                batch,
                patch_size=patch_size,
            )

            return batch

        return data_shim

    @staticmethod
    def update_gt_depth_range(batch):
        assert "depth" in batch["context"]
        batch["context"]["near"] = batch["context"]["depth"].min(dim=3)[0].min(dim=2)[0].clamp(min=0.01)
        batch["context"]["far"] = batch["context"]["depth"].max(dim=3)[0].max(dim=2)[0].clamp(max=1000.)
        batch["target"]["near"] = batch["target"]["depth"].min(dim=3)[0].min(dim=2)[0].clamp(min=0.01)
        batch["target"]["far"] = batch["target"]["depth"].max(dim=3)[0].max(dim=2)[0].clamp(max=1000.)

    def predict_scale(self, batch):
        context = batch["context"]
        # [B, V, H, W]
        init_depth = self.encoder.scale_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
        )['depth_preds'][-1]
        if not self.encoder.cfg.no_pred_depth_range:
            new_near = init_depth.min(dim=3)[0].min(dim=2)[0].clamp(min=0.1)  # [B, V]
            new_far = init_depth.max(dim=3)[0].max(dim=2)[0].clamp(max=100.)

            batch["context"]["near"] = new_near
            batch["context"]["far"] = new_far

            batch["target"]["near"] = new_near.min(dim=1, keepdim=True)[0].repeat(1,
                                                                                  batch["target"]["near"].shape[1])
            batch["target"]["far"] = new_far.max(dim=1, keepdim=True)[0].repeat(1, batch["target"]["near"].shape[1])
        if self.encoder.cfg.norm_by_points:
            b, v, h, w = init_depth.shape
            # get point cloud
            xy_ray, _ = sample_image_grid((h, w), batch["context"]["image"].device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")

            # [B, V, H*W, 1, 2]
            tmp_coords = xy_ray.unsqueeze(0).unsqueeze(0).repeat(b, v, 1, 1, 1)

            # [B, V, H*W, 1, 1]
            tmp_depth = rearrange(init_depth, "b v h w -> b v (h w) () ()")

            # [B, V, 1, 1, 4, 4]
            tmp_extrinsics = context["extrinsics"].unsqueeze(2).unsqueeze(2)
            # [B, V, 1, 1, 3, 3]
            tmp_intrinsics = context["intrinsics"].unsqueeze(2).unsqueeze(2)

            # [B, V, H*W, 1, 3]
            origins, directions = get_world_rays(tmp_coords, tmp_extrinsics, tmp_intrinsics)
            point_cloud = origins + directions * tmp_depth

            point_cloud = rearrange(point_cloud, "b v h w c -> b (v h w) c")

            point_dist = point_cloud.norm(dim=-1).mean(dim=-1)  # [B]

            norm_factor = point_dist.clamp(min=1e-6)

            # normalize near, far and extrinsics
            batch["context"]["near"] = batch["context"]["near"] / norm_factor.view(b, 1)
            batch["context"]["far"] = batch["context"]["far"] / norm_factor.view(b, 1)

            batch["target"]["near"] = batch["target"]["near"] / norm_factor.view(b, 1)
            batch["target"]["far"] = batch["target"]["far"] / norm_factor.view(b, 1)

            batch["context"]["extrinsics"][:, :, :3, -1] /= norm_factor.view(b, 1, 1)
            batch["target"]["extrinsics"][:, :, :3, -1] /= norm_factor.view(b, 1, 1)

    def eval_preprocessing(self, batch, train_cfg):
        # use gt depth range instead of a fixed one
        if train_cfg.use_gt_depth_range:
            self.update_gt_depth_range(batch)

        # use a pretrained depth model to predict scale
        if self.cfg.predict_scale:
            self.predict_scale(batch)
