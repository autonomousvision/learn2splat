"""The learned optimizer: a kNN point transformer that refines a scene's Gaussians.

Holds `KnnBasedOptimizerCfg` (the architecture settings, selected by the `knn_based`,
`learn2splat` and `resplat_v1`/`resplat_v2` config groups), `KnnBasedOptimizerState` (the
per-Gaussian latent state carried across iterations) and `KnnBasedOptimizer`, whose own
docstring walks through one optimization step. The base classes it builds on, and the
pipeline types it passes in and out, are in `optimizer.py`.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
from einops import rearrange
from torch import nn, Tensor

from learn2splat.dataset.data_types import BatchedViews
from learn2splat.misc.general_utils import SkipBatchException
from learn2splat.misc.io import FrequencyScheduler
from learn2splat.model.decoder.decoder import Decoder
from learn2splat.model.backbones.layer import ResNetFeatureWarpper
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import build_covariance
from learn2splat.scene_trainer.common.gaussians import quaternion_multiply_xyzw, rotation_matrix_to_quaternion_xyzw
from learn2splat.scene_trainer.initializer import InitializerCfg, InitializerColmapCfg, InitializerDepthCfg, \
    InitializerEdgsCfg, InitializerRandomCfg, InitializerPointcloudCfg
from learn2splat.scene_trainer.initializer import InitializerPlyCfg
from learn2splat.scene_trainer.initializer.initializer_resplat import ResplatInitializerCfg
from learn2splat.scene_trainer.optimizer.optimizer import OptimizerInput, LearnedOptimizer, OptimizerOutput, OptimizerState, \
    OptimizerPreviousOutput, OptimizerCfg
from learn2splat.scene_trainer.optimizer.optimizer_utils import unpack_gaussians

try:
    from learn2splat.model.backbones.point_transformer.layer import (PlainPointTransformer,
                                                               PointLinearWrapper,
                                                               MultViewLowresAttn)
except:
    pass

from learn2splat.scene_trainer.optimizer.layer import CustomGroupNorm, AdamInputSmoothing, SlicedG3RNorm, AdamState
from learn2splat.scene_trainer.initializer.initializer import InitializerOutput
from learn2splat.scene_trainer.optimizer.time_embed import get_embedder, get_ddpm_sinusoidal_time_embedding, TimeEncodingWrapper

from learn2splat.loss.loss_depth_smooth import get_smooth_loss
from learn2splat.scene_trainer.optimizer.optimizer_utils import (
    inner_loss_for_input_gradients,
    chunk_index_iter,
    split_grads,
    get_gaussian_param_slices,
    get_gaussian_param_sizes,
    pack_gaussians,
)

_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Gaussian parameter groups the delta head predicts, paired with their key in
# get_gaussian_param_sizes. Order matches pack_gaussians / split_delta_gaussians.
_GAUSSIAN_GROUPS = [
    ("means", "means"),
    ("scales", "scales"),
    ("rotations", "quats"),
    ("opacities", "opacities"),
    ("shs", "shs"),
]


@dataclass
class KnnBasedOptimizerCfg(OptimizerCfg):
    name: Literal["knn_based", "resplat_v1", "resplat_v2", "l2s"]
    # iterative refine
    no_render_error: bool
    input_error_shallow_resnet_feature: bool
    input_error_resnet_feature_layers: int
    num_basic_blocks: int
    num_blocks: int
    concat_init_state: bool  # always concat init state during updates
    replace_init_state: bool  # always use the init state during updates
    state_channels: int
    block_rmsnorm: bool
    block_layernorm: bool
    pt_qk_norm: bool
    norm_pt_block: bool
    delta_gaussian_multiple: int  # predict more gaussian residuals based on the previous gaussian center
    residual_init_state: bool  # add residual connection in the prediction head to the initial state
    clamp_max_scale: float
    clamp_min_scale: float | int
    clamp_min_raw_scales: float | int
    clamp_max_raw_scales: float | int
    # Bound on |position| for the KNN-input copy ONLY (None = off). See _apply_point_transformer.
    knn_pos_clamp: float | None
    clamp_min_raw_opacities: float | int
    clamp_max_raw_opacities: float | int
    clamp_min_sh0: float | int
    clamp_max_sh0: float | int
    clamp_min_shs: float | int
    clamp_max_shs: float | int
    clamp_shs_soft: bool

    residual_grad_scale: float | int
    attn_proj_channels: int | None
    no_knn_attn: bool
    no_tran_block_norm: bool
    tran_block_act: str | None
    multi_gaussian_scale_smaller: bool
    condition_pt_feature: bool
    same_num_points: bool  # when init_gaussian_multiple > 1, refine directly works on it instead of subsampling points

    knn_samples: int
    max_active_gaussians: int

    # KNN
    use_fused_attn: bool
    knn_idx_update_every: int

    # point transformer
    pt_heads: int

    # inputs
    input_alpha: bool
    input_depth: bool
    input_depth_smooth_error: bool

    # input error
    input_error: bool  # render error as input to the refine head
    input_error_rgb_no_shuffle: bool  # sample single pixel instead of pixel unshuffling
    input_error_add_rgb_feature: bool

    # resnet
    input_error_resnet_feature: bool
    input_error_cache_resnet_feature: bool
    input_error_no_freeze_resnet_feature: bool

    # number of views for render error
    input_error_num_views: int
    input_error_additional_cross_attn: bool
    input_error_num_intermediate_views: int

    # render error with remaining context views
    input_error_remain_context: bool
    input_error_merge_remain_context: bool
    input_error_warp_remain_context: bool
    input_error_random_num_remain_context: bool
    input_error_num_remain_context_test: int

    # render error mv attn
    input_error_mv_attn: bool
    input_error_mv_attn_blocks: int

    # refine camera-frame raw quaternions (released ReSplat convention); needs the initializer's camera_frame_quats
    camera_frame_quats: bool

    # refine global attention
    with_mv_attn: bool
    with_mv_attn_lowres: bool
    no_mv_attn: bool  # remove only the attn
    mv_attn_conv_with_norm: bool  # unet-attn conv with norm
    mv_shuffle_attn: bool  # use pixel shuffle to save computation instead of unet
    mv_attn_with_pos_enc: bool
    shuffle_attn_no_norm: bool
    mv_unimatch_attn: bool

    # input gradients
    input_gradient: bool
    input_gradient_log: bool
    input_gradient_log_clip_deltas: float | int
    input_gradient_scale: float | int
    input_gradient_same_loss: bool  # use the same loss as the gaussian update
    input_gradient_loss_reduction: str
    scale_residual_grads: bool

    # time encoding
    use_time_encoding: bool
    time_encoding_max_steps: int
    # Encoding form for rel_step = t / time_encoding_max_steps: "frequency" (NeRF-style band
    # encoding) or "ddpm" (G3R/QuickSplat sinusoidal). Pair "ddpm" with time_encoding_max_steps set
    # to the horizon (e.g. max_t at train, num_update_steps at eval).
    time_encoding_type: str

    train_global_only: bool

    # random size refine
    # update more for low resolution, less for high
    random_step_with_size: bool

    # amp
    use_amp: bool
    pt_head_amp: bool
    pt_update_amp: bool

    use_checkpointing: bool
    recurrent_use_checkpointing: bool

    # Normalizing input
    input_gradient_normalize: bool
    input_gradient_normalize_type: str
    input_normalize_state: bool
    input_normalize_gaussians: bool

    # State scaling
    predict_state_scale: bool
    predict_state_scale_norm: bool  # whether to normalize the state before scaling

    # Use optimizer without condition features
    init_state_wo_features: bool
    init_state_type: Literal["random", "constant"]
    init_state_scale: float | int

    opt_scales_before_act: bool  # optimize scale before activation (raw -> exp -> scale -> log -> raw)

    # Preprocessing the init gaussians
    scale_initial_opacities: float | int

    update_only_nonzero_grad: bool

    # update learn residual state
    residual_state: bool

    # Update head
    delta_head_layer_num: int
    delta_head_concat_img: bool
    delta_head_act: str | None  # delta_head activation to predict the deltas
    delta_head_final_act: str | None  # final activation in the delta_head

    delta_head_scalar_scale: bool  # predict deltas as scalar * delta / norm(delta)
    delta_head_scalar_scale_act: str  # activation for the scalar scale output

    # Per-parameter-group update head (Feature A)
    delta_head_per_param_heads: bool  # separate heads per param group, each with own normalize+scale
    delta_head_per_param_hidden_dim: int  # hidden dim for per-param heads (SH head gets 2x)
    # Per-parameter scalar scales (Feature B) — requires delta_head_scalar_scale=true
    delta_head_per_param_scales: bool  # per-group scalar scales instead of one global scalar

    # Config from initializer
    sh_d: int | None
    init_sh_d: int | None = None
     # Run a checkpoint trained at full sh_d but with a lower *effective* SH degree:
    # the SH bands above cap_sh_degree are held at zero (DC and any kept bands stay
    # trainable), so the network architecture / weights are unchanged but the model
    # renders with only the lower-degree color. None = use all bands. 0 = DC only.
    cap_sh_degree: int | None = None
    # For initialization from feed forward, gaussians are aligned with pixels.
    init_gaussian_multiple: int | None = None
    latent_downsample: int | None = None

    # KNN backend for the point-transformer neighbour search: "simple_knn" (default;
    # Morton/box-pruned, far faster at large N) or "pointops" (brute force, for comparison).
    # Identical neighbour sets. simple_knn is mandatory.
    knn_backend: str = "simple_knn"

    @property
    def scales_clamp_lims(self) -> tuple:
        """Scale clamp limits in the space scales are refined in.

        With opt_scales_before_act scales are refined in log space, so the raw (log-space)
        limits apply; otherwise the activation-space limits apply.
        """
        if self.opt_scales_before_act:
            return self.clamp_min_raw_scales, self.clamp_max_raw_scales
        return self.clamp_min_scale, self.clamp_max_scale

    def update(self, initializer_cfg: InitializerCfg, source_initializer_cfgs: "dict | None" = None):
        """Set the optimizer's input dimensions (SH coefficients, feature channels, Gaussians per
        point) from the initializer(s) it will refine: the default `initializer_cfg` plus the
        per-source overrides of a mixed dataset."""
        POINT_CLOUD = (InitializerPlyCfg, InitializerColmapCfg, InitializerDepthCfg, InitializerEdgsCfg,
                       InitializerRandomCfg, InitializerPointcloudCfg)

        # General settings (from the default; per-source inits share sh_degree, see SceneTrainerCfg).
        self.init_sh_d = initializer_cfg.get_sh_d()
        if self.sh_d is None:
            self.sh_d = initializer_cfg.get_sh_d()

        all_inits = [initializer_cfg, *(source_initializer_cfgs or {}).values()]
        for i in all_inits:
            if not isinstance(i, (ResplatInitializerCfg, *POINT_CLOUD)):
                raise ValueError(f"Unsupported initializer config type: {type(i)}")
        feature_inits = [i for i in all_inits if isinstance(i, ResplatInitializerCfg)]

        # Point-cloud scenes have no per-pixel alignment, so the optimizer cannot use pixel attributes.
        if any(isinstance(i, POINT_CLOUD) for i in all_inits):
            assert not self.input_error, "The error calculation assumes per pixel gaussians"
            assert not self.delta_head_concat_img
            assert not self.input_alpha

        # Both sides must agree on the frame of rotations_unnorm (camera-frame raw vs. world-frame).
        for i in all_inits:
            init_camera_frame = getattr(i, "camera_frame_quats", False)
            if init_camera_frame != self.camera_frame_quats:
                raise ValueError(
                    f"camera_frame_quats mismatch: optimizer={self.camera_frame_quats}, "
                    f"initializer {i.name}={init_camera_frame}")
        if self.camera_frame_quats and self.any_adc:
            raise ValueError("camera_frame_quats is not supported with densification (needs the per-view Gaussian layout)")

        if feature_inits:
            # The feature branch (state_proj) takes its input width from the first feature init.
            fi = feature_inits[0]
            self.latent_downsample = fi.latent_downsample
            self.init_gaussian_multiple = fi.init_gaussian_multiple
            self.condition_channels = (fi.gaussian_regressor_channels if self.condition_pt_feature
                                       else fi.get_pt_in_channels())
        else:
            # No feature init: every scene synthesizes its state per-Gaussian (get_vector_state).
            self.init_gaussian_multiple = 1
            self.latent_downsample = 1


@dataclass
class KnnBasedOptimizerState(OptimizerState):
    """OptimizerState subclass for KNN-based optimizers.

    Holds the per-Gaussian learned state vectors and adds ADC mutation methods
    (called by adc/base.py helpers) that keep them in sync with clone/split/prune.
    """
    state: torch.Tensor | None = None
    init_state: torch.Tensor | None = None  # scene-start state; None when concat/replace/residual flags are all off
    adam_state: AdamState | None = None
    # How a densified Gaussian gets its latent state, keyed "clone"/"split"/"relocate"/"add" and set
    # at scene start from the <op>_gaussian_state cfg fields (documented on BaseStrategyCfg).
    new_state_policy: dict = field(default_factory=dict)  # per-op; missing keys resolve to "copy"
    new_state_scale: float | int = 1.0  # matches cfg.init_state_scale (float | int)

    def _each_state(self):
        """Yield ``(attr, tensor)`` for each present per-Gaussian state tensor (state, init_state)."""
        for attr in ("state", "init_state"):
            t = getattr(self, attr)
            if t is not None:
                yield attr, t

    def _fresh(self, n: int, like: torch.Tensor, policy: str) -> torch.Tensor:
        """A fresh block of ``n`` state rows shaped like ``like[1:]``: random (the scene-start random
        init) when ``policy == "random"``, else zeros."""
        shape = (n, *like.shape[1:])
        if policy == "random":
            return torch.randn(shape, device=like.device, dtype=like.dtype) * self.new_state_scale
        return torch.zeros(shape, device=like.device, dtype=like.dtype)

    def clone(self, clone_mask: torch.Tensor) -> None:
        # Append a copy of each cloned Gaussian: copy the parent, or a fresh block (zero/random).
        policy = self.new_state_policy.get("clone", "copy")
        for attr, t in self._each_state():
            new = t[clone_mask] if policy == "copy" else self._fresh(int(clone_mask.sum()), t, policy)
            setattr(self, attr, torch.cat([t, new], dim=0))

    def split(self, split_mask: torch.Tensor, num_splits: int) -> None:
        # Canonical 3DGS split layout (matches vanilla.py's state_fn and the Gaussian tensors): drop
        # the split parents, then append num_splits block-repeated parent copies (copy) or a fresh
        # block (zero/random). Block order keeps each row paired with its Gaussian.
        policy = self.new_state_policy.get("split", "copy")
        rest = ~split_mask
        for attr, t in self._each_state():
            repeated = t[split_mask].repeat(num_splits, 1)
            new = repeated if policy == "copy" else self._fresh(repeated.shape[0], t, policy)
            setattr(self, attr, torch.cat([t[rest], new], dim=0))

    def replace(self, from_indices: torch.Tensor, dest_indices: torch.Tensor) -> None:
        # MCMC relocate: dead Gaussians copy the sampled-alive state, or get a fresh block (zero/random).
        policy = self.new_state_policy.get("relocate", "copy")
        for _, t in self._each_state():
            t[dest_indices] = t[from_indices] if policy == "copy" else self._fresh(len(dest_indices), t, policy)

    def prune(self, prune_mask: torch.Tensor) -> None:
        for attr, t in self._each_state():
            setattr(self, attr, t[~prune_mask])

    def add(self, num_new: int) -> None:
        # MCMC add_new: brand-new Gaussians have no parent, so always a fresh block ("copy" -> zeros).
        if num_new <= 0:
            return
        policy = self.new_state_policy.get("add", "copy")
        for attr, t in self._each_state():
            setattr(self, attr, torch.cat([t, self._fresh(num_new, t, policy)], dim=0))


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


def get_activation_cls(activation: Optional[str] = None):
    if activation in ['none', None, 'identity']:
        return nn.Identity
    elif activation == 'tanh':
        return nn.Tanh
    elif activation == "gelu":
        return nn.GELU
    elif activation == 'sigmoid':
        return nn.Sigmoid
    elif activation == 'relu':
        return nn.ReLU
    elif activation == "softplus":
        return nn.Softplus
    elif activation == "abs":
        return Abs
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class KnnBasedOptimizer(LearnedOptimizer[KnnBasedOptimizerCfg]):
    """Learned optimizer that refines a scene's Gaussians with a point-transformer (the core method).

    Replaces hand-tuned gradient descent (Adam + heuristics) with a network that predicts, at each
    iteration, how every Gaussian should change. The scene is treated as a point cloud (one point per
    Gaussian) and the network reasons over local neighborhoods via KNN attention, so an update to one
    Gaussian is informed by its spatial neighbors.

    Each Gaussian carries a learned latent state vector alongside its parameters (means, scales,
    rotations, opacity, SH). The state is what the network reads and writes across iterations; the
    Gaussian parameters are updated by adding predicted deltas.

    One optimization step (`_apply_step`, run per iteration by `_forward_impl`):
      1. Input signal (`prepare_input_signal`): render the context views and measure how well they match
         the ground-truth inputs. The signal is the per-Gaussian rendering gradients
         (`_calc_input_gradients` — a backprop of the reconstruction loss, used only as a feature, never
         to update weights) and/or the render error in feature space. This tells the network "what is
         wrong" with the current Gaussians.
      2. Update (point transformer): KNN attention over the point cloud mixes each Gaussian's state with
         its neighbors', conditioned on the input signal, producing an updated per-Gaussian state.
      3. Delta head: maps the updated state to per-parameter deltas, which are added to the Gaussians
         (with clamping/activation handling per parameter group).
      4. Optional adaptive density control (clone/split/prune), keeping the state in sync with the
         changing Gaussian count.

    `optimizer_preprocessing` runs once at scene start to seed the latent state from the initializer's
    Gaussians. This class implements both ReSplat and Learn2Splat through different settings in
    `KnnBasedOptimizerCfg`: ReSplat (`resplat_v1.yaml`) drives the update with feature-space render
    error and multi-view attention, while Learn2Splat (`learn2splat.yaml`) drives it with normalized
    per-Gaussian gradients. The many `_build_*` / `_calc_*` helpers wire up these settings.
    """

    OPTIMIZER_NAME = "knn_based"

    def __init__(self, cfg: KnnBasedOptimizerCfg, save_every: Optional[FrequencyScheduler] = None) -> None:
        assert cfg.name == self.OPTIMIZER_NAME, f"Expected optimizer name {self.OPTIMIZER_NAME}, got {cfg.name}"

        super().__init__(cfg, save_every)

        if self.cfg.residual_state:
            assert not self.cfg.residual_init_state

        # State channel
        self.state_channels = self.cfg.state_channels

        # time embedder. Both types encode rel_step = t / time_encoding_max_steps (see
        # TimeEncodingWrapper) and differ only in the encoding form.
        if self.cfg.use_time_encoding:
            if self.cfg.time_encoding_type == "ddpm":
                # Match the frequency encoder's output dim so the network is identical across types.
                _, self.time_embedding_dim = get_embedder(multires=6)
                self.time_encoder_fn = lambda x: get_ddpm_sinusoidal_time_embedding(x, self.time_embedding_dim).squeeze(0)
            else:  # "frequency": NeRF-style band encoding
                self.time_encoder_fn, self.time_embedding_dim = get_embedder(multires=6)
        else:
            self.time_encoder_fn = None
            self.time_embedding_dim = 0

        # state_proj
        if not self.cfg.init_state_wo_features:
            self.state_proj = nn.Conv2d(self.cfg.condition_channels, self.state_channels, 1)

        channels, in_channels, update_gaussian_param_num, out_channels, error_features_channels = (
            self._define_channels())
        self.error_features_channels = error_features_channels
        self.gaussian_param_num = out_channels

        if self.cfg.input_error:

            self.error_feature_extractor = self.get_input_error_feature_extractor()
            if self.cfg.input_error_add_rgb_feature:
                if self.cfg.init_gaussian_multiple == 4:  # re10k
                    self.rgb_error_proj = nn.Sequential(
                        nn.Linear(3, error_features_channels),
                        nn.LayerNorm(error_features_channels)
                    )
                else:
                    self.rgb_error_proj = nn.Sequential(
                        nn.Linear(3 * self.cfg.latent_downsample ** 2, error_features_channels),
                        nn.LayerNorm(error_features_channels)
                    )
        self.input_norm = self._build_input_norm(in_channels)
        self.point_transformer = self._build_point_transformer(channels, in_channels)

        # predict multiple gaussians
        out_channels = out_channels * self.cfg.delta_gaussian_multiple

        if not self.cfg.same_num_points:
            out_channels = out_channels * self.cfg.init_gaussian_multiple

        # make sure the input size of the gaussian head is updated accordingly
        if self.cfg.use_time_encoding:
            channels += self.time_embedding_dim

        # Compute per-param group dims (needed by per_param_heads and per_param_scales)
        if self.cfg.delta_head_per_param_heads or self.cfg.delta_head_per_param_scales:
            self._per_param_group_dims = self._compute_per_param_group_dims(out_channels)

        # Scaling state for update head
        if self.cfg.predict_state_scale:
            self.state_scale_head = self._build_state_scale_head(in_channels)

        self.delta_head = self._build_delta_head(in_channels, channels, out_channels)

        # Define error calculation
        # add global attention to the render error
        if self.cfg.input_error and self.cfg.input_error_mv_attn:
            assert self.cfg.input_error_resnet_feature
            self.error_mv_attn = nn.ModuleList([
                MultViewLowresAttn(error_features_channels)
                for _ in range(self.cfg.input_error_mv_attn_blocks)
            ])

        self.param_slices = get_gaussian_param_slices(self.cfg.sh_d)

    def _reset_knn_caches(self) -> None:
        """Invalidate cached KNN indices on all point-transformer sub-modules.

        Must be called whenever the number of Gaussians changes (e.g. after add_new)
        so the next forward recomputes KNN from scratch instead of using stale indices
        that index out-of-bounds into the grown point cloud.
        """
        for module in self.modules():
            if hasattr(module, "cache_knn_idx"):
                module.cache_knn_idx = None

    @property
    def adc_object_dict_to_adjust(self):
        if self.cfg.any_adc:
            object_dict: dict[str, Any] = {"optimizer_state": None}
            # For ADC
            if self.cfg.input_gradient_normalize and self.cfg.input_gradient_normalize_type == "adam":
                object_dict.update(self.input_norm.subgroups_view(self.param_slices))
        else:
            return None

        return object_dict

    def _frozen_param_groups(self) -> set:
        """Gaussian parameter groups the delta head does not predict (frozen).

        Any combination of freeze_mean/scale/rotation/opacity is supported. The knn
        optimizer treats SH as a single group, so SH is frozen only when both freeze_sh0
        and freeze_shN are set — a partial SH freeze is not expressible here.
        """
        frozen = set()
        if self.cfg.freeze_mean:
            frozen.add("means")
        if self.cfg.freeze_scale:
            frozen.add("scales")
        if self.cfg.freeze_rotation:
            frozen.add("rotations")
        if self.cfg.freeze_opacity:
            frozen.add("opacities")
        assert self.cfg.freeze_sh0 == self.cfg.freeze_shN, \
            "knn optimizer treats SH as one group; freeze_sh0 and freeze_shN must match"
        if self.cfg.freeze_sh0 and self.cfg.freeze_shN:
            frozen.add("shs")
        return frozen

    def _compute_per_param_group_dims(self, out_channels):
        """Compute per-parameter-group output dimensions from total out_channels.

        Returns a dict {group_name: dim} in the same order as split_delta_gaussians,
        omitting frozen groups, with the delta/init-gaussian multipliers folded in.
        """
        p = get_gaussian_param_sizes(self.cfg.sh_d)
        excluded = self._frozen_param_groups()

        multiplier = self.cfg.delta_gaussian_multiple
        if not self.cfg.same_num_points:
            multiplier *= self.cfg.init_gaussian_multiple

        group_dims = {name: p[key] * multiplier for name, key in _GAUSSIAN_GROUPS if name not in excluded}

        assert sum(group_dims.values()) == out_channels, (
            f"Per-param group dims {dict(group_dims)} sum={sum(group_dims.values())} != out_channels={out_channels}"
        )
        return group_dims

    def _build_per_param_heads(self, channels, out_channels):
        """Build per-parameter-group heads (Feature A).

        Each head: Linear(channels, hidden) -> act -> Linear(hidden, dim+1)
        The +1 is a per-group scalar scale. Each head independently normalizes + scales.
        """
        act_cls = get_activation_cls(self.cfg.delta_head_act)
        hidden_dim = self.cfg.delta_head_per_param_hidden_dim

        # Set up scale activation (shared across all per-param heads)
        scale_act_name = self.cfg.delta_head_scalar_scale_act
        init_bias_map = {'softplus': -1, 'relu': 1e-8, 'abs': 1e-8}
        if scale_act_name not in init_bias_map:
            raise ValueError(f"Unsupported scalar_scale_act: {scale_act_name}")
        act_class = get_activation_cls(scale_act_name)
        self.scale_act = act_class(beta=1) if scale_act_name == 'softplus' else act_class()

        heads = nn.ModuleDict()
        for name, dim in self._per_param_group_dims.items():
            # SH head gets 2x hidden dim (more outputs to predict)
            h = hidden_dim * 2 if name == "shs" else hidden_dim

            layers = [nn.Linear(channels, h), act_cls()]
            for _ in range(self.cfg.delta_head_layer_num - 2):
                layers += [nn.Linear(h, h), act_cls()]
            layers.append(nn.Linear(h, dim + 1))  # +1 for scalar scale

            head = nn.Sequential(*layers)

            # Zero-init last layer (deltas start at 0)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            # Init scale bias
            nn.init.constant_(head[-1].bias[-1], init_bias_map[scale_act_name])

            heads[name] = head

        return heads

    def _build_delta_head(self, in_channels, channels, out_channels):
        update_head_activation_cls = get_activation_cls(self.cfg.delta_head_act)
        final_head_activation_cls = get_activation_cls(self.cfg.delta_head_final_act)

        # skip connection to the image color
        if self.cfg.delta_head_concat_img:
            channels += 3 * (self.cfg.latent_downsample ** 2)

        # Feature A: per-parameter-group heads (early return — builds ModuleDict instead of Sequential)
        if self.cfg.delta_head_per_param_heads:
            assert not self.cfg.delta_head_per_param_scales, "per_param_heads already includes per-group scales"
            return self._build_per_param_heads(channels, out_channels)

        if self.cfg.delta_head_scalar_scale:
            if self.cfg.delta_head_per_param_scales:
                # Feature B: one scalar scale per parameter group
                out_channels = out_channels + len(self._per_param_group_dims)
            else:
                out_channels = out_channels + 1

        # Hidden layers keep the point-transformer channel width.
        hidden_dim = channels

        # Build update head
        layers_list = [
            nn.Linear(channels, hidden_dim),
            update_head_activation_cls()
        ]
        for i in range(self.cfg.delta_head_layer_num - 2):
            layers_list += [
                nn.Linear(hidden_dim, hidden_dim),
                update_head_activation_cls(),
            ]

        layers_list += [
            nn.Linear(hidden_dim, out_channels),
            final_head_activation_cls()
        ]
        delta_head = nn.Sequential(*layers_list)

        # init the delta as 0
        nn.init.zeros_(delta_head[-2].weight)
        if final_head_activation_cls == torch.nn.Sigmoid:
            desired_init_delta = 0.005
            bias = math.log(desired_init_delta / (1 - desired_init_delta))  # ~= -4.6
            nn.init.constant_(delta_head[-2].bias, bias)
        else:
            nn.init.zeros_(delta_head[-2].bias)

        # Scalar scale output
        if self.cfg.delta_head_scalar_scale:
            # Set the initial scale to very low number, to get the gradients flow
            init_bias_map = {
                'softplus': -1,
                'relu': 1e-8,
                'abs': 1e-8,
            }

            act_name = self.cfg.delta_head_scalar_scale_act
            if act_name not in init_bias_map:
                raise ValueError(f"Unsupported scalar_scale_out_act: {act_name}")

            # Initialize bias for scale output(s)
            if self.cfg.delta_head_per_param_scales:
                num_groups = len(self._per_param_group_dims)
                for i in range(num_groups):
                    nn.init.constant_(delta_head[-2].bias[-(num_groups - i)], init_bias_map[act_name])
            else:
                nn.init.constant_(delta_head[-2].bias[-1], init_bias_map[act_name])

            # Create activation
            act_class = get_activation_cls(act_name)
            self.scale_act = act_class(beta=1) if act_name == 'softplus' else act_class()

        return delta_head

    def _build_input_norm(self, in_channels):
        if self.cfg.input_gradient_normalize:
            assert self.cfg.input_gradient, "for now we only normalize when using gradient as input"
            if self.cfg.input_gradient_normalize_type == 'layer':
                return nn.LayerNorm(in_channels)
            elif self.cfg.input_gradient_normalize_type == 'group':
                return CustomGroupNorm([self.gaussian_param_num, self.state_channels, self.gaussian_param_num])
            elif self.cfg.input_gradient_normalize_type == 'batch':
                return nn.BatchNorm1d(in_channels, affine=False)
            elif self.cfg.input_gradient_normalize_type == 'g3r':
                return SlicedG3RNorm(in_channels, slice(-self.gaussian_param_num, None))
            elif self.cfg.input_gradient_normalize_type == 'adam':
                assert not self.cfg.input_gradient_log and self.cfg.input_gradient_scale == 1
                return AdamInputSmoothing(input_slice=slice(-self.gaussian_param_num, None))
            else:
                raise ValueError(f"normalization type not supported {self.cfg.input_gradient_normalize_type}")
        else:
            return nn.Identity()

    def _build_point_transformer(self, channels, in_channels):
        point_transformer = nn.Sequential(
            PointLinearWrapper(in_channels, channels),
            PlainPointTransformer(channels, self.cfg.knn_samples,
                                  num_blocks=self.cfg.num_basic_blocks,
                                  qk_norm=self.cfg.pt_qk_norm,
                                  norm_pt_block=self.cfg.norm_pt_block,
                                  num_heads=self.cfg.pt_heads,
                                  no_rpe=True,
                                  no_attn=self.cfg.no_knn_attn,
                                  no_norm=self.cfg.no_tran_block_norm,
                                  act=self.cfg.tran_block_act,
                                  attn_proj_channels=self.cfg.attn_proj_channels,
                                  with_mv_attn=self.cfg.with_mv_attn,
                                  with_mv_attn_lowres=self.cfg.with_mv_attn_lowres,
                                  no_mv_attn=self.cfg.no_mv_attn,
                                  conv_with_norm=self.cfg.mv_attn_conv_with_norm,
                                  mv_shuffle_attn=self.cfg.mv_shuffle_attn,
                                  with_pos_enc=self.cfg.mv_attn_with_pos_enc,
                                  shuffle_attn_no_norm=self.cfg.shuffle_attn_no_norm,
                                  mv_unimatch_attn=self.cfg.mv_unimatch_attn,
                                  use_checkpointing=self.cfg.use_checkpointing,
                                  use_fused_attn=self.cfg.use_fused_attn,
                                  knn_idx_update_every=self.cfg.knn_idx_update_every,
                                  knn_backend=self.cfg.knn_backend,
                                  knn_pos_clamp=self.cfg.knn_pos_clamp,
                                  )
        )

        # Init normalization layers
        if self.cfg.input_normalize_state:
            for block in point_transformer[1].blocks:
                nn.init.zeros_(block.norm1.bias)
                nn.init.zeros_(block.norm2.bias)
                nn.init.ones_(block.norm1.weight)
                nn.init.ones_(block.norm2.weight)

        return point_transformer

    @staticmethod
    def _build_state_scale_head(in_channels):
        state_scale_head = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.ReLU()
        )

        # Init the scale to 1
        # nn.init.zeros_(state_scale_head[-2].weight)
        nn.init.ones_(state_scale_head[-2].bias)

        return state_scale_head

    def _define_channels(self):
        # The optimizer refines a fixed per-Gaussian representation, independent of the initializer:
        # means/3D position(3), scales(3), rotation(4), opacity(1), SH(3*sh_d). Whatever position
        # encoding the initializer used (e.g. resplat's 2D pixel offset) has already been unprojected
        # to a 3D point by the time gaussians reach the optimizer. Sized from the same per-group sizes
        # the freeze logic uses below, so the total and the per-group counts cannot drift.
        param_sizes = get_gaussian_param_sizes(self.cfg.sh_d)
        gaussian_param_num = sum(param_sizes.values())

        # Get error channels
        if self.cfg.input_error:
            error_channels, error_feature_channels = self.define_error_channels()
        else:
            error_channels, error_feature_channels = 0, 0

        # Get gradient channels
        if self.cfg.input_gradient:
            gradient_channels = gaussian_param_num * self.cfg.init_gaussian_multiple
        else:
            gradient_channels = 0

        # final input channels
        input_signal_channels = gradient_channels + error_channels

        if self.cfg.same_num_points:
            in_channels = (gaussian_param_num
                           + self.state_channels
                           + input_signal_channels)
        else:
            in_channels = (gaussian_param_num * self.cfg.init_gaussian_multiple
                           + self.state_channels
                           + input_signal_channels)

        if self.cfg.concat_init_state:
            in_channels += self.state_channels

        out_channels = gaussian_param_num
        for name, key in _GAUSSIAN_GROUPS:
            if name in self._frozen_param_groups():
                out_channels -= param_sizes[key]
        channels = self.state_channels
        if self.cfg.input_alpha:
            # pixel shuffle the alpha channel to the latent resolution
            in_channels += self.cfg.latent_downsample ** 2  # alpha
        if self.cfg.input_depth or self.cfg.input_depth_smooth_error:
            # pixel shuffle the depth channel to the latent resolution
            in_channels += self.cfg.latent_downsample ** 2  # depth
        return channels, in_channels, gaussian_param_num, out_channels, error_feature_channels

    def define_error_channels(self):
        if self.cfg.no_render_error:
            error_channels = 0
        else:
            if self.cfg.input_error_rgb_no_shuffle:
                error_channels = 3
            else:
                error_channels = 3 * self.cfg.latent_downsample ** 2

        if self.cfg.input_error_resnet_feature:
            # 3 scales: 1/2, 1/4, 1/8, channels: 64, 64, 128
            if self.cfg.input_error_resnet_feature_layers in (18, 34):
                error_feature_channels = 64 + 64 if self.cfg.input_error_shallow_resnet_feature else 64 + 64 + 128
            elif self.cfg.input_error_resnet_feature_layers == 50:
                error_feature_channels = 64 + 256 + 512
            else:
                raise NotImplementedError
            error_channels = error_feature_channels
        else:
            error_feature_channels = 256

        return error_channels, error_feature_channels

    def optimizer_preprocessing(self, optimizer_input: OptimizerInput, from_init: bool) -> None:
        """Prepare the initializer's Gaussians and seed the per-Gaussian latent state at scene start.

        On a fresh scene (from_init) scales the initial opacities, pads/truncates SH to sh_d, and builds
        the latent state vector; on a buffer resume it only refreshes the state, leaving init_state intact.
        """
        if self.cfg.input_error_remain_context or self.cfg.input_error_merge_remain_context:
            assert self.cfg.input_error_cache_resnet_feature

        # Image dimensions
        context = optimizer_input.context
        b, v, _, h, w = context["image"].shape

        # Prepare Gaussians
        if from_init:
            # Scale initial opacities (in normal scale)
            opacities = optimizer_input.prev_output.gaussians.opacities  # post activation, in [0, 1]
            scaled_opacities = opacities * self.cfg.scale_initial_opacities  # default to 1.0
            optimizer_input.prev_output.gaussians.opacities = scaled_opacities

            # Process shs
            shs = optimizer_input.prev_output.gaussians.harmonics  # [B, N, 3, init_sh_d]
            init_sh_d = shs.shape[-1]
            if init_sh_d != self.cfg.sh_d:
                if init_sh_d > self.cfg.sh_d:
                    shs = shs[:, :, :, :self.cfg.sh_d]  # truncate  [B, N, 3, sh_d]
                else:
                    pad = self.cfg.sh_d - init_sh_d
                    shs = F.pad(shs, (0, pad), "constant", 0)
                optimizer_input.prev_output.gaussians.harmonics = shs

            # Effective SH-degree cap: zero the higher bands at init so they start
            # (and, with delta_shN zeroed below, stay) at zero -> DC-only rendering.
            if self.cfg.cap_sh_degree is not None:
                keep = (self.cfg.cap_sh_degree + 1) ** 2
                shs = optimizer_input.prev_output.gaussians.harmonics
                if keep < shs.shape[-1]:
                    shs[..., keep:] = 0.0

        # Prepare state
        # Gaussians dimensions
        n = optimizer_input.prev_output.gaussians.means.shape[1]
        vector_state = self.get_vector_state(b, v, n, optimizer_input, from_init)

        if from_init:
            # Set everything so that the optimizer isn't aware whether it's a new scene
            # Convert InitializerOutput to OptimizerPreviousOutput
            optimizer_input.prev_output = OptimizerPreviousOutput(
                gaussians=optimizer_input.prev_output.gaussians,
                # Per-op latent-state policy from the refiner; "random" reuses the scene-start init
                # scale (cfg.init_state_scale). Adam moments are unaffected (they follow FastGS).
                state=KnnBasedOptimizerState(
                    new_state_scale=self.cfg.init_state_scale,
                    new_state_policy={
                        "clone": self.cfg.refiner.clone_gaussian_state,
                        "split": self.cfg.refiner.split_gaussian_state,
                        "relocate": self.cfg.refiner.relocate_gaussian_state,
                        "add": self.cfg.refiner.add_gaussian_state,
                    },
                ),
            )
        optimizer_input.prev_output.state.state = vector_state
        # init_state captures the scene-start state used by some experiments;
        # only set it on a fresh scene so ckpt-buffer resumes preserve the original value.
        # Left None when none of the three init_state flags are active.
        if from_init and (self.cfg.concat_init_state or self.cfg.replace_init_state or self.cfg.residual_init_state):
            optimizer_input.prev_output.state.init_state = vector_state

    def _forward_impl(
            self,
            i: int,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput,
            full_context: BatchedViews,
            full_target: BatchedViews,
            **kwargs
    ) -> OptimizerOutput:
        """Run one optimization iteration: apply the update step, then ADC, then save renders/stats."""

        with self.benchmarker.time("iter"):
            # Unpack
            iter_context: BatchedViews = optimizer_input.context
            target: BatchedViews = optimizer_input.target
            renderer: Decoder = optimizer_input.renderer
            b, v, _, h, w = iter_context["image"].shape
            assert b == 1, "Batch size > 1 not supported for post-processing"

            # Log number of gaussians
            self.benchmarker.record("gaussians", optimizer_input.prev_output.gaussians.means.shape[1])

            # One optimization step
            res = self._apply_step(
                i, optimizer_input, optimizer_output
            )
            updated_gaussians: Gaussians = res[0]
            state: Tensor = res[1]
            meta_for_adc: dict = res[2]
            updates: dict[str, Tensor] = res[3]
            grads_raw: Tensor | None = res[4]
            normalized_grads: Tensor | None = res[5]
            updated_scaled_state: Tensor | None = res[6]
            gaussians_sel: Tensor | None = res[7]

        # Log stats
        if grads_raw is not None:
            nonzero_grads = (grads_raw != 0).any(-1)  # [B, G]
            assert nonzero_grads.shape[0] == 1
            self.benchmarker.record("nonzero_grads", nonzero_grads[0].sum().item())

        # Densification and Pruning
        if self.cfg.any_adc:

            n_before_adc = updated_gaussians.means.shape[1]

            # Prepare objects to adjust during ADC.
            # Write the current state back so ADC mutation methods see it.
            optimizer_input.prev_output.state.state = state
            object_dict = self.adc_object_dict_to_adjust
            object_dict["optimizer_state"] = optimizer_input.prev_output.state

            # Apply ADC — mutates state (and init_state if set) in place
            self.apply_adc(
                i=i, v=v, h=h, w=w, adc_state=optimizer_input.prev_output.state.adc_state,
                gaussians=updated_gaussians, meta=meta_for_adc, object_dict_to_adjust=object_dict,
                renderer=optimizer_input.renderer, full_context=full_context,  # full scene, not minibatch
            )

            # Read back the (possibly grown/pruned) state tensor
            state = optimizer_input.prev_output.state.state

            del object_dict["optimizer_state"]
            if self.cfg.input_gradient_normalize and self.cfg.input_gradient_normalize_type == "adam":
                self.input_norm.aggregate_from_subgroups(object_dict, self.param_slices)

            # If N changed (add_new grew the population), stale KNN caches in the
            # point transformer modules would index out-of-bounds on the next forward
            # pass → CUDA illegal memory access. Reset them so they are recomputed.
            if updated_gaussians.means.shape[1] != n_before_adc:
                self._reset_knn_caches()

        # Save updated gaussians and state
        optimizer_input.prev_output.gaussians = updated_gaussians
        optimizer_input.prev_output.state.state = state

        if self.cfg.input_gradient_normalize_type == "adam":
            optimizer_input.prev_output.state.adam_state = self.input_norm.get_state()

        if self.training:
            optimizer_output.gaussian_list.append(updated_gaussians)

        # Save per-iteration debug info (test time only)
        if not self.training and self.save_every(i + 1, tag="info"):
            self._save_info_stats(optimizer_output, updated_gaussians, updates, grads_raw,
                                  normalized_grads, updated_scaled_state, gaussians_sel, state)

        # Post-update context + target renders
        self._save_post_update_renders(
            i, optimizer_input, optimizer_output, updated_gaussians,
            full_context, full_target,
        )

        # Optimizer output is being changed in place, but for clarity we return it
        return optimizer_output

    def _save_info_stats(self, optimizer_output, updated_gaussians, updates, grads_raw,
                         normalized_grads, updated_scaled_state, gaussians_sel, state):
        """Prepare per-iteration debug stats (deltas, gradients, state norms) and save them.

        knn-specific prep: unpack shs into sh0s/shNs, split the flat gradient vectors into
        per-param dicts, and restore zeroed-out (excluded) Gaussians; then hand the
        already-CPU values to the shared Optimizer._append_info.
        """
        # unpack shs into sh0s / shNs
        shs = updates.pop("shs")  # [1, N, 3*sh_d]
        assert shs.shape[0] == 1, "Batch size > 1 not supported"
        shs = rearrange(shs.squeeze(0), "n (c x) -> n c x", c=3, x=self.cfg.sh_d)  # [N, 3, sh_d]
        updates["sh0s"] = shs[..., 0:1]
        updates["shNs"] = shs[..., 1:] if self.cfg.sh_d > 1 else None

        info = {"deltas": {k: v.squeeze(0).cpu() if v is not None else None for k, v in updates.items()}}

        if grads_raw is not None:
            if gaussians_sel is not None:
                # Restore the zeroed-out (excluded) Gaussians so the logged tensors are full-size
                b, g_valid, d = grads_raw.shape
                g = state.shape[0]
                grads_raw_full = torch.zeros((b, g, d))
                normalized_grads_full = torch.zeros((b, g, d))
                grads_raw_full[:, gaussians_sel, :] = grads_raw.cpu()
                normalized_grads_full[:, gaussians_sel, :] = normalized_grads.cpu()
                grads_raw = grads_raw_full
                normalized_grads = normalized_grads_full

            info["grads"] = split_grads(grads_raw.cpu(), self.cfg)
            if normalized_grads is not None:
                info["normalized_grads"] = split_grads(normalized_grads.cpu(), self.cfg)
                assert info["grads"]["means"].shape == info["normalized_grads"]["means"].shape, \
                    f"Shape mismatch between grads and normalized_grads: " \
                    f"{info['grads']['means'].shape} vs {info['normalized_grads']['means'].shape}"
            else:
                info["normalized_grads"] = None
            if updated_scaled_state is not None:
                info["states_norms"] = torch.norm(updated_scaled_state, dim=-1).cpu()  # [B, N]

        self._append_info(optimizer_output, updated_gaussians, **info)

    def _apply_step(
            self,
            i,
            optimizer_input: OptimizerInput,
            optimizer_output: OptimizerOutput
    ) -> tuple[Gaussians, Tensor, dict, dict[str, Tensor], Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        """Compute one update: input signal -> point-transformer state update -> delta head -> new Gaussians.

        Returns the updated Gaussians and latent state plus the intermediates ADC and the debug logging need.
        """
        # Unpacking
        context = optimizer_input.context
        gaussians = optimizer_input.prev_output.gaussians  # Gaussian object of [B, N, C]
        state = optimizer_input.prev_output.state.state  # [N, C]
        init_state = optimizer_input.prev_output.state.init_state  # [N, C]
        # Get input signal for the optimizer model (errors/gradients)
        with self.benchmarker.time("decoder"):
            input_signal, gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads, means2d_abs_grads = (
                self.prepare_input_signal(context, gaussians, optimizer_input.renderer)
            )

        # Preparing meta for ADC
        if means2d_grads is not None:
            means2d_grads = means2d_grads.detach()  # [B, V, N, 2]
        if means2d_abs_grads is not None:
            means2d_abs_grads = means2d_abs_grads.detach()  # [B, V, N, 2]
        meta_for_adc = {
            "visibility_filter": context_render_output.visibility_filter.detach(),  # [B, V, N]
            "radii": context_render_output.radii.detach(),  # [B, V, N, 1]
            "means_2d_grads": means2d_grads,  # [B, V, N, 2]
            "means_2d_abs_grads": means2d_abs_grads,  # FastGS Abs-GS split signal (None for other backends)
        }

        # Some initializers pad the scene with extra, invisible (zero-opacity) Gaussians so that every
        # GPU optimizes the same number of Gaussians (ColmapInitializer, train_fixed_gaussians_num; the
        # true count is gaussians.nr_valid). The optimizer predicts an update for every Gaussian, so
        # updating one of these dummies would raise its opacity and turn it into a real Gaussian. Because
        # the dummies are invisible their gradient is zero, so updating only nonzero-gradient Gaussians
        # (update_only_nonzero_grad=True, done by filter_gaussians_for_optimizer_step) leaves them as they
        # are. Every training config does this; the check below rejects the combination we don't support.
        has_padding = 0 <= gaussians.nr_valid < gaussians.means.shape[1]
        if has_padding and not self.cfg.update_only_nonzero_grad:
            raise ValueError(
                f"update_only_nonzero_grad=False with padded init Gaussians (nr_valid="
                f"{gaussians.nr_valid} < N={gaussians.means.shape[1]}): the invisible padding Gaussians "
                f"would be optimized and turn into real ones. Set update_only_nonzero_grad=True (it skips "
                f"them, since their gradient is zero), or use an initializer without padding."
            )

        # Handle zero gradient gaussians
        # We either prune them, or exclude them from the input/output update
        if self.cfg.update_only_nonzero_grad and gaussian_grads is not None:
            gaussian_grads, gaussian_grads_raw, gaussians, grad_sign, init_state, input_signal, state = (
                self.filter_gaussians_for_optimizer_step(gaussian_grads, gaussian_grads_raw, gaussians, grad_sign,
                                                         init_state, input_signal, state)
            )

        # During training, skip a scene whose active-Gaussian count is outside the workable
        # range: too many to fit in memory, or too few for the KNN neighborhood.
        # NOTE: over-cap scenes are discarded wholesale; subsampling down to
        # max_active_gaussians instead is a possible future alternative (would change
        # training dynamics, so it needs validation).
        active_gaussians_num = state.shape[0]
        if self.training:
            if active_gaussians_num > self.cfg.max_active_gaussians:
                raise SkipBatchException(
                    f"iter {i}: {active_gaussians_num} active gaussians exceeds max ({self.cfg.max_active_gaussians})"
                )
        if active_gaussians_num < self.cfg.knn_samples:
            raise SkipBatchException(
                f"iter {i}: only {active_gaussians_num} active gaussians (need >= {self.cfg.knn_samples})"
            )

        # Unpack Gaussians
        means, scales, rotations_unnorm, opacities_raw, shs = unpack_gaussians(
            gaussians,
            scales_log=self.cfg.opt_scales_before_act,
            opacities_logit=True,
            opacities_unsqueeze=True,
            detach=True,  # stop gradient of last predictions
            scales_lims=self.cfg.scales_clamp_lims,
            raw_opacities_lims=(self.cfg.clamp_min_raw_opacities, self.cfg.clamp_max_raw_opacities)
        )

        # camera_frame_quats: the network sees (and updates) the camera-frame quaternion conj(q_c2w) * q_world
        # of each Gaussian's source view; Gaussians are laid out per view as (v n).
        q_c2w = None
        if self.cfg.camera_frame_quats:
            if gaussians.sel is not None:
                raise ValueError("camera_frame_quats needs the full per-view Gaussian grid (no active-subset selection)")
            q_c2w = rotation_matrix_to_quaternion_xyzw(context["extrinsics"][..., :3, :3])  # [B, V, 4]
            q_c2w = q_c2w.repeat_interleave(means.shape[1] // q_c2w.shape[1], dim=1)  # [B, N, 4]
            q_c2w_conj = q_c2w * q_c2w.new_tensor([-1., -1., -1., 1.])
            gaussians_concat = pack_gaussians(means, scales, quaternion_multiply_xyzw(q_c2w_conj, rotations_unnorm),
                                              opacities_raw, shs)  # [B, N, C]
        else:
            gaussians_concat = pack_gaussians(means, scales, rotations_unnorm, opacities_raw, shs)  # [B, N, C]

        b, v, c, h, w = context["image"].shape
        latent_h = h // self.cfg.latent_downsample
        latent_w = w // self.cfg.latent_downsample

        # prepare pt input
        point_cloud, tmp_batch_size = self.get_point_cloud(latent_h, latent_w, means, v)
        # Create offset directly on device to avoid CPU-GPU transfer
        offset = torch.arange(1, b + 1, device=state.device, dtype=torch.long) * tmp_batch_size

        # reshape
        gaussians_flat = self.reshape_gaussians_to_nc(latent_h, latent_w, gaussians_concat, v)  # [B, N, C] --> [BN, C]
        # add global attention to exchange info across views
        if self.cfg.input_error_mv_attn:
            input_signal = self.apply_global_attn(h, input_signal, latent_h, latent_w, v, w)

        input_signal_flat = input_signal.reshape(-1,
                                                 input_signal.shape[
                                                     -1])  # [B, N, C] --> [BN, C] - faster than rearrange
        input_signal_flat = self.append_to_input_signal(b, context, context_render_output, input_signal_flat, v)

        # Normalize state before input it to the update module
        if self.cfg.input_normalize_state:
            state_norm = state.norm(dim=1, keepdim=True) / math.sqrt(state.shape[-1])  # [BG, 1]
            state = state / (state_norm + 1e-8)  # [BG, C]

        normalized_input = self.input_norm(input_signal_flat)

        if self.cfg.input_normalize_gaussians:
            gaussians_flat_mean = gaussians_flat.mean()
            gaussians_flat_std = gaussians_flat.std()
            gaussians_flat = (gaussians_flat - gaussians_flat_mean) / (gaussians_flat_std + 1e-8)

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            point_cloud, gaussians_flat, state, update_input = self.prepare_update_input(init_state, normalized_input,
                                                                                         point_cloud, gaussians_flat,
                                                                                         state)

            updated_state = self._apply_point_transformer(b, latent_h, latent_w, offset,
                                                          point_cloud, update_input, v, state, i)

            # Hard coded extract normalized gradients
            if self.cfg.input_gradient and self.cfg.input_gradient_normalize:
                normalized_grads = normalized_input
            else:
                normalized_grads = None

            # Recover the state norm
            if self.cfg.input_normalize_state:
                # state = state * state_std + state_mean
                updated_state = updated_state * state_norm

            # Predict a scale for the updtaed scale for the MLP deltas prediction
            # The updated state for the next stage remains the same
            if self.cfg.predict_state_scale:
                state_scale = self.state_scale_head(update_input.detach())
                if self.cfg.predict_state_scale_norm:
                    # Normalize the state vector
                    state_scale = state_scale / (state_scale.norm(p=2, dim=1, keepdim=True) + 1e-8)
            else:
                state_scale = torch.tensor([1], device=state.device, dtype=state.dtype)
            updated_scaled_state = state_scale * updated_state

            # optionally append time encodiing to normalize input
            with TimeEncodingWrapper(self.cfg.use_time_encoding,
                                     self.time_encoder_fn,
                                     optimizer_output.t,
                                     self.cfg.time_encoding_max_steps,
                                     updated_scaled_state) as embedded_state:
                if self.cfg.use_time_encoding:
                    assert not self.cfg.concat_init_state
                    assert not self.cfg.replace_init_state

                # delta gaussian head
                delta_gaussians = self.apply_delta_gaussian_head(b, context, init_state, embedded_state, v)

        delta_means, delta_opacities, delta_rotations, delta_scales, delta_shs, init_repeat, delta_gaussians = (
            self.postprocess_deltas(b, delta_gaussians, gaussian_grads, gaussians_concat, grad_sign,
                                    normalized_grads, state, optimizer_output.t)
        )

        means, opacities_raw, rotations_unnorm, scales, shs = self.repeat_gaussians(means, opacities_raw,
                                                                                    rotations_unnorm, scales, shs)

        if q_c2w is not None:
            # A camera-frame delta d on q_cam is the delta q_c2w * d on q_world = q_c2w * q_cam
            delta_rotations = quaternion_multiply_xyzw(q_c2w, delta_rotations)
        covariances, means, scales, rotations, rotations_unnorm, opacities_raw, shs = self._apply_gaussian_deltas(
            delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs,
            means, scales, rotations_unnorm, opacities_raw, shs, init_repeat)

        # Recover the state in non valid gaussians (and grad for logging)
        if gaussians.sel is not None:
            sel = gaussians.sel  # [B, G]
            full_state = optimizer_input.prev_output.state.state

            # Convert full state to the dtype of state
            full_state = full_state.to(state.dtype)
            # Use non-in-place index_put to avoid in-place modification of tensors
            # in the autograd computation graph (fixes version mismatch errors with stability loss)
            updated_state = full_state.index_put((sel,), updated_state)
        else:
            sel = None

        # update gaussians (only where mask is True)
        # Use view instead of rearrange for speed
        shs_reshaped = shs.view(shs.shape[0], shs.shape[1], 3, -1)
        gaussians = gaussians.update_object_by_curr_mask(
            means=means,
            covariances=covariances,
            harmonics=shs_reshaped,
            opacities=opacities_raw.squeeze(-1).sigmoid(),
            scales=scales,
            rotations=rotations,
            rotations_unnorm=rotations_unnorm,
            sel=None,
            deltas=delta_gaussians if self.training else None,
            gradients=gaussian_grads_raw if self.training else None,
            norm_gradients=normalized_grads.unsqueeze(0) if normalized_grads is not None and self.training else None
        )

        updates = {
            "means": delta_means.detach(),
            "scales": delta_scales.detach(),
            "rotations": delta_rotations.detach(),
            "opacities": delta_opacities.detach(),
            "shs": delta_shs.detach()
        }

        grads_raw = gaussian_grads.detach() if gaussian_grads is not None else None
        grads_adam = normalized_grads.detach() if normalized_grads is not None else None

        return gaussians, updated_state, meta_for_adc, updates, grads_raw, grads_adam, updated_scaled_state, sel

    def postprocess_deltas(self, b, delta_gaussians, gaussian_grads, gaussians_concat, grad_sign,
                           normalized_grads, state, t):
        # Head outputs -> parameter deltas (gradient-input rescaling / log scale, scalar scale, negation)
        delta_gaussians_raw = delta_gaussians
        delta_gaussians = self._postprocess_head_deltas(delta_gaussians_raw, grad_sign, normalized_grads)

        # Rearrange back to [B, N, C]
        delta_gaussians, delta_gaussians_raw = self.rearrange_delta_gaussians(b, delta_gaussians, delta_gaussians_raw)

        # Split
        delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs, init_repeat = (
            self.split_delta_gaussians(delta_gaussians)
        )

        # Apply lr
        delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs = self.scale_deltas_with_lr(
            t, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs
        )

        return delta_means, delta_opacities, delta_rotations, delta_scales, delta_shs, init_repeat, delta_gaussians

    def filter_gaussians_for_optimizer_step(self, gaussian_grads, gaussian_grads_raw, gaussians, grad_sign,
                                            init_state, input_signal, state):
        # Compute a mask for gaussians that did not contribute to any pixel of context views.
        # Their gradients are strictly zero; skip them for the update but don't prune (they
        # may be relevant in other views).
        assert gaussian_grads.shape[0] == 1, "Batch size > 1 not supported with mask"

        # A gaussian is valid if it has a nonzero gradient in any channel
        valid_g = gaussian_grads[0].any(dim=-1)  # [G] bool
        sel = None

        # if everything is valid, skip all slicing work
        if not valid_g.all():
            sel = valid_g.nonzero(as_tuple=True)[0]  # [G_valid]

            input_signal = input_signal[:, sel, :]  # [B, G_valid, C]

            gaussian_grads = gaussian_grads[:, sel, :]  # [B, G_valid, D]
            if gaussian_grads_raw is not None:
                gaussian_grads_raw = gaussian_grads_raw[:, sel, :]
            if grad_sign is not None:
                grad_sign = grad_sign[:, sel, :]

            state = state[sel, :]  # [G_valid, C]
            if init_state is not None:
                init_state = init_state[sel, :]  # [G_valid, C]

        gaussians.sel = sel

        if self.cfg.input_gradient_normalize_type == "adam":
            self.input_norm.sel = sel
        return gaussian_grads, gaussian_grads_raw, gaussians, grad_sign, init_state, input_signal, state

    def scale_deltas_with_lr(self, t, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs):
        # Scale deltas with learning rates
        delta_means = delta_means * self.scheduler.get_lr(t, "means")
        delta_scales = delta_scales * self.scheduler.get_lr(t, "scales")
        if delta_rotations is not None:
            delta_rotations = delta_rotations * self.scheduler.get_lr(t, "rotations")
        delta_opacities = delta_opacities * self.scheduler.get_lr(t, "opacities")

        # Use view instead of rearrange for speed
        delta_shs = delta_shs.view(delta_shs.shape[0], delta_shs.shape[1], 3, -1)  # [b, g, 3, c]
        delta_sh0 = delta_shs[..., 0]  # [B, N, C]
        delta_shN = delta_shs[..., 1:]
        delta_sh0 = delta_sh0 * self.scheduler.get_lr(t, "sh0")
        delta_shN = delta_shN * self.scheduler.get_lr(t, "shN")
        delta_shs = torch.cat((delta_sh0.unsqueeze(-1), delta_shN), dim=-1)
        delta_shs = delta_shs.flatten(-2)  # [b, g, d*c] - faster than rearrange
        return delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs

    def _unshuffle_to_flat(self, x: torch.Tensor, b: int, v: int) -> torch.Tensor:
        """Pixel-unshuffle a (b*v, c, h, w) tensor and flatten spatial dims: -> (b*v*h*w, c)."""
        x = F.pixel_unshuffle(x, downscale_factor=self.cfg.latent_downsample)
        return rearrange(x, "(b v) c h w -> (b v h w) c", b=b, v=v)

    def _downsample_error(self, x: torch.Tensor, b: int, v: int) -> torch.Tensor:
        """Downsample a (b*v, c, h, w) error tensor to latent resolution: -> (b, v*h*w, c)."""
        if self.cfg.input_error_rgb_no_shuffle:
            x = F.interpolate(x, scale_factor=1. / self.cfg.latent_downsample,
                              mode='bilinear', align_corners=True)
        else:
            # NOTE: pixel_unshuffle assumes per-pixel, grid-aligned Gaussians. A non-grid
            # subsampling (e.g. farthest-point sampling) would need a different error->Gaussian
            # mapping to stay aligned — not supported here.
            x = F.pixel_unshuffle(x, downscale_factor=self.cfg.latent_downsample)
        return rearrange(x, "(b v) c h w -> b (v h w) c", b=b, v=v)

    def append_to_input_signal(self, b, context, context_render, input_signal_flat, v):
        if self.cfg.input_alpha:
            render_alpha = rearrange(context_render.accumulated_alpha.detach(), "b v h w -> (b v) () h w")
            render_alpha = self._unshuffle_to_flat(render_alpha, b, v)
            input_signal_flat = torch.cat((input_signal_flat, render_alpha), dim=-1)
        if self.cfg.input_depth:
            render_depth = rearrange(context_render.depth.detach(), "b v h w -> (b v) () h w")
            render_depth = self._unshuffle_to_flat(render_depth, b, v)
            input_signal_flat = torch.cat((input_signal_flat, render_depth), dim=-1)
        if self.cfg.input_depth_smooth_error:
            disp = 1. / context_render.depth.detach().clamp(min=1e-3, max=1000.)  # [B, V, H, W]
            disp = rearrange(disp, "b v h w -> (b v) () h w")

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            tmp_imgs = rearrange(context["image"], "b v c h w -> (b v) c h w")

            depth_smooth_error = get_smooth_loss(norm_disp, tmp_imgs, no_mean=True)
            depth_smooth_error = self._unshuffle_to_flat(depth_smooth_error, b, v)
            input_signal_flat = torch.cat((input_signal_flat, depth_smooth_error), dim=-1)
        return input_signal_flat

    def repeat_gaussians(self, prev_means, prev_opacities_raw, prev_rotations_unnorm, prev_scales, prev_shs):
        # NOTE: only repeat at the first iteration
        refine_repeat = self.cfg.delta_gaussian_multiple
        if refine_repeat > 1:
            # predict multiple gaussians for each point
            prev_means = prev_means.repeat(1, refine_repeat, 1)
            prev_scales = prev_scales.repeat(1, refine_repeat, 1)
            prev_rotations_unnorm = prev_rotations_unnorm.repeat(1, refine_repeat, 1)
            prev_opacities_raw = prev_opacities_raw.repeat(1, refine_repeat, 1)  # smaller opacities, important
            prev_shs = prev_shs.repeat(1, refine_repeat, 1)
        return prev_means, prev_opacities_raw, prev_rotations_unnorm, prev_scales, prev_shs

    def split_delta_gaussians(self, delta_gaussians):
        if self.cfg.init_gaussian_multiple > 1 and not self.cfg.same_num_points:
            init_repeat = self.cfg.init_gaussian_multiple
        else:
            init_repeat = 1
        p = get_gaussian_param_sizes(self.cfg.sh_d)
        frozen = self._frozen_param_groups()

        # Split the head output among the predicted (non-frozen) groups; frozen groups get
        # an additive-identity zero delta so downstream apply/rearrange is uniform.
        b, n = delta_gaussians.shape[0], delta_gaussians.shape[1]
        active = [(name, key) for name, key in _GAUSSIAN_GROUPS if name not in frozen]
        active_chunks = delta_gaussians.split([p[key] * init_repeat for _, key in active], dim=-1)
        deltas = dict(zip((name for name, _ in active), active_chunks))
        for name, key in _GAUSSIAN_GROUPS:
            if name in frozen:
                deltas[name] = delta_gaussians.new_zeros(b, n, p[key] * init_repeat)

        delta_means, delta_scales = deltas["means"], deltas["scales"]
        delta_rotations, delta_opacities, delta_shs = deltas["rotations"], deltas["opacities"], deltas["shs"]

        if (
                self.cfg.delta_gaussian_multiple > 1 or self.cfg.init_gaussian_multiple > 1) and not self.cfg.same_num_points:
            delta_means = rearrange(delta_means, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_scales = rearrange(delta_scales, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_rotations = rearrange(delta_rotations, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_opacities = rearrange(delta_opacities, "b n (c k) -> b (n k) c", k=init_repeat)
            delta_shs = rearrange(delta_shs, "b n (c k) -> b (n k) c", k=init_repeat)
        return delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs, init_repeat

    def rearrange_delta_gaussians(self, b, delta_gaussians, delta_gaussians_raw):
        # [BV, C]
        # update gaussian parameters
        delta_gaussians = rearrange(delta_gaussians, "(b n) c -> b n c", b=b)
        delta_gaussians_raw = rearrange(delta_gaussians_raw, "(b n) c -> b n c", b=b)
        return delta_gaussians, delta_gaussians_raw

    def _apply_gaussian_deltas(self, delta_means, delta_scales, delta_rotations, delta_opacities, delta_shs,
                               means, scales, rotations_unnorm, opacities_raw, shs,
                               repeat):
        means = self._apply_mean_delta(delta_means, means)

        # clamp the scale
        scales = self._apply_scale_delta(delta_scales, scales, repeat)
        if self.cfg.opt_scales_before_act:
            scales = scales.exp()

        if not self.cfg.freeze_rotation:
            rotations, rotations_unnorm = self._apply_rotation_delta(delta_rotations, rotations_unnorm)
        else:
            rotations = F.normalize(rotations_unnorm, dim=-1)

        # compute covariance
        covariances = build_covariance(scales, rotations)  # ([1, VHW, 3, 3])

        opacities_raw = self._apply_opacity_delta(delta_opacities, opacities_raw, repeat)
        shs = self._apply_shs_delta(delta_shs, shs)
        return covariances, means, scales, rotations, rotations_unnorm, opacities_raw, shs

    def _apply_shs_delta(self, delta_shs, prev_shs):
        shs = prev_shs + delta_shs  # [B, N, 3*sh_d]

        if self.cfg.clamp_shs_soft:
            assert self.cfg.clamp_min_shs == -self.cfg.clamp_max_shs, "For soft clamp, min and max should be symmetric around 0"
            shs = torch.tanh(shs / self.cfg.clamp_max_shs) * self.cfg.clamp_max_shs
        else:
            shs = shs.clamp(min=self.cfg.clamp_min_shs, max=self.cfg.clamp_max_shs)

        # Effective SH-degree cap: hold the bands above cap_sh_degree at zero so a
        # full-sh_d checkpoint renders with only the lower-degree color. shs is the
        # final per-step harmonics [B, N, 3*sh_d] (channel-major [3, sh_d] flattened).
        if self.cfg.cap_sh_degree is not None:
            keep = (self.cfg.cap_sh_degree + 1) ** 2
            sh_d = shs.shape[-1] // 3
            if keep < sh_d:
                shs = shs.view(*shs.shape[:-1], 3, sh_d)
                shs[..., keep:] = 0.0
                shs = shs.flatten(-2)

        return shs

    def _apply_opacity_delta(self, delta_opacities, prev_opacities_raw, repeat):
        # update init opacities when predicting multiple gaussians
        if repeat > 1 and not self.cfg.multi_gaussian_scale_smaller and (self.cfg.init_gaussian_multiple == 1):
            # Given y = sigmoid(x), to get new x' such that sigmoid(x') = y / K:
            # The formula is: x' = x + log((1 - y) / (K - y))
            # This adjusts x so that the sigmoid output is scaled down by a factor of K
            tmp_sigmoid = prev_opacities_raw.sigmoid()
            prev_opacities_raw = prev_opacities_raw + torch.log(
                (1 - tmp_sigmoid) / (repeat - tmp_sigmoid)) + delta_opacities
        else:
            prev_opacities_raw = prev_opacities_raw + delta_opacities
            # prev_opacities_raw = prev_opacities_raw.clamp(min=-5, max=5)
        return prev_opacities_raw

    @staticmethod
    def _apply_rotation_delta(delta_rotations, prev_rotations_unnorm):
        assert delta_rotations is not None
        prev_rotations_unnorm = prev_rotations_unnorm + delta_rotations
        # normazlie
        prev_rotations = prev_rotations_unnorm / (prev_rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
        return prev_rotations, prev_rotations_unnorm

    def _apply_scale_delta(self, delta_scales, prev_scales, repeat):
        """Add the predicted scale delta and clamp the result.

        When one point is represented by `repeat` overlapping gaussians (repeat > 1), the
        cluster is too dense unless we compensate. `multi_gaussian_scale_smaller` chooses to
        compensate on size: each gaussian's initial scale is divided by `repeat`, so the
        cluster covers about the same footprint as the single original gaussian (opacity is
        left unchanged). When the flag is off, size is kept and the opacity is reduced instead
        (see _apply_opacity_delta).
        """
        if repeat > 1 and self.cfg.multi_gaussian_scale_smaller:
            # smaller initial scales (the floor is applied by the shared clamp below).
            # The divide-by-repeat is in activation space, so this branch assumes scales
            # are not in raw/log space.
            assert not self.cfg.opt_scales_before_act, \
                "multi_gaussian_scale_smaller assumes opt_scales_before_act=False (activation-space scales)"
            new_scales = prev_scales / repeat + delta_scales
        else:
            new_scales = prev_scales + delta_scales

        if self.cfg.opt_scales_before_act:
            min_scale = self.cfg.clamp_min_raw_scales
            max_scale = self.cfg.clamp_max_raw_scales
        else:
            min_scale = self.cfg.clamp_min_scale
            max_scale = self.cfg.clamp_max_scale

        new_scales = new_scales.clamp(min=min_scale, max=max_scale)

        return new_scales

    @staticmethod
    def _apply_mean_delta(delta_means, prev_means):
        prev_means = (prev_means + delta_means)
        return prev_means

    def _on_scene_start_impl(self, optimizer_input: OptimizerInput) -> None:
        # Reset the state
        if isinstance(optimizer_input.prev_output, InitializerOutput):  # New scene
            from_init = True
            # Reset the optimizer state for a new scene
            # We cannot just use super().on_scene_start() because we need to process the InitializerOutput in case it
            # contain conditioning features
            self.reset_logs()

            if self.cfg.input_gradient_normalize_type == "adam":
                self.input_norm.reset()
                nr_gaussians = rearrange(optimizer_input.prev_output.gaussians.means, "b n c -> (b n) c").shape[0]
                param_num = self.gaussian_param_num
                self.input_norm.initialize(shape=(nr_gaussians, param_num),
                                           device=optimizer_input.prev_output.gaussians.means.device)

            # make sure xyz are contiguous
            optimizer_input.prev_output.gaussians.means = optimizer_input.prev_output.gaussians.means.contiguous()
        elif isinstance(optimizer_input.prev_output, OptimizerPreviousOutput):
            from_init = False
            if self.cfg.input_gradient_normalize_type == "adam":
                # Continuing previous optimization from ckpt buffer
                self.input_norm.update_state(optimizer_input.prev_output.state.adam_state)

            # Note: logs are not handled right now for continuing from ckpt buffer
            self.reset_logs()
        else:
            raise ValueError(f"Unknown prev_output type {type(optimizer_input.prev_output)}")

        # Prepare the input for both cases (new scene and continuing from the ckpt buffer), converting an init_output to a prev_output internally.
        self.optimizer_preprocessing(optimizer_input, from_init=from_init)

        # initialize adc state, after converting to prev_output
        if from_init and self.cfg.any_adc:
            self.initialize_adc_state(self.cfg, optimizer_input)

    def reshape_gaussians_to_nc(self, latent_h, latent_w, gaussians_concat, v):
        if self.cfg.init_gaussian_multiple > 1 and not self.cfg.same_num_points:
            # gaussians are with more points, reshape
            factor = self._block_factor()
            gaussians_flat = rearrange(gaussians_concat, "b (v h x w y) c -> (b v h w) (c x y)",
                                       v=v, h=latent_h, w=latent_w, x=factor, y=factor)
        else:
            gaussians_flat = rearrange(gaussians_concat, "b n c -> (b n) c")
        return gaussians_flat

    def get_point_cloud(self, latent_h, latent_w, prev_means, v):
        if self.cfg.init_gaussian_multiple > 1 and not self.cfg.same_num_points:
            # The initializer predicts a (factor x factor) block of gaussians per latent pixel,
            # so there are more points than the base grid. Subsample back to the base grid so the
            # point cloud (the KNN structure for the transformer) stays at the base resolution.
            factor = self._block_factor()
            point_cloud = rearrange(prev_means, "b (v h w) c -> b v h w c",
                                    v=v, h=latent_h * factor, w=latent_w * factor,
                                    )
            # use a uniform grid subsample of the point cloud to reduce points
            point_cloud = point_cloud[:, :, ::factor, ::factor]
            point_cloud = rearrange(point_cloud, "b v h w c -> (b v h w) c")
            tmp_batch_size = v * latent_h * latent_w
        else:
            point_cloud = rearrange(prev_means, "b n c -> (b n) c")
            tmp_batch_size = prev_means.shape[1]
        return point_cloud, tmp_batch_size

    def get_vector_state(self, b, v, n, optimizer_input, from_init):
        if from_init:
            # Starting a new scene directly from the initializer
            # State should not be provided
            # Create initial state
            # optimizer_input.prev_output is of type InitializerOutput
            if optimizer_input.prev_output.features is None or self.cfg.init_state_wo_features:
                # Synthesize state per-Gaussian when this scene's init gives no features (a point-cloud
                # init like colmap, incl. the point-cloud scenes of a mixed run) or the optimizer has no
                # feature branch (init_state_wo_features). Feature scenes (resplat) take the else branch.
                with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
                    dtype = torch.get_autocast_dtype('cuda')
                    if self.cfg.init_state_type == "constant":
                        state = torch.ones((b, n, self.cfg.state_channels), device=self.device, dtype=dtype)
                    elif self.cfg.init_state_type == "random":
                        state = torch.randn((b, n, self.cfg.state_channels), device=self.device, dtype=dtype)
                    else:
                        raise ValueError(f"Unknown init_state_type {self.cfg.init_state_type}")
                    state = state * self.cfg.init_state_scale
            else:
                # Calculating state from initializer features
                state = self.get_state_from_condition_features(b, optimizer_input.prev_output.features,
                                                               v)  # [B, N, C]

            # combine gaussians of all scenes in the batch [B, N, C] -> [B*N, C]
            state = rearrange(state, "b n c -> (b n) c")
        else:
            # Restarting a scene from a ckpt buffer: the state is already stored flat as [B*N, C].
            state = optimizer_input.prev_output.state.state

        return state

    @staticmethod
    def _align_features(features, latent_h: int, latent_w: int) -> list:
        """Resize each feature map to (latent_h, latent_w) if needed and return as a list."""
        out = []
        vals = features.values() if isinstance(features, dict) else features
        for feat in vals:
            if feat.shape[-2:] != (latent_h, latent_w):
                feat = F.interpolate(feat, size=(latent_h, latent_w), mode='bilinear', align_corners=True)
            out.append(feat)
        return out

    def _block_factor(self) -> int:
        """Per-pixel block side length when the initializer predicts multiple gaussians per pixel.

        init_gaussian_multiple gaussians form a (factor x factor) block, so factor = sqrt(multiple).
        """
        multiple = self.cfg.init_gaussian_multiple
        factor = math.isqrt(multiple)
        assert factor * factor == multiple, f"init_gaussian_multiple must be a perfect square, got {multiple}"
        return factor

    def _get_latent_size(self, h: int, w: int) -> tuple[int, int]:
        """Compute latent (H, W) from image (H, W), accounting for init_gaussian_multiple upsampling."""
        latent_h = h // self.cfg.latent_downsample
        latent_w = w // self.cfg.latent_downsample
        if self.cfg.init_gaussian_multiple > 1 and self.cfg.same_num_points:
            factor = self._block_factor()
            latent_h *= factor
            latent_w *= factor
        return latent_h, latent_w

    def render_input_views_for_error_calc(self, context, prev_gaussians, renderer):
        _, _, _, h, w = context["image"].shape  # [B, V, C, H, W]

        render_res = (h, w)

        # Use only first N views
        end = self.cfg.input_error_num_views if self.cfg.input_error_num_views > 0 else None

        return renderer.forward_batch_subset(
            prev_gaussians,
            context,
            render_res,
            start=None,
            end=end,
            return_radii=False
        )

    def get_state_from_condition_features(self, b, condition_features, v):
        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp, dtype=torch.bfloat16):
            if not self.cfg.pt_update_amp and condition_features.dtype == torch.bfloat16:
                condition_features = condition_features.float()
            state = self.state_proj(condition_features.detach())  # [B, C, H, W]
        if self.cfg.init_gaussian_multiple > 1 and self.cfg.same_num_points:
            factor = self._block_factor()
            state = F.interpolate(state, scale_factor=factor, mode='bilinear', align_corners=True)
        # Convert to vector of Gaussians per batch [B, N, C]
        state = rearrange(state, "(b v) c h w -> b (v h w) c", b=b, v=v)  # N = v * h * w
        return state

    def prepare_update_input(self, init_state, input_signal, point_cloud, gaussians_flat, state):
        if self.cfg.replace_init_state:
            state = init_state

        if self.cfg.no_render_error:
            update_input = torch.cat((gaussians_flat, state), dim=-1)
        else:
            update_input = torch.cat((gaussians_flat, state, input_signal), dim=-1)
        if self.cfg.concat_init_state:
            update_input = torch.cat((update_input, init_state), dim=-1)
        return point_cloud, gaussians_flat, state, update_input

    def _apply_point_transformer(self, b, latent_h, latent_w, offset, point_cloud, update_input, v, state, iter):

        # Non-finite (inf/nan) positions or features are a divergence, skip the scene in eval
        if not (torch.isfinite(point_cloud).all() and torch.isfinite(update_input).all()):
            if self.training:
                raise AssertionError("Non-finite Gaussians before point-transformer KNN")
            raise SkipBatchException("Non-finite Gaussians before point-transformer KNN")

        def recurrent_chunk(update_input, point_cloud, offset):
            pxo = self.point_transformer[0]([point_cloud, update_input, offset])
            state = self.point_transformer[1](pxo, iter=iter, b=b, v=v, h=latent_h, w=latent_w)
            return state

        if self.cfg.use_checkpointing or self.cfg.recurrent_use_checkpointing:
            new_state = torch.utils.checkpoint.checkpoint(
                recurrent_chunk,
                update_input, point_cloud, offset,
                use_reentrant=False,
            )
        else:
            new_state = recurrent_chunk(update_input, point_cloud, offset)

        if self.cfg.residual_state:
            new_state = new_state + state
        return new_state

    def apply_delta_gaussian_head(self, b, context, init_state, state, v):
        if self.cfg.delta_head_concat_img:
            img_unshuffle = rearrange(context["image"], "b v c h w -> (b v) c h w")
            img_unshuffle = self._unshuffle_to_flat(img_unshuffle, b, v)
            head_input = torch.cat((state, img_unshuffle), dim=-1)

        else:
            if self.cfg.residual_init_state:
                head_input = state + init_state
            else:
                head_input = state

        if self.cfg.delta_head_per_param_heads:
            delta_gaussians = self._apply_per_param_heads(head_input)
        else:
            delta_gaussians = self.delta_head(head_input)

        return delta_gaussians

    def _apply_per_param_heads(self, head_input):
        """Run per-parameter-group heads and concatenate results.

        Each head outputs [N, dim+1] where the last dim is the scalar scale.
        Per-group normalize + scale is applied independently.
        """
        deltas = []
        for name, dim in self._per_param_group_dims.items():
            raw = self.delta_head[name](head_input)  # [N, dim+1]
            scale = self.scale_act(raw[:, -1:])  # [N, 1]
            delta = raw[:, :-1]  # [N, dim]
            if dim > 1:
                delta = delta / (delta.norm(p=2, dim=-1, keepdim=True) + 1e-8) * scale
            else:
                # 1-d (e.g. opacities): no direction to normalize, just scale magnitude
                delta = delta * scale
            deltas.append(delta)
        return torch.cat(deltas, dim=-1)

    def apply_global_attn(self, h, input_signal, latent_h, latent_w, v, w):
        assert self.cfg.input_error_resnet_feature
        assert self.cfg.input_error

        if self.cfg.input_gradient and self.cfg.input_error:
            input_render_error = input_signal[..., :self.error_features_channels]
        else:
            input_render_error = input_signal

        with torch.amp.autocast(device_type='cuda', enabled=self.cfg.use_amp, dtype=torch.bfloat16):
            for blk in self.error_mv_attn:
                if self.cfg.same_num_points:
                    # no downsample, for re10k 256
                    input_render_error = blk(input_render_error, v=v, h=h, w=w)
                else:
                    input_render_error = blk(input_render_error, v=v, h=latent_h, w=latent_w)

        if self.cfg.input_gradient and self.cfg.input_error:
            input_signal[..., :self.error_features_channels] = input_render_error
        else:
            input_signal = input_render_error

        return input_signal

    def prepare_input_signal(self, context, gaussians, renderer):
        """Build the per-Gaussian signal that drives the update: render gradients and/or feature error.

        Renders the context views once and turns the errors with the inputs into the network's input
        signal (gradients from `_calc_input_gradients`, render error, or both concatenated).
        """
        # make sure at least one of the following is True
        assert self.cfg.input_gradient or self.cfg.input_error
        input_signal = None
        input_render_error = None
        gaussian_grads_raw = None
        gaussian_grads = None
        grad_sign = None
        means2d_grads = None
        means2d_abs_grads = None

        # calculate input gradients
        if self.cfg.input_gradient:
            gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads, means2d_abs_grads = (
                self._calc_input_gradients(context, gaussians, renderer)
            )

            input_signal = gaussian_grads_raw
        else:
            # Render if input_gradient=False. Otherwise, the gradients render can be reused for the error
            # calculation, all further usage detach before use.
            context_render_output = self.render_input_views_for_error_calc(context, gaussians, renderer)

        # calculate input rendering errors
        if self.cfg.input_error:
            if means2d_grads is None and self.cfg.need_2d_grads:
                raise NotImplementedError("Calculating 2dgrad for ADC is not implemented for error input alone")
            input_render_error = self._calc_input_errors(context, context_render_output)
            input_signal = input_render_error

        if self.cfg.input_gradient and self.cfg.input_error:
            # Concatenate both gradients and errors
            input_signal = torch.cat((input_render_error, gaussian_grads), dim=-1)

        return input_signal, gaussian_grads_raw, gaussian_grads, grad_sign, context_render_output, means2d_grads, means2d_abs_grads

    def _calc_input_errors(self, context, input_render):
        b, v, _, h, w = context["image"].shape
        # Detach the last rendered object
        input_rgb = input_render.color.detach()
        # compute input view rendering error
        if self.cfg.input_error_resnet_feature:
            input0 = rearrange(input_rgb, "b v c h w -> (b v) c h w")
            if self.cfg.input_error_num_views > 0:
                gt_input = context["image"][:, :self.cfg.input_error_num_views, :, :, :]
            else:
                gt_input = context["image"]
            input1 = rearrange(gt_input, "b v c h w -> (b v) c h w")

            transform = _IMAGENET_NORM
            concat = torch.cat((input0, input1), dim=0)

            input_tensor = transform(concat)
            with torch.amp.autocast(device_type='cuda', enabled=self.cfg.pt_update_amp,
                                    dtype=torch.bfloat16):
                with torch.no_grad():
                    features = self.error_feature_extractor(input_tensor)

            # align to the latent resolution
            latent_h, latent_w = self._get_latent_size(h, w)

            all_features = torch.cat(self._align_features(features, latent_h, latent_w), dim=1)

            render_view_features = all_features[:input0.shape[0]]
            gt_view_features = all_features[input0.shape[0]:]

            feature_error = render_view_features - gt_view_features

            if self.cfg.input_error_num_views > 0:
                # pad to V views
                curr_v = self.cfg.input_error_num_views
                indices = torch.arange(v) * curr_v // v
                feature_error = rearrange(feature_error, "(b v) c h w -> b v c h w", b=b)
                feature_error = feature_error[torch.arange(b).unsqueeze(1), indices, :, :, :]
                input_render_error = rearrange(feature_error, "b v c h w -> b (v h w) c")
            else:
                input_render_error = rearrange(feature_error, "(b v) c h w -> b (v h w) c", b=b)

        else:
            input_render_error = (input_render.color - context["image"]).abs()  # [B, V, 3, H, W]
            input_render_error = rearrange(input_render_error, "b v c h w -> (b v) c h w")

            input_render_error = self._downsample_error(input_render_error, b, v)  # [B, N, C]

        # include both feature error and image error
        if self.cfg.input_error_add_rgb_feature:
            rgb_render_error = input_render.color - context["image"]
            rgb_render_error = rearrange(rgb_render_error, "b v c h w -> (b v) c h w")

            rgb_render_error = self._downsample_error(rgb_render_error, b, v)  # [B, N, C]

            rgb_render_error = self.rgb_error_proj(rgb_render_error)
            input_render_error = input_render_error + rgb_render_error

        return input_render_error

    def get_input_error_feature_extractor(self):
        error_feature_extractor = None
        # resnet feature
        if self.cfg.input_error_resnet_feature:
            error_feature_extractor = ResNetFeatureWarpper(
                shallow_resnet_feature=self.cfg.input_error_shallow_resnet_feature)

            if self.cfg.input_error_no_freeze_resnet_feature:
                # remove unused layers
                # NOTE: layer 3 is also not used
                error_feature_extractor.layer3 = nn.Identity()
                error_feature_extractor.train()
                for params in error_feature_extractor.parameters():
                    params.requires_grad = True
            else:
                error_feature_extractor.eval()

                for params in error_feature_extractor.parameters():
                    params.requires_grad = False

        return error_feature_extractor

    def _postprocess_head_deltas(self, delta_gaussians, grad_sign, normalized_grad):
        if self.cfg.input_gradient:
            delta_gaussians = delta_gaussians / self.cfg.input_gradient_scale
            if self.cfg.input_gradient_log:
                grad_sign = rearrange(grad_sign, "b n c -> (b n) c")
                # recover log scale for applying the deltas.
                # For loss calculation the delta should still be in log scale

                delta_gaussians = grad_sign * (delta_gaussians.exp() - 1e-8)

                if self.cfg.input_gradient_log_clip_deltas > 0:
                    # clip the delta to avoid too large updates
                    clip_value = self.cfg.input_gradient_log_clip_deltas
                    clip_mask = delta_gaussians.abs() > clip_value
                    delta_gaussians[clip_mask] = delta_gaussians[clip_mask].sign() * clip_value

        if self.cfg.delta_head_scalar_scale:
            if self.cfg.delta_head_per_param_heads:
                # Already handled in _apply_per_param_heads — nothing to do here
                pass
            elif self.cfg.delta_head_per_param_scales:
                # Feature B: per-group scalar scales
                num_groups = len(self._per_param_group_dims)
                scales = delta_gaussians[:, -num_groups:]  # [G, num_groups]
                scales = self.scale_act(scales)
                deltas = delta_gaussians[:, :-num_groups]  # [G, D]

                normalized_deltas = []
                offset = 0
                for i, (name, dim) in enumerate(self._per_param_group_dims.items()):
                    group_delta = deltas[:, offset:offset + dim]  # [G, dim]
                    group_scale = scales[:, i:i + 1]  # [G, 1]
                    if dim > 1:
                        group_delta = group_delta / (group_delta.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                    group_delta = group_delta * group_scale
                    normalized_deltas.append(group_delta)
                    offset += dim

                delta_gaussians = torch.cat(normalized_deltas, dim=-1)
            else:
                # Original global scalar scale
                scale = delta_gaussians[:, -1:]  # [G, 1]
                scale = self.scale_act(scale)  # make sure scale is positive
                deltas_unnorm = delta_gaussians[:, :-1]  # [G, D]
                deltas_norm = deltas_unnorm / (deltas_unnorm.norm(p=2, dim=1, keepdim=True) + 1e-8)  # [G, D]
                delta_gaussians = deltas_norm * scale

        if self.cfg.input_gradient:
            if self.cfg.scale_residual_grads:
                delta_gaussians = delta_gaussians * normalized_grad * self.cfg.residual_grad_scale  # 1.0

            # To match the default behavior of SGD, Adam, and other optimizers, deltas are negated.
            # SGD applies the gradients as `x = x - lr * grad`, while resaplt applies them as `x = x + lr * deltas`.
            delta_gaussians = -delta_gaussians

        return delta_gaussians

    def _calc_input_gradients(self, context, gaussians, renderer):
        """Render the context views and backprop the reconstruction loss to per-Gaussian parameter gradients.

        These gradients are a feature fed to the network (the "what is wrong" signal), not used to update
        weights — the graph is detached. Chunked over views to bound memory; optionally also returns the
        2D means gradients ADC consumes.
        """
        assert not self.cfg.input_gradient_same_loss, "input_gradient_same_loss is not implemented"
        _, v, _, h, w = context["image"].shape

        with torch.enable_grad():

            # Unpack gaussians
            means, scales, rotations_unnorm, opacities_raw, shs = unpack_gaussians(
                gaussians,
                scales_log=self.cfg.opt_scales_before_act,
                opacities_logit=True,
                opacities_unsqueeze=True,
                detach=True,
                clone=False,
                requires_grad=True,
                scales_lims=self.cfg.scales_clamp_lims,
                raw_opacities_lims=(self.cfg.clamp_min_raw_opacities, self.cfg.clamp_max_raw_opacities)
            )

            # Create temporary Gaussians with same values but requires_grad=True
            grad_batch_size = self.cfg.input_gradients_chunk_size
            if grad_batch_size == -1:
                grad_batch_size = v
            gaussian_grads = 0
            means2d_grads_chunks = []
            means2d_abs_grads_chunks = []  # FastGS Abs-GS split signal; stays empty for gsplat/inria
            inner_loss_sum = 0.  # accumulate per-chunk inner loss (for the progress bar)
            inner_loss_components_sum = {}  # accumulate its l1/ssim components for the bar
            nr_chunks = math.ceil(v / grad_batch_size)

            # Pre-compute shapes and config flags outside the loop
            shs_shape = (shs.shape[0], shs.shape[1], 3, -1)
            opt_scales_before_act = self.cfg.opt_scales_before_act
            # Pre-compute normalized rotations once (not in gradient inputs, so no grad needed)
            with torch.no_grad():
                rotations = rotations_unnorm / (rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)

            for chunk_idx, start, stop in chunk_index_iter(v, grad_batch_size):
                # zero grads

                means = means.detach().requires_grad_(True)
                scales = scales.detach().requires_grad_(True)
                rotations_unnorm = rotations_unnorm.detach().requires_grad_(True)
                opacities_raw = opacities_raw.detach().requires_grad_(True)
                shs = shs.detach().requires_grad_(True)

                # Apply activation to scales if needed (before calculating covariance)
                scales_act = scales.exp() if opt_scales_before_act else scales

                tmp_gaussians = Gaussians(
                    means=means,
                    covariances=None,
                    harmonics=shs.view(shs_shape),
                    opacities=torch.sigmoid(opacities_raw.squeeze(-1)),
                    scales=scales_act,
                    rotations=rotations,
                    rotations_unnorm=rotations_unnorm,
                )

                # render input views, calculate inner loss and backprop to get gradients
                context_render_output = renderer.forward_batch_subset(
                    tmp_gaussians,
                    context,
                    start=start,
                    end=stop,
                    image_shape=(h, w),
                )

                inputs = [means, scales, rotations_unnorm, opacities_raw, shs]

                if self.cfg.need_2d_grads:
                    assert context_render_output.means2d is not None, "output_renderer.means2d is None"
                    means2d = context_render_output.means2d  # [B, V, N, 2]
                    # means2d.retain_grad()  # retain grad for means2d grads computation
                    inputs.append(means2d)
                    # FastGS only: second leaf for the Abs-GS split gradient (None for gsplat/inria).
                    if context_render_output.means2d_abs is not None:
                        inputs.append(context_render_output.means2d_abs)

                inner_loss, inner_loss_components = inner_loss_for_input_gradients(
                    context["image"][:, start:stop],
                    context_render_output,
                    reduction=self.cfg.input_gradient_loss_reduction,
                    with_ssim=self.cfg.input_gradient_with_ssim_loss,
                    l2=self.cfg.input_gradient_l2_loss,
                    return_components=True,
                )
                if self.cfg.opacity_reg_lambda > 0.0:
                    inner_loss = inner_loss + self.cfg.opacity_reg_lambda * torch.sigmoid(opacities_raw).mean()
                inner_loss_sum = inner_loss_sum + inner_loss.detach()
                for _k, _v in inner_loss_components.items():
                    inner_loss_components_sum[_k] = inner_loss_components_sum.get(_k, 0.) + _v.detach()
                # allow_unused: the abs leaf is unused by gsplat/inria, so leave its grad None there.
                grads = torch.autograd.grad(outputs=inner_loss,
                                            inputs=inputs,
                                            create_graph=False,
                                            retain_graph=False,
                                            allow_unused=True,
                                            )

                gaussian_grads = gaussian_grads + torch.cat(grads[:5], dim=-1)  # [B, G, D]
                if torch.isnan(gaussian_grads).any():
                    # A non-finite input gradient means this scene's optimization has diverged. During
                    # eval (test/val) raise SkipBatchException so optimize_scene stops early and scores
                    # the steps completed before the blow-up (or drops the scene if it diverged at the
                    # first step), instead of aborting the whole run. During training we still fail hard
                    # so a real meta-training instability is surfaced rather than silently skipped.
                    if self.training:
                        raise AssertionError("NaN detected in gaussian_grads")
                    raise SkipBatchException("NaN detected in gaussian_grads")
                if self.cfg.need_2d_grads:
                    means2d_grads_chunks.append(grads[5])  # [B, V_chunk, N, 2]
                    # FastGS Abs-GS split signal (grads[6]); absent for gsplat/inria.
                    if context_render_output.means2d_abs is not None and grads[6] is not None:
                        means2d_abs_grads_chunks.append(grads[6])  # [B, V_chunk, N, 2]

            gaussian_grads = gaussian_grads / nr_chunks
            self.last_inner_loss = inner_loss_sum / nr_chunks  # latest inner loss (tensor); float()'d only when the bar shows it
            self.last_inner_loss_components = {_k: _v / nr_chunks for _k, _v in inner_loss_components_sum.items()}

            if self.cfg.need_2d_grads:
                means2d_grads = torch.cat(means2d_grads_chunks, dim=1)  # [B, V, N, 2]
                # Normalize to NDC per the renderer's convention so the ADC stays renderer-agnostic
                # (gsplat=pixel→×w/2,h/2; inria/fastgs already NDC).
                means2d_grads = renderer.means2d_grad_to_ndc(means2d_grads, (h, w))
                if self.cfg.input_gradient_loss_reduction == "mean_pixels_sum_views":
                    means2d_grads = means2d_grads / v
            else:
                means2d_grads = None

            # FastGS Abs-GS split signal, concatenated + NDC-normalized like means2d_grads above;
            # None for gsplat/inria (no abs leaf, so nothing was collected).
            means2d_abs_grads = None
            if means2d_abs_grads_chunks:
                means2d_abs_grads = torch.cat(means2d_abs_grads_chunks, dim=1)  # [B, V, N, 2]
                means2d_abs_grads = renderer.means2d_grad_to_ndc(means2d_abs_grads, (h, w))
                if self.cfg.input_gradient_loss_reduction == "mean_pixels_sum_views":
                    means2d_abs_grads = means2d_abs_grads / v

            gaussian_grads_raw = gaussian_grads * self.cfg.input_gradient_scale
            if self.cfg.input_gradient_log:
                # log gradients
                grads_sign = gaussian_grads.sign()
                gaussian_grads_raw = (gaussian_grads_raw.abs() + 1e-8).log()
            else:
                grads_sign = None

            # Detach gradients to avoid gradient flow through the input
            gaussian_grads = gaussian_grads.detach()
            gaussian_grads_raw = gaussian_grads_raw.detach()
            if grads_sign is not None:
                grads_sign = grads_sign.detach()

        # Returning also the render output, but it can only be used for visualization,
        # as we already backpropagate gradients through it
        return gaussian_grads_raw, gaussian_grads, grads_sign, context_render_output, means2d_grads, means2d_abs_grads


