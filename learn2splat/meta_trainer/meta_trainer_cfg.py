from dataclasses import dataclass
from pathlib import Path

from learn2splat.meta_trainer.ckpt_buffer import CkptBufferCfg
from learn2splat.scene_trainer.postprocessing import PostProcessCfg
from learn2splat.model.decoder.decoder import DepthRenderingMode


@dataclass
class MetaOptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    lr_depth: float
    weight_decay: float
    warm_up_ratio: float
    adamw_8bit: bool


@dataclass
class TestCfg:
    output_path: Path | None
    compute_scores: bool
    compute_scores_metrics: list[str] | None
    metrics_batch_size: int
    eval_initialization: bool
    save_render_image: bool
    save_render_image_last_only: bool
    save_gt_image: bool
    save_render_depth: bool
    save_gt_depth: bool
    save_init_pred_depth: bool  # save the initializer's predicted input-view depth (not the rendered depth)
    save_error_image: bool
    save_video: bool
    save_video_optim: bool
    save_video_view_index: int
    save_video_frame_repeat: int
    save_video_orbit: bool
    save_video_orbit_steps: list | None
    save_video_orbit_with_optim: bool
    save_video_optim_orbit: bool
    save_video_optim_orbit_steps: list | None
    save_video_orbit_span: int
    # How long each spliced-in camera sweep plays: how many forward+backward passes over the
    # path, and how many times each rendered frame is held.
    save_video_orbit_sweeps: int
    save_video_orbit_frame_repeat: int
    # Sweep along a smooth arc anchored at save_video_view_index instead of the views themselves,
    # for datasets whose view order is not a continuous camera path (DTU, dense colmap scenes).
    # The arc turns at most span_deg away from that view (and less where the cameras stop
    # covering the scene); frames sets how slowly it gets there.
    save_video_orbit_fitted_path: bool
    save_video_orbit_fitted_path_frames: int
    save_video_orbit_fitted_path_span_deg: float | int
    save_gaussian: bool
    save_gaussian_last_only: bool
    save_poses: bool
    save_cameras_json: bool
    save_cameras_npz: bool
    # Views per chunk when rendering a camera path for the videos. Only test_render_videos_views
    # reads it; the init / per-step / saved-step renders chunk by scene_trainer.iter_batch_size.
    # TODO: unify with scene_trainer.iter_batch_size — two names for "views per render call",
    # each covering a different set of renders, is a trap when chasing a rasterizer OOM.
    render_chunk_size: int | None
    stabilize_camera: bool
    stab_camera_kernel: int
    eval_context_views: bool
    inference_window_size: int | None
    postprocessing: PostProcessCfg | None
    save_at_iters: list[int] | None
    save_every_freq: list[int] | None
    save_every_steps: list[int] | None
    skip_if_outputs_exist: bool
    scenes_filter: list[str] | None


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_save_model: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    eval_depth: bool

    eval_render_depth: bool
    viz_render_depth: bool
    viz_depth_separate: bool

    use_gt_depth_range: bool

    no_log_video: bool

    # when doing refinement, supervise input view or not since we also render input views
    loss_on_target_views: bool
    loss_on_target_views_num: int
    loss_on_input_views: bool
    loss_on_input_views_num: int

    # local window training
    train_window_size: int | None

    # Ckpt buffer
    use_ckpt_buffer: bool
    ckpt_buffer_cfg: CkptBufferCfg | None

    # L2 weight decay regularization on Gaussian properties (meta-loss)
    scale_l2_loss_weight: float
    sh_l2_loss_weight: float
    opacity_l2_loss_weight: float