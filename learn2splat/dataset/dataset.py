from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    pose_align_middle_view: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
    opencv_pose_format: bool | None

    test_start_idx: int  # skip the first N scenes during test (for scene-chunked SLURM jobs)