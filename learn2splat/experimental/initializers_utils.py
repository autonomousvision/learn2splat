import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

try:
    # GPU O(N log N) neighbour distances (3DGS simple-knn). torch is imported above, so its
    # libc10.so is already loaded and the C extension resolves.
    from simple_knn._C import distCUDA2
except Exception:
    distCUDA2 = None

from learn2splat.misc.general_utils import SkipBatchException
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.common.gaussian_adapter import RGB2SH, build_covariance


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


# Flipped to True the first time a distCUDA2 call raises, so we fall back permanently instead of
# retrying every step. This catches an installed simple-knn build that imports fine but lacks the
# running GPU's compute arch (e.g. "not compiled for SM 80" on A100).
_distcuda2_unusable = False


def nn_dist2_for_scales(xyz: Tensor) -> Tensor:
    """Mean squared distance to each point's nearest neighbours -> initial isotropic Gaussian scales
    (3DGS convention; shared by all point-based initializers). Uses simple-knn's GPU distCUDA2 when
    available and usable on this GPU, else the sklearn `knn` fallback, which copies points to CPU
    (the dominant cost of the per-step depth init: ~100ms -> ~6ms)."""
    global _distcuda2_unusable
    if distCUDA2 is not None and xyz.is_cuda and not _distcuda2_unusable:
        try:
            return distCUDA2(xyz.contiguous())  # [N]
        except RuntimeError as e:
            _distcuda2_unusable = True
            warnings.warn(
                f"distCUDA2 failed ({e}); falling back to CPU sklearn knn for Gaussian scale init. "
                "Rebuild simple-knn for this GPU's compute arch to restore the faster GPU path."
            )
    return (knn(xyz, 4)[:, 1:] ** 2).mean(dim=-1)  # [N]


def colored_points_to_gaussians(
    xyz: Tensor,
    rgbs: Tensor,
    *,
    sh_degree: int,
    scaling_factor: float,
    init_opacity: float,
    fixed_num: int | None = None,
    device: Optional[torch.device] = None,
) -> Gaussians:
    """Convert a colored point cloud into a Gaussians object.

    The build tail of the depth initializer: knn-based scale init, constant opacity, identity
    rotations, RGB->SH, and padding up to a fixed count for DDP consistency. The caller subsamples
    the point cloud first (`Initializer.subsample_points`), so this only pads. `nr_valid` records
    the real (pre-pad) count; padded Gaussians are invisible (zero opacity/scale).
    """
    if xyz.shape[0] == 0:
        raise SkipBatchException("No valid points to initialize Gaussians from. Skipping batch.")

    dist2_avg = nn_dist2_for_scales(xyz)  # [N]
    scales = torch.sqrt(dist2_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    opacities = torch.full((xyz.shape[0],), init_opacity)
    nr_valid = xyz.shape[0]

    if fixed_num is not None and xyz.shape[0] < fixed_num:
        pad = fixed_num - xyz.shape[0]
        xyz = F.pad(xyz, (0, 0, 0, pad), value=0.0)
        rgbs = F.pad(rgbs, (0, 0, 0, pad), value=0.0)
        scales = F.pad(scales, (0, 0, 0, pad), value=1e-10)
        opacities = F.pad(opacities, (0, pad), value=1e-10)

    gaussians_dict = points_to_gaussians(
        {"xyz": xyz, "rgb": rgbs, "scales": scales * scaling_factor, "opacities": opacities},
        sh_degree=sh_degree,
        device=device,
    )
    sh0, shN = gaussians_dict["sh0"], gaussians_dict["shN"]
    harmonics = (torch.cat([sh0, shN], dim=1) if shN is not None else sh0).permute(0, 2, 1)  # [N, 3, sh_d]
    opacities = torch.sigmoid(gaussians_dict["opacities_raw"])
    scales = torch.exp(gaussians_dict["scales_raw"])
    rotations = F.normalize(gaussians_dict["rotations_unnorm"], dim=-1)
    covariances = build_covariance(scale=scales, rotation_xyzw=rotations)

    return Gaussians(
        means=gaussians_dict["xyz"].unsqueeze(0),
        covariances=covariances.unsqueeze(0),
        harmonics=harmonics.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        scales=scales.unsqueeze(0),
        rotations=rotations.unsqueeze(0),
        rotations_unnorm=gaussians_dict["rotations_unnorm"].unsqueeze(0),
        nr_valid=nr_valid,
    )


def points_to_gaussians(
    points_dict: dict[str, Tensor],
    sh_degree: int = 3,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Tensor]:
    
    xyz = points_dict["xyz"].clone().to(device)
    N = xyz.shape[0]
    
    # color is SH coefficients
    rgbs = points_dict["rgb"].clone().to(device)  # [N, 3], in [0, 1]
    
    # if sh_degree > 0:
    shs = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=device)  # [N, K, 3]
    shs[:, 0, :] = RGB2SH(rgbs)
    sh0 = shs[:, :1, :]  # [N, 1, 3]
    if sh_degree > 0:
        shN = shs[:, 1:, :]  # [N, K-1, 3]
    else:
        shN = None

    # Identity rotation (XYZW [0,0,0,1] -> R=I, since quaternion_to_matrix unbinds real-last),
    # matching 3DGS/FastGS SfM init. Random quats are render-neutral at init (isotropic scales ->
    # covariance s^2 I regardless of R) but diverge once scales go anisotropic; identity keeps the
    # init faithful to the reference across all SfM initializers (colmap/random/pointcloud/edgs).
    quats_unnorm = torch.zeros((N, 4), device=device)
    quats_unnorm[:, 3] = 1.0  # [N, 4]

    scales = points_dict["scales"].clone().to(device)  # [N, 3]
    scales_raw = torch.log(scales)

    opacities = points_dict["opacities"].clone().to(device)  # [N,]
    opacities_raw = torch.logit(opacities)
    
    return {
        "xyz": xyz,
        "sh0": sh0,  # [N, 1, 3]
        "shN": shN,  # [N, sh_d-1, 3] or None
        "scales_raw": scales_raw,
        "rotations_unnorm": quats_unnorm,
        "opacities_raw": opacities_raw,
    }
