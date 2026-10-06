import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from tqdm import tqdm
from PIL import Image
from learn2splat.experimental.edgs.utils import (
    select_cameras_kmeans,
    k_closest_vectors,
    aggregate_confidences_and_warps,
    extract_keypoints_and_colors,
    triangulate_points,
    select_best_keypoints
)


@torch.no_grad()
def init_gaussians_with_corr(
    viewpoints_img: Tensor,   # (N, 3, H, W) original images, column-major
    viewpoints_w2c: Tensor,   # (N, 4, 4) camera-to-world matrices, column-major
    viewpoints_proj: Tensor,  # (N, 4, 4) projection matrices, column-major
    camera_centers: Tensor,   # (N, 3) camera centers in world coordinates
    init_opacity: float,
    roma_model_type: str,
    verbose: bool = False
):
    """
    For a given input gaussians and a scene we instantiate a RoMa model(change to indoors if necessary) and process scene
    training frames to extract correspondences. Those are used to initialize gaussians
    Args:
        scene: object of the Scene class.
        cfg: configuration. Use init_wC
    Returns:
        gaussians: inplace transforms object gaussians of the class GaussianModel.

    """
    # default values used in original EDGS code
    num_refs: int = 180
    nns_per_ref: int = 3
    matches_per_ref: int = 20000
    proj_err_tolerance: float = 0.01
    device = viewpoints_w2c.device

    try:
        from romatch import roma_outdoor, roma_indoor
    except ImportError as e:
        raise ImportError(
            "The edgs initializer requires RoMa (romatch), which is not "
            "installed. Install it with: "
            "pip install git+https://github.com/Parskatt/RoMa.git"
        ) from e
    if roma_model_type == "indoors":
        roma_model = roma_indoor(device=device)
    else:
        roma_model = roma_outdoor(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False
    M = matches_per_ref
    upper_thresh = roma_model.sample_thresh
    expansion_factor = 1
    keypoint_fit_error_tolerance = proj_err_tolerance
    visualizations = {}
    
    N_VIEWS = viewpoints_img.shape[0]
    NUM_REFERENCE_FRAMES = min(num_refs, N_VIEWS)
    NUM_NNS_PER_REFERENCE = min(nns_per_ref , N_VIEWS)
    
    # Select cameras using K-means
    # viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    viewpoint_cam_all = viewpoints_w2c.reshape(N_VIEWS, -1)  # (N_VIEWS, 16)
    
    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)
   
    # Find the k-closest vectors for each vector
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    
    if verbose:
        print("Indices of k-closest vectors for each vector:\n", closest_indices)

    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_rgb = []
    all_new_scaling = []
    all_new_opacities_raw = []

    # Run roma_model.match once to kinda initialize the model
    viewpoint_img1 = viewpoints_img[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    viewpoint_img2 = viewpoints_img[1].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    imA = Image.fromarray(np.clip(viewpoint_img1 * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(viewpoint_img2 * 255, 0, 255).astype(np.uint8))
    
    warp, certainty_warp = roma_model.match(imA, imB, device=device)
    if verbose:
        print("Once run full roma_model.match warp.shape:", warp.shape)
        print("Once run full roma_model.match certainty_warp.shape:", certainty_warp.shape)
    del warp, certainty_warp
    torch.cuda.empty_cache()

    for source_idx in tqdm(sorted(selected_indices)):
        
        # 1. Compute keypoints and warping for all the neigboring views
        # Call the aggregation function to get imA and imB_compound
        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = aggregate_confidences_and_warps(
            # viewpoint_stack=viewpoint_stack,
            viewpoints_img=viewpoints_img,
            closest_indices=closest_indices_selected,
            roma_model=roma_model,
            source_idx=source_idx,
            verbose=verbose, output_dict=visualizations
        )

        # Triangulate keypoints
        matches = warps_max
        certainty = certainties_max
        certainty = certainty.clone()
        certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )

        # Select based on certainty elements with high confidence. These are basically all of
        # kptsA_np.
        good_samples = torch.multinomial(certainty,
                                            num_samples=min(expansion_factor * M, len(certainty)),
                                            replacement=False)

        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all
        reference_image_dict = {
            "ref_image": imA,
            "NNs_images": imB_compound,
            "certainties_all": certainties_all,
            "warps_all": warps_all,
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []

        }
        for NN_idx in tqdm(range(len(warps_all))):
            matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

            # Extract keypoints and colors
            kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
            )

            # proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
            # proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform
            
            proj_matrices_A = viewpoints_proj[source_idx]
            proj_matrices_B = viewpoints_proj[closest_indices_selected[source_idx, NN_idx]]
            # exit(0)
            triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                P1=torch.stack([proj_matrices_A] * M, axis=0),
                P2=torch.stack([proj_matrices_B] * M, axis=0),
                k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

            reference_image_dict["triangulated_points"].append(triangulated_points)
            reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
            reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)

        NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
            NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
            NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
            NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
        
        # 4. Save as gaussians
        # N = len(NNs_triangulated_points_selected)

        new_xyz = NNs_triangulated_points_selected[:, :-1]
        all_new_xyz.append(new_xyz)  # seeked_splats
        all_new_rgb.append(torch.tensor(kptsA_color.astype(np.float32) / 255.).to(device))
        
        mask_bad_points = torch.tensor(
                NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
                dtype=torch.float32)
        mask_bad_points = mask_bad_points.to(device)
        print("Number of bad points for source_idx", source_idx, ":", mask_bad_points.sum().item())
        # exit(0)
        new_opacities = torch.ones((new_xyz.shape[0]), device=device) * init_opacity
        new_opacities_raw = torch.logit(new_opacities)
        new_opacities_raw = new_opacities_raw - mask_bad_points * (1e1)
        all_new_opacities_raw.append(new_opacities_raw)
        camera_center = camera_centers[source_idx].unsqueeze(0)  # (1, 3)
        dist_points_to_cam1 = torch.linalg.norm(camera_center - new_xyz, dim=1, ord=2)
        all_new_scaling.append((dist_points_to_cam1 * 0.001).unsqueeze(1).repeat(1, 3))
    
    all_new_xyz = torch.cat(all_new_xyz, dim=0) 
    all_new_rgb = torch.cat(all_new_rgb, dim=0)
    all_new_scaling = torch.cat(all_new_scaling, dim=0)
    all_new_opacities_raw = torch.cat(all_new_opacities_raw, dim=0)
    all_new_opacities = torch.sigmoid(all_new_opacities_raw)
    
    points_dict = {
        "xyz": all_new_xyz,
        "rgb": all_new_rgb,
        "scales": all_new_scaling,
        "opacities": all_new_opacities,
    }

    return closest_indices_selected, visualizations, points_dict
