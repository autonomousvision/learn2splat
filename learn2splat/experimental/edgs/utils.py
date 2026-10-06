import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from PIL import Image
import torch.nn.functional as F


def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff


def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices


def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    # Ensure input is a NumPy array
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    # Perform K-means clustering
    cluster_centers, _ = kmeans(cameras, K)

    # Assign each camera to a cluster and find distances to cluster centers
    cluster_assignments, _ = vq(cameras, cluster_centers)

    # Find the camera nearest to each cluster center
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices


def compute_warp_and_confidence(
    # viewpoint_cam1,
    # viewpoint_cam2,
    viewpoint_img1: Tensor,  # (3, H, W)
    viewpoint_img2: Tensor,  # (3, H, W)
    roma_model,
    device="cuda",
    verbose=False,
    output_dict={}
):
    """
    Computes the warp and confidence between two viewpoint cameras using the roma_model.

    Args:
        viewpoint_cam1: Source viewpoint camera.
        viewpoint_cam2: Target viewpoint camera.
        roma_model: Pre-trained Roma model for correspondence matching.
        device: Device to run the computation on.
        verbose: If True, displays the images.

    Returns:
        certainty: Confidence tensor.
        warp: Warp tensor.
        imB: Processed image B as numpy array.
    """
    # Prepare images
    # imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    # imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = viewpoint_img1.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    imB = viewpoint_img2.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        cax1 = ax[0].imshow(imA)
        ax[0].set_title("Image 1")
        cax2 = ax[1].imshow(imB)
        ax[1].set_title("Image 2")
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])
    
        for axis in ax:
            axis.axis('off')
        # Save the figure into the dictionary
        output_dict[f'image_pair'] = fig
   
    # Transform images
    ws, hs = roma_model.w_resized, roma_model.h_resized

    from romatch.utils import get_tuple_transform_ops
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((imA, imB))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    # Forward pass through Roma model
    corresps = roma_model.forward(batch) if not roma_model.symmetric else roma_model.forward_symmetric(batch)
    finest_scale = 1
    hs, ws = roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)

    # Process certainty and warp
    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty -= 0.5 * low_res_certainty * (low_res_certainty < 0)

    # Upsample predictions if needed
    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
        )

    # Convert predictions to final format
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        indexing='ij'
    ), dim=0).permute(1, 2, 0).unsqueeze(0).expand(im_A_to_im_B.size(0), -1, -1, -1)

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    certainty = certainty.sigmoid()

    return certainty[0, 0], warp[0], np.array(imB)


def resize_batch(tensors_3d, tensors_4d, target_shape):
    """
    Resizes a batch of tensors with shapes [B, H, W] and [B, H, W, 4] to the target spatial dimensions.

    Args:
        tensors_3d: Tensor of shape [B, H, W].
        tensors_4d: Tensor of shape [B, H, W, 4].
        target_shape: Tuple (target_H, target_W) specifying the target spatial dimensions.

    Returns:
        resized_tensors_3d: Tensor of shape [B, target_H, target_W].
        resized_tensors_4d: Tensor of shape [B, target_H, target_W, 4].
    """
    target_H, target_W = target_shape

    # Resize [B, H, W] tensor
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Resize [B, H, W, 4] tensor
    B, _, _, C = tensors_4d.shape
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return resized_tensors_3d, resized_tensors_4d


def aggregate_confidences_and_warps(
    # viewpoint_stack,
    viewpoints_img,
    # viewpoints_c2w,
    closest_indices,
    roma_model,
    source_idx,
    verbose=False,
    output_dict={}
):
    """
    Aggregates confidences and warps by iterating over the nearest neighbors of the source viewpoint.

    Args:
        viewpoint_stack: Stack of viewpoint cameras.
        closest_indices: Indices of the nearest neighbors for each viewpoint.
        roma_model: Pre-trained Roma model.
        source_idx: Index of the source viewpoint.
        verbose: If True, displays intermediate results.

    Returns:
        certainties_max: Aggregated maximum confidences.
        warps_max: Aggregated warps corresponding to maximum confidences.
        certainties_max_idcs: Pixel-wise index of the image  from which we taken the best matching.
        imB_compound: List of the neighboring images.
    """
    certainties_all, warps_all, imB_compound = [], [], []

    for nn in tqdm(closest_indices[source_idx]):

        # viewpoint_cam1 = viewpoint_stack[source_idx]
        # viewpoint_cam2 = viewpoint_stack[nn]
        viewpoint_img1 = viewpoints_img[source_idx]  # (3, H, W)
        viewpoint_img2 = viewpoints_img[nn]  # (3, H, W)

        certainty, warp, imB = compute_warp_and_confidence(
            # viewpoint_cam1,
            # viewpoint_cam2,
            viewpoint_img1,
            viewpoint_img2,
            roma_model,
            verbose=verbose,
            output_dict=output_dict
        )
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]
    if verbose: 
        print("certainties_all.shape:", certainties_all.shape)
        print("torch.stack(warps_all, dim=0).shape:", torch.stack(warps_all, dim=0).shape)
        print("target_shape:", target_shape)        

    certainties_all_resized, warps_all_resized = resize_batch(certainties_all,
                                                              torch.stack(warps_all, dim=0),
                                                              target_shape
                                                              )

    if verbose:
        print("warps_all_resized.shape:", warps_all_resized.shape)
        for n, cert in enumerate(certainties_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence")
            output_dict[f'certainty_{n}'] = fig

        for n, warp in enumerate(warps_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp")
            output_dict[f'warp_resized_{n}'] = fig

        for n, cert in enumerate(certainties_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence resized")
            output_dict[f'certainty_resized_{n}'] = fig

        for n, warp in enumerate(warps_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp resized")
            output_dict[f'warp_resized_{n}'] = fig

    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
    H, W = certainties_max.shape

    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

    # imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = viewpoint_img1.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    imA = np.clip(imA * 255, 0, 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized



def extract_keypoints_and_colors(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model,
                                 verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB_compound: List of target images as NumPy arrays [(H_B, W_B, C), ...].
        certainties_max: Tensor of pixel-wise maximum confidences.
        certainties_max_idcs: Tensor of pixel-wise indices for the best matches.
        matches: Matches in normalized coordinates.
        roma_model: Roma model instance for keypoint operations.
        verbose: if to show intermediate outputs and visualize results

    Returns:
        kptsA_np: Keypoints in imA in normalized coordinates.
        kptsB_np: Keypoints in imB in normalized coordinates.
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB based on certainties_max_idcs.
    """
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # Convert matches to pixel coordinates
    kptsA, kptsB = roma_model.to_pixel_coordinates(
        matches, W_A, H_A, H, W  # W, H
    )

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()
    kptsA_np = kptsA_np[:, [1, 0]]

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imA)
        ax.set_title("Reference image, imA")
        output_dict[f'reference_image'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imB_compound[0])
        ax.set_title("Image to compare to image, imB_compound")
        output_dict[f'imB_compound'] = fig
    
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imA))
        cax = ax.scatter(kptsA_np[:, 0], H_A - kptsA_np[:, 1], s=.03)
        ax.set_title("Keypoints in imA")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsA'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_compound[0]))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("Keypoints in imB")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsB'] = fig

    # Keypoints are in format (row, column) so the first value is alwain in range [0;height] and second is in range[0;width]

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    # Extract colors for keypoints in imA (vectorized)
    # New experimental version
    kptsA_x = np.round(kptsA_np[:, 0] / 1.).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1] / 1.).astype(int)
    kptsA_color = imA[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
   
    # Create a composite image from imB_compound
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    # Extract colors for keypoints in imB using certainties_max_idcs
    imB_np = imB_compound_np[
            certainties_max_idcs.detach().cpu().numpy(),
            np.arange(H).reshape(-1, 1),
            np.arange(W)
        ]
    
    if verbose:
        print("imB_np.shape:", imB_np.shape)
        print("imB_np:", imB_np)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_np))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("np.flipud(imB_np[0]")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'np.flipud(imB_np[0]'] = fig


    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsB_proj_matrices_idx = certainties_max_idcs_np[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, np.clip(kptsB_y, 0, H - 1), np.clip(kptsB_x, 0, W - 1)]

    # Normalize keypoints in both images
    kptsA_np[:, 0] = kptsA_np[:, 0] / H * 2.0 - 1.0
    kptsA_np[:, 1] = kptsA_np[:, 1] / W * 2.0 - 1.0
    kptsB_np[:, 0] = kptsB_np[:, 0] / W_B * 2.0 - 1.0
    kptsB_np[:, 1] = kptsB_np[:, 1] / H_B * 2.0 - 1.0

    return kptsA_np[:, [1, 0]], kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color

def prepare_tensor(input_array, device):
    """
    Converts an input array to a torch tensor, clones it, and detaches it for safe computation.
    Args:
        input_array (array-like): The input array to convert.
        device (str or torch.device): The device to move the tensor to.
    Returns:
        torch.Tensor: A detached tensor clone of the input array on the specified device.
    """
    if not isinstance(input_array, torch.Tensor):
        return torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
    return input_array.clone().detach().to(device).to(torch.float32)

def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """
    Solves for a batch of 3D points given batches of projection matrices and corresponding image points.

    Parameters:
    - P1, P2: Tensors of projection matrices of size (batch_size, 4, 4) or (4, 4)
    - k1_x, k1_y: Tensors of shape (batch_size,)
    - k2_x, k2_y: Tensors of shape (batch_size,)

    Returns:
    - X: A tensor containing the 3D homogeneous coordinates, shape (batch_size, 4)
    """
    EPS = 1e-4
    # Ensure inputs are tensors

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y =  prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    # Expand P1 and P2 if they are not batched
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0 = P1[:, :, 0]  # Shape: (batch_size, 4)
    P1_1 = P1[:, :, 1]
    P1_2 = P1[:, :, 2]

    P2_0 = P2[:, :, 0]
    P2_1 = P2[:, :, 1]
    P2_2 = P2[:, :, 2]

    # Reshape kx and ky to (batch_size, 1)
    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct the equations for each batch
    # For camera 1
    A1 = P1_0 - k1_x * P1_2  # Shape: (batch_size, 4)
    A2 = P1_1 - k1_y * P1_2
    # For camera 2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    # Stack the equations
    A = torch.stack([A1, A2, A3, A4], dim=1)  # Shape: (batch_size, 4, 4)

    # Right-hand side (constants)
    b = -A[:, :, 3]  # Shape: (batch_size, 4)
    A_reduced = A[:, :, :3]  # Coefficients of x, y, z

    # Solve using torch.linalg.lstsq (supports batching)
    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(2)  # Shape: (batch_size, 3)

    # Append 1 to get homogeneous coordinates
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # Shape: (batch_size, 4)

    # Now compute the errors of projections.
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (EPS + seeked_splats_proj1[:, [3]])
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (EPS + seeked_splats_proj2[:, [3]])
    proj1_target = torch.concat([k1_x, k1_y], dim=1)
    proj2_target = torch.concat([k2_x, k2_y], dim=1)
    errors_proj1 = torch.abs(seeked_splats_proj1[:, :2] - proj1_target).sum(1).detach().cpu().numpy()
    errors_proj2 = torch.abs(seeked_splats_proj2[:, :2] - proj2_target).sum(1).detach().cpu().numpy()

    return X, errors_proj1, errors_proj2



def select_best_keypoints(
        NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """
    From all the points fitted to  keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        NNs_triangulated_points:  torch tensor with keypoints coordinates (num_nns, num_points, dim). dim can be arbitrary,
            usually 3 or 4(for homogeneous representation).
        NNs_errors_proj1:  numpy array with projection error of the estimated keypoint on the reference frame (num_nns, num_points).
        NNs_errors_proj2:  numpy array with projection error of the estimated keypoint on the neighbor frame (num_nns, num_points).
    Returns:
        selected_keypoints: keypoints with the best score.
    """

    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)

    # Convert indices to PyTorch tensor
    indices = torch.from_numpy(np.argmin(NNs_errors_proj, axis=0)).long().to(device)

    # Create index tensor for the second dimension
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    # Use advanced indexing to select elements
    NNs_triangulated_points_selected = NNs_triangulated_points[indices, n_indices, :]  # Shape: [N, k]

    return NNs_triangulated_points_selected, np.min(NNs_errors_proj, axis=0)