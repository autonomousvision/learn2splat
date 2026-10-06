"""
https://github.com/google/dynibar/blob/main/ibrnet/data_loaders/llff_data_utils.py
"""

import numpy as np
import cv2


def render_stabilization_path(poses, k_size=45, start_idx=0, end_idx=None, loop=False):
    """Rendering stablizaed camera path."""

    # hwf = poses[0, :, 4:5]

    poses = poses[start_idx:end_idx]
    if loop:
        # Go back and forth
        poses = np.concatenate([poses, poses[::-1]], axis=0)

    num_frames = poses.shape[0]
    output_poses = []

    input_poses = []

    for i in range(num_frames):
        input_poses.append(
            np.concatenate(
                [poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], axis=-1
            )
        )

    input_poses = np.array(input_poses)

    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1)
    output_r1 = cv2.filter2D(input_poses[:, :, 0], -1, gaussian_kernel)
    output_r2 = cv2.filter2D(input_poses[:, :, 1], -1, gaussian_kernel)

    output_r1 = output_r1 / np.linalg.norm(output_r1, axis=-1, keepdims=True)
    output_r2 = output_r2 / np.linalg.norm(output_r2, axis=-1, keepdims=True)

    output_t = cv2.filter2D(input_poses[:, :, 2], -1, gaussian_kernel)

    for i in range(num_frames):
        output_r3 = np.cross(output_r1[i], output_r2[i])

        render_pose = np.concatenate(
            [
                output_r1[i, :, None],
                output_r2[i, :, None],
                output_r3[:, None],
                output_t[i, :, None],
            ],
            axis=-1,
        )

        output_poses.append(render_pose[:3, :])

    return output_poses



def render_looped_stabilization_path(
    poses,
    start_idx=0,
    num_poses=None,
    k_size=45,
    loop_frames=None
):
    """
    Smooth a subset of camera poses and optionally loop back to the starting pose.

    Args:
        poses (np.ndarray): Original poses of shape (N, 3, 4)
        start_idx (int): Index of the first pose to use
        num_poses (int): Number of poses to consider from start_idx
        k_size (int): Gaussian kernel size for smoothing
        loop_frames (int): If set, number of extra frames to interpolate back to start

    Returns:
        output_poses (np.ndarray): Smoothed (and looped) poses
    """
    # Slice poses
    if num_poses is None:
        selected_poses = poses[start_idx:]
    else:
        selected_poses = poses[start_idx:start_idx + num_poses]

    num_frames = selected_poses.shape[0]
    output_poses = []

    # Convert to columns for smoothing
    input_poses = []
    for i in range(num_frames):
        input_poses.append(
            np.concatenate(
                [selected_poses[i, :3, 0:1], selected_poses[i, :3, 1:2], selected_poses[i, :3, 3:4]], axis=-1
            )
        )
    input_poses = np.array(input_poses)

    # Gaussian smoothing
    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1)
    output_r1 = cv2.filter2D(input_poses[:, :, 0], -1, gaussian_kernel)
    output_r2 = cv2.filter2D(input_poses[:, :, 1], -1, gaussian_kernel)
    output_r1 /= np.linalg.norm(output_r1, axis=-1, keepdims=True)
    output_r2 /= np.linalg.norm(output_r2, axis=-1, keepdims=True)
    output_t = cv2.filter2D(input_poses[:, :, 2], -1, gaussian_kernel)

    # Build smoothed poses
    for i in range(num_frames):
        r3 = np.cross(output_r1[i], output_r2[i])
        pose = np.concatenate(
            [output_r1[i, :, None], output_r2[i, :, None], r3[:, None], output_t[i, :, None]], axis=-1
        )
        output_poses.append(pose[:3, :])

    output_poses = np.array(output_poses)

    # Optionally loop back to start
    if loop_frames is not None and loop_frames > 0:
        start_pose = output_poses[0]
        end_pose = output_poses[-1]
        looped_poses = []
        for i in range(1, loop_frames + 1):
            alpha = i / loop_frames
            # Linear interpolation for translation
            t_interp = (1 - alpha) * end_pose[:, 3] + alpha * start_pose[:, 3]
            # Linear interpolation + re-orthonormalize rotations
            r1_interp = (1 - alpha) * end_pose[:, 0] + alpha * start_pose[:, 0]
            r2_interp = (1 - alpha) * end_pose[:, 1] + alpha * start_pose[:, 1]
            r1_interp /= np.linalg.norm(r1_interp)
            r2_interp /= np.linalg.norm(r2_interp)
            r3_interp = np.cross(r1_interp, r2_interp)
            looped_pose = np.stack([r1_interp, r2_interp, r3_interp, t_interp], axis=-1)
            looped_poses.append(looped_pose)
        output_poses = np.concatenate([output_poses, np.array(looped_poses)], axis=0)

    return output_poses