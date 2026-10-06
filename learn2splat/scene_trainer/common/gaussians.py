import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def rotation_matrix_to_quaternion_xyzw(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in xyzw format.

    Args:
        R: Tensor of shape [..., 3, 3] (or any batch shape ending with (3,3)).
        eps: small number for numerical stability.

    Returns:
        q: Tensor of shape [..., 4], quaternion order [x, y, z, w].
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected last two dimensions (3,3), got {R.shape[-2:]}")

    # Force float (preserve device)
    dtype = R.dtype if torch.is_floating_point(R) else torch.float32
    R = R.to(dtype=dtype)

    m00 = R[..., 0, 0]; m01 = R[..., 0, 1]; m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]; m11 = R[..., 1, 1]; m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]; m21 = R[..., 2, 1]; m22 = R[..., 2, 2]

    trace = m00 + m11 + m22

    # conditions (shape: [...])
    cond1 = trace > 0.0
    cond2 = (m00 >= m11) & (m00 >= m22) & (~cond1)
    cond3 = (m11 > m22) & (~cond1) & (~cond2)
    # cond4 is the remaining case
    cond4 = ~(cond1 | cond2 | cond3)

    # Candidate 1 (trace > 0): s = 4*w
    s1 = (trace + 1.0).clamp_min(eps).sqrt() * 2.0
    inv_s1 = 1.0 / s1
    q1_x = (m21 - m12) * inv_s1
    q1_y = (m02 - m20) * inv_s1
    q1_z = (m10 - m01) * inv_s1
    q1_w = 0.25 * s1
    q1 = torch.stack([q1_x, q1_y, q1_z, q1_w], dim=-1)

    # Candidate 2 (m00 largest): s = 4*x
    s2 = (1.0 + m00 - m11 - m22).clamp_min(eps).sqrt() * 2.0
    inv_s2 = 1.0 / s2
    q2_x = 0.25 * s2
    q2_y = (m01 + m10) * inv_s2
    q2_z = (m02 + m20) * inv_s2
    q2_w = (m21 - m12) * inv_s2
    q2 = torch.stack([q2_x, q2_y, q2_z, q2_w], dim=-1)

    # Candidate 3 (m11 largest): s = 4*y
    s3 = (1.0 + m11 - m00 - m22).clamp_min(eps).sqrt() * 2.0
    inv_s3 = 1.0 / s3
    q3_x = (m01 + m10) * inv_s3
    q3_y = 0.25 * s3
    q3_z = (m12 + m21) * inv_s3
    q3_w = (m02 - m20) * inv_s3
    q3 = torch.stack([q3_x, q3_y, q3_z, q3_w], dim=-1)

    # Candidate 4 (m22 largest): s = 4*z
    s4 = (1.0 + m22 - m00 - m11).clamp_min(eps).sqrt() * 2.0
    inv_s4 = 1.0 / s4
    q4_x = (m02 + m20) * inv_s4
    q4_y = (m12 + m21) * inv_s4
    q4_z = 0.25 * s4
    q4_w = (m10 - m01) * inv_s4
    q4 = torch.stack([q4_x, q4_y, q4_z, q4_w], dim=-1)

    # Broadcast masks over last dimension and select candidates
    cond1_u = cond1.unsqueeze(-1)
    cond2_u = cond2.unsqueeze(-1)
    cond3_u = cond3.unsqueeze(-1)
    # nested where ensures every position picks exactly one candidate
    q = torch.where(cond1_u, q1,
                    torch.where(cond2_u, q2,
                                torch.where(cond3_u, q3, q4)))

    # Normalize to unit quaternion
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

    return q



# # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_multiply_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product a*b of xyzw quaternions (R(a*b) = R(a) R(b)); keeps the norm of unnormalized inputs."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )
