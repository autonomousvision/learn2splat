import torch
from jaxtyping import Float
from torch import Tensor


def _normalize(x: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _focus_point(origins: Float[Tensor, "view 3"],
                 directions: Float[Tensor, "view 3"]) -> Float[Tensor, " 3"]:
    """The point closest to every camera's optical axis, in the least-squares sense.

    Each axis contributes (I - d d^T), which measures distance perpendicular to it; summing
    them and solving gives the point the cameras collectively look at.
    """
    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    perp = eye - directions[:, :, None] * directions[:, None, :]  # [V, 3, 3]
    a = perp.sum(dim=0)
    b = (perp @ origins[:, :, None]).sum(dim=0)  # [3, 1]
    return torch.linalg.lstsq(a, b).solution[:, 0]


def _rotation_about(axis: Float[Tensor, " 3"],
                    angles: Float[Tensor, " frame"]) -> Float[Tensor, "frame 3 3"]:
    """Rotations by `angles` about `axis` through the origin (Rodrigues' formula)."""
    cross = axis.new_tensor([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
    outer = axis[:, None] * axis[None, :]
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    cos, sin = torch.cos(angles)[:, None, None], torch.sin(angles)[:, None, None]
    return cos * eye + sin * cross + (1 - cos) * outer


def generate_orbit_path(
        extrinsics: Float[Tensor, "view 4 4"],
        anchor: Float[Tensor, "4 4"],
        num_frames: int,
        span_deg: float | int,
) -> Float[Tensor, "frame 4 4"]:
    """A turntable arc that starts at `anchor` and turns about the subject.

    Datasets whose views are not captured as a continuous walk (DTU's hemisphere grid, the
    every-8th target subset of a dense scene) give a jumpy video when rendered in index order.
    This replaces that order with one move: the anchor view carried rigidly around the vertical
    line through the point the cameras converge on, as though the scene were on a turntable.

    Turning about the scene's vertical rather than the anchor's own up axis is what keeps the
    move level. A view that looks down at its subject has an up axis tilted by that pitch, and
    a circle perpendicular to it dives under the subject on one side and climbs over it on the
    other. Here every frame keeps the anchor's height and its horizon.

    The first frame IS the anchor, exactly, since a rotation by zero is the identity. Playing
    the arc forward and then backward therefore leaves and returns to the view the rest of the
    video sits at, with no jump at either seam. The subject also holds its place in frame,
    because points on the axis do not move under a rotation about it.

    The arc turns one way only, toward whichever side the other cameras cover better, and stops
    at `span_deg` or at the last camera on that side, whichever comes first, so no frame lands
    where nothing was reconstructed. Angle and `num_frames` together set how fast the subject
    turns, since playback speed is fixed.
    """
    positions = extrinsics[:, :3, 3]                 # camera centers
    forward = _normalize(extrinsics[:, :3, 2])       # +Z looks into the screen
    # The cameras agree on which way is up (+Y points down in an OpenCV camera), and on the
    # point they all look at; together those give the axis the turntable spins about.
    up = _normalize(_normalize(-extrinsics[:, :3, 1]).mean(dim=0))
    center = _focus_point(positions, forward)

    # Angles are measured about that axis, from the anchor, so the anchor sits at 0.
    def _in_plane(points):
        offsets = points - center
        return offsets - (offsets @ up)[..., None] * up

    axis_1 = _in_plane(anchor[None, :3, 3])[0]
    if axis_1.norm() < 1e-6:
        raise ValueError("the anchor view sits on the axis the scene turns about, so there is "
                         "no circle for it to travel along")
    axis_1 = _normalize(axis_1)
    axis_2 = torch.linalg.cross(up, axis_1)
    around = _in_plane(positions)
    angles = torch.atan2(around @ axis_2, around @ axis_1)

    # Turn toward the better-covered side, no further than the last camera on it.
    reach_positive = angles[angles > 0].max() if (angles > 0).any() else angles.new_zeros(())
    reach_negative = -angles[angles < 0].min() if (angles < 0).any() else angles.new_zeros(())
    direction = 1.0 if reach_positive >= reach_negative else -1.0
    span = torch.minimum(torch.maximum(reach_positive, reach_negative),
                         angles.new_tensor(span_deg * torch.pi / 180))

    theta = direction * span * torch.linspace(0, 1, num_frames, dtype=angles.dtype,
                                              device=angles.device)
    rotation = _rotation_about(up, theta)            # [F, 3, 3]

    poses = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    poses = poses.repeat(num_frames, 1, 1)
    poses[:, :3, :3] = rotation @ anchor[:3, :3]
    poses[:, :3, 3] = center + (rotation @ (anchor[:3, 3] - center)[:, None])[:, :, 0]
    return poses
