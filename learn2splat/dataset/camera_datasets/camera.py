
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from pathlib import Path
import os
import json

from learn2splat.geometry.projection import get_fov, get_projection_matrix


def get_scene_scale(camtoworlds: Float[np.ndarray, "N 4 4"]) -> float:
    # camtoworlds: [N, 4, 4]
    # size of the scene measured by cameras as in gsplat
    camera_locations = camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    dists = np.linalg.norm(camera_locations - scene_center, axis=1)
    scene_scale = np.max(dists)
    return float(scene_scale) * 1.1


class Camera(nn.Module):
    """
    A camera class that stores the camera parameters and the image for Re10k dataset.

    Attributes:
        image_name:
        extrinsics: C2W matrix (4x4 torch.Tensor)
        intrinsics: K matrix (3x3 torch.Tensor)
        near: Near clipping plane distance
        far: Far clipping plane distance
        image: RGB image (3xHxW torch.Tensor)
        fov_x: Field of view in x direction
        fov_y: Field of view in y direction
        image_heigth: Height of the image
        image_width: Width of the image
        view_matrix: View matrix (4x4 torch.Tensor)
        full_projection_matrix: Full projection matrix (4x4 torch.Tensor)
        camera_center: Camera center (3 torch.Tensor)
    """
    def __init__(
            self, 
            colmap_id: str, 
            extrinsics: Float[Tensor, "4 4"], 
            intrinsics: Float[Tensor, "3 3"], 
            extrinsics_render_view: Float[Tensor, "4 4"],
            intrinsics_render_view: Float[Tensor, "3 3"],
            scale_matrix: Float[Tensor, "4 4"],
            trans_matrix: Float[Tensor, "4 4"],
            image: Float[Tensor, "3 h w"],
            raw_image_shape: tuple[int, int],
            image_name: str, 
            uid: int, 
            near: Float[Tensor, "1"],
            far: Float[Tensor, "1"],
            data_device: torch.device,
            gt_alpha_mask: Float[Tensor, "1 h w"] | None = None,
            trans=np.array([0.0, 0.0, 0.0]), 
            scale=1.0
        ):
        super(Camera, self).__init__()

        self.idx = -1
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_name = image_name

        try:
            self.data_device = data_device
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.extrinsics = extrinsics.to(self.data_device)   # C2W matrix! (not really extrinsics)
        self.intrinsics = intrinsics.to(self.data_device)
        self.extrinsics_render_view = extrinsics_render_view.to(self.data_device)
        self.intrinsics_render_view = intrinsics_render_view.to(self.data_device)
        self.scale_matrix = scale_matrix.to(self.data_device)
        self.trans_matrix = trans_matrix.to(self.data_device)

        self.raw_image_shape = raw_image_shape

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = far.to(self.data_device)
        self.znear = near.to(self.data_device)

        self.trans = trans
        self.scale = scale

        fov_x, fov_y = get_fov(self.intrinsics.unsqueeze(0)).unbind(dim=-1)

        self.FoVx = fov_x.item()
        self.FoVy = fov_y.item()

        projection_matrix = get_projection_matrix(self.znear, self.zfar, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(self.extrinsics.inverse(), "i j -> j i")
        full_projection = (view_matrix.unsqueeze(0) @ projection_matrix)[0]

        self.camera_center = self.extrinsics[:3, 3]
        self.projection_matrix = projection_matrix[0].transpose(0, 1)
        self.world_view_transform = view_matrix
        self.full_proj_transform = full_projection

    def save(self, save_dir: Path):
        cam_dir = save_dir / self.image_name
        os.makedirs(cam_dir, exist_ok=True)

        torch.save(self.extrinsics, cam_dir / "extrinsics.pt")
        torch.save(self.intrinsics, cam_dir / "intrinsics.pt")
        torch.save(self.original_image, cam_dir / "image.pt")

        if self.gt_alpha_mask is not None:
            torch.save(self.gt_alpha_mask, cam_dir / "gt_alpha_mask.pt")

        with open(cam_dir / "cam_info.json", "w") as f:
            json.dump(
                {
                    "colmap_id": self.colmap_id,
                    "image_name": self.image_name,
                    "uid": self.uid,
                    "raw_image_shape": self.raw_image_shape,
                    "near": self.znear.item(),
                    "far": self.zfar.item()
                },
                f,
                indent=4,
            )

def patch_shim(cams: list[Camera], patch_size: int) -> list[Camera]:
    new_cams = []

    for cam in cams:
        _, h, w = cam.original_image.shape

        assert h % 2 == 0 and w % 2 == 0

        h_new = (h // patch_size) * patch_size
        row = (h - h_new) // 2
        w_new = (w // patch_size) * patch_size
        col = (w - w_new) // 2

        # Center-crop the image.
        new_original_image = cam.original_image[:, row : row + h_new, col : col + w_new]

        # Adjust the intrinsics to account for the cropping.
        new_intrinsics = cam.intrinsics.clone()
        new_intrinsics[0, 2] -= col
        new_intrinsics[1, 2] -= row

        # Adjust the intrinsics to account for the cropping.
        new_render_view_intrinsics = cam.intrinsics_render_view.clone()
        new_render_view_intrinsics[0] -= col
        new_render_view_intrinsics[1] -= row

        new_cams.append(
            Camera(
                colmap_id=cam.colmap_id,
                image_name=cam.image_name,
                uid=cam.uid,
                near=cam.znear,
                far=cam.zfar,
                data_device=cam.data_device,
                raw_image_shape=cam.raw_image_shape,
                image=new_original_image,
                extrinsics=cam.extrinsics,
                intrinsics=new_intrinsics,
                extrinsics_render_view=cam.extrinsics_render_view,
                intrinsics_render_view=new_render_view_intrinsics,
                scale_matrix=cam.scale_matrix,
                trans_matrix=cam.trans_matrix,
                gt_alpha_mask=cam.gt_alpha_mask
            )
        )
    
    return new_cams


def save_cameras(cameras: list[Camera], save_dir: Path):
    os.makedirs(save_dir, exist_ok=True)

    extrinsics = torch.stack([cam.extrinsics for cam in cameras])
    intrinsics = torch.stack([cam.intrinsics for cam in cameras])
    images = torch.stack([cam.original_image for cam in cameras])

    torch.save(extrinsics, save_dir / "extrinsics.pt")
    torch.save(intrinsics, save_dir / "intrinsics.pt")
    torch.save(images, save_dir / "images.pt")
    
    if cameras[0].gt_alpha_mask is not None:
        gt_alpha_masks = torch.stack([cam.gt_alpha_mask for cam in cameras])
        torch.save(gt_alpha_masks, save_dir / "gt_alpha_masks.pt")

    with open(save_dir / "cam_info.json", "w") as f:
        json.dump(
            {
                "num_cameras": len(cameras),
                "image_shape": [(cam.image_height, cam.image_width) for cam in cameras],
                "znear": [cam.znear.item() for cam in cameras],
                "zfar": [cam.zfar.item() for cam in cameras],
                "uids": [cam.uid for cam in cameras],
                "colmap_ids": [cam.colmap_id for cam in cameras],
                "raw_image_shapes": [cam.raw_image_shape for cam in cameras],
            },
            f,
            indent=4,
        )

def load_cameras(cam_dir: Path, device: torch.device) -> list[Camera]:
    cameras = []

    extrinsics = torch.load(cam_dir / "extrinsics.pt")
    intrinsics = torch.load(cam_dir / "intrinsics.pt")
    images = torch.load(cam_dir / "images.pt")

    if (cam_dir / "gt_alpha_masks.pt").exists():
        gt_alpha_masks = torch.load(cam_dir / "gt_alpha_masks.pt")
    else:
        gt_alpha_masks = [None] * len(images)

    with open(cam_dir / "cam_info.json", "r") as f:
        cam_info = json.load(f)

    for idx in range(cam_info["num_cameras"]):
        cameras.append(
            Camera(
                colmap_id=cam_info["colmap_ids"][idx],
                image_name=f"image_{idx:04d}",
                uid=cam_info["uids"][idx],
                near=torch.Tensor([cam_info["znear"][idx]]).to(device),
                far=torch.Tensor([cam_info["zfar"][idx]]).to(device),
                data_device=device,
                image=images[idx].to(device),
                extrinsics=extrinsics[idx].to(device),
                intrinsics=intrinsics[idx].to(device),
                raw_image_shape=tuple(cam_info["raw_image_shapes"][idx]),
                gt_alpha_mask=gt_alpha_masks[idx].to(device) if gt_alpha_masks[idx] is not None else None
            )
        )
    
    return cameras
    