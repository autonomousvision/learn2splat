from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor
from learn2splat.model.types import Gaussians
from learn2splat.scene_trainer.gaussian_module import GaussiansModule


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    # extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian"],
    path: Path,
    # align_to_view: bool = False,  # whether to align world space to the view space (camera space) of the extrinsics
):
    means = means.detach().cpu().numpy()
    scales = scales.log().detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()
    harmonics = harmonics.detach() # .cpu().numpy()
    opacities = torch.logit(opacities[..., None]).detach().cpu().numpy()

    num_rest = 3 * (harmonics.shape[-1] - 1)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(num_rest)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means,
        np.zeros_like(means),
        harmonics[..., 0].cpu().numpy(),
        harmonics[..., 1:].flatten(start_dim=1).cpu().numpy(),
        opacities,
        scales,
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(
    gaussians: Gaussians | GaussiansModule,
    save_path,
):
    """
    Save Gaussians to a .ply file for visualization.

    The saved object will have opacities and scales in the pre-activation space,
    i.e., before applying the activation functions (sigmoid for opacity, exp for scales).

    """

    if isinstance(gaussians, GaussiansModule):
        
        # no batch dimension
        means = gaussians.means  # [H*W, 3]
        rotations = gaussians.rotations  # [H*W, 4] in xyzw
        scales = gaussians.scales  # [H*W, 3]
        opacities = gaussians.opacities  # [H*W]
        harmonics = gaussians.harmonics  # [H*W, 3, d_sh]
    
    elif isinstance(gaussians, Gaussians):
        assert gaussians.means.shape[0] == 1, "Batch size > 1 not supported for saving ply."
        means = gaussians.means[0]  # [H*W, 3]
        rotations = F.normalize(gaussians.rotations_unnorm[0], dim=-1)  # [H*W, 4] in xyzw
        scales = gaussians.scales[0]  # [H*W, 3]
        opacities = gaussians.opacities[0]  # [H*W]
        harmonics = gaussians.harmonics[0]  # [H*W, 3, d_sh]

        # export_ply expects activated values (post-exp scales, post-sigmoid opacities)
        # and applies inverse activation internally. If values are already deactivated,
        # we must activate them first to avoid double inverse activation.
        if not gaussians.stores_activated:
            scales = torch.exp(scales)
            opacities = torch.sigmoid(opacities)
    else:
        raise ValueError(f"Unknown type of gaussians: {type(gaussians)}")

    # convert to wxyz for saving
    rotations = rotations[:, [3, 0, 1, 2]]  # [H*W, 4] in wxyz

    # This fn invert activation of opacity and scales (for standard gaussian object, loaded by viewer)
    export_ply(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,  # [H*W, 3, d_sh]
        opacities=opacities,
        path=save_path,    
    )


def load_gaussians_ply(path, max_sh_degree=3) -> Gaussians:
    """ Load Gaussians from a .ply file saved by export_ply().
    The loaded object will have opacities and scales in the pre-activation space,
    i.e., before applying the activation functions (sigmoid for opacity, expfor scales).

    """
    
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # 
    if len(extra_f_names) == 0:
        # SH degree 0: only the DC band is present, no higher-order coefficients.
        # f_dc still holds the SH DC term (SH0), not raw RGB -- the 3DGS .ply
        # convention stores SH0, and the colour is recovered as rgb = 0.5 + C0 * SH0.
        # So f_dc is read directly into harmonics[..., 0]; the remaining bands are zero.
        print("Loaded PLY has no SH coefficients, only DC features.")
        features_extra = np.zeros((xyz.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    
    elif len(extra_f_names) == (3 * (max_sh_degree + 1) ** 2 - 3):
        # loaded ply has full SH coefficients
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    else:
        # not know how to handle
        raise ValueError("Mismatch in number of SH coefficients.")

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Create Gaussian object
    means = torch.tensor(xyz, dtype=torch.float32) # [P, 3]

    opacities = torch.tensor(opacities, dtype=torch.float32).squeeze(-1) # [P]
    opacities = torch.sigmoid(opacities)  # convert to post-activation space

    harmonics = torch.zeros((xyz.shape[0], 3, (max_sh_degree + 1) ** 2), dtype=torch.float32)  # [P, 3, d_sh]
    harmonics[:, :, 0] = torch.tensor(features_dc[:, :, 0], dtype=torch.float32)
    harmonics[:, :, 1:] = torch.tensor(features_extra, dtype=torch.float32)

    scales = torch.tensor(scales, dtype=torch.float32)
    scales = torch.exp(scales) # convert to post-activation space

    quats = torch.tensor(rots, dtype=torch.float32)  # in wxyz
    quats = quats[:, [1, 2, 3, 0]]  # convert to xyzw
    quats = F.normalize(quats, dim=-1)  # the rasterizers expect unit quaternions

    return Gaussians(
            means=means.unsqueeze(0),
            harmonics=harmonics.unsqueeze(0),
            opacities=opacities.unsqueeze(0),
            scales=scales.unsqueeze(0),
            rotations=quats.unsqueeze(0),
            rotations_unnorm=quats.unsqueeze(0),
        )
