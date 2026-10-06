import torch
import torch.utils.data
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm


# https://github.com/autonomousvision/unimatch/blob/master/utils/visualization.py


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def viz_depth_tensor(disp, return_numpy=False, colormap="plasma", as_uint8=True, vmin=None, vmax=None):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)
    device = disp.device

    disp = disp.cpu().numpy()
    if vmin is None:
        vmin = disp.min()
    if vmax is None:
        vmax = disp.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = mapper.to_rgba(disp)[:, :, :3]
    if as_uint8:
        colormapped_im = (colormapped_im * 255).astype(np.uint8)
    else:
        colormapped_im = (colormapped_im).astype(np.float32)

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]
    viz = viz.to(device)

    return viz
