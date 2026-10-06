import torch


def get_smooth_loss(disp, img, no_mean=False):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    ref: https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L202
    """
    if no_mean:
        out = torch.zeros_like(disp)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if no_mean:
        out[:, :, :, :-1] = out[:, :, :, :-1] + grad_disp_x
        out[:, :, :-1, :] = out[:, :, :-1, :] + grad_disp_y

        return out

    return grad_disp_x.mean() + grad_disp_y.mean()

    