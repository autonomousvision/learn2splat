from pathlib import Path
from typing import Union
# import skvideo.io
import imageio
import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from learn2splat.misc.io import CustomPath

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    # Round-half-up, to match torchvision.utils.save_image.
    image = (image.detach().clip(min=0, max=1) * 255 + 0.5).clip(0, 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


# def save_video(
#     images: list[FloatImage],
#     path: Union[Path, str],
#     fps: None | int = None
# ) -> None:
#     """Save an image. Assumed to be in range 0-1."""

#     # Create the parent directory if it doesn't already exist.
#     path = Path(path)
#     path.parent.mkdir(exist_ok=True, parents=True)

#     # prepare frames as uint8 HxWx3 numpy arrays in range 0-255
#     frames = [prep_image(img) for img in images]

#     outputdict = {'-pix_fmt': 'yuv420p', '-crf': '23', '-vf': 'setpts=1.*PTS'}
#     if fps is not None:
#         outputdict['-r'] = str(fps)

#     # pass a string filename
#     writer = skvideo.io.FFmpegWriter(str(path), outputdict=outputdict)
#     for frame in frames:
#         writer.writeFrame(frame)
#     writer.close()

def save_video(images, path, fps=None, iterations=None):
    if len(images) < 3:
        return
    if iterations is not None:
        assert len(images) == len(iterations)
    
    path = CustomPath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fps is None:
        fps = 30
    # ensure frames are uint8
    frames = [
        np.ascontiguousarray(prep_image(img).clip(0, 255).astype("uint8"))
        for img in images
    ]
    # write iteration number on each frame if given
    if iterations is not None:
        for i in range(len(frames)):
            frame = frames[i]
            iter_num = iterations[i]
            cv2.putText(frame, f"Iter {iter_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frames[i] = frame

    # NOTE: the encoder cannot save videos with odd frame dimensions
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)