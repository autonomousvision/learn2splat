from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
# from lpips import LPIPS
# from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


@torch.no_grad()
def compute_psnr(
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, ""]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    # Use native torch ops instead of einops reduce for speed
    mse = ((ground_truth - predicted) ** 2).mean(dim=(1, 2, 3))  # [b]
    return -10 * mse.log10().mean()


@cache
def get_alex_lpips(device: torch.device) -> LearnedPerceptualImagePatchSimilarity:
    return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True, reduction='none').to(device)


@cache
def get_vgg_lpips(device: torch.device) -> LearnedPerceptualImagePatchSimilarity:
    return LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True, reduction='none').to(device)


@torch.no_grad()
def compute_lpips(
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    predicted = torch.clamp(predicted, 0.0, 1.0)
    ground_truth = torch.clamp(ground_truth, 0.0, 1.0)
    vgg_value = get_vgg_lpips(predicted.device)(ground_truth, predicted).mean()
    # Note: skipping alex lpips for efficiency, always return 0.
    # alex_value = get_alex_lpips(predicted.device)(ground_truth, predicted)
    alex_value = torch.zeros_like(vgg_value).mean()
    return alex_value, vgg_value


@cache
def get_ssim(device: torch.device) -> StructuralSimilarityIndexMeasure:
    return StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)


@torch.no_grad()
def compute_ssim(
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, ""]:
    predicted = torch.clamp(predicted, 0.0, 1.0)
    ground_truth = torch.clamp(ground_truth, 0.0, 1.0)
    ssim_value = get_ssim(predicted.device)(predicted, ground_truth).mean()
    return ssim_value


metric_fn_dict = {
    "psnr": compute_psnr,
    "ssim": compute_ssim,
    "lpips": compute_lpips,
}


def compute_rgb_metrics(rgb, rgb_gt, metrics: list[str], iter_batch_size: int = -1) -> dict:
    metric_scores = {}
    for m in metrics:

        # check if metric is recognized
        if m not in metric_fn_dict:
            raise ValueError(f"Metric {m} not recognized. Available metrics: {list(metric_fn_dict.keys())}")

        # compute metric score
        if iter_batch_size == -1:
            # compute all at once
            # move back to device
            rgb = rgb.to("cuda")
            rgb_gt = rgb_gt.to("cuda")
            score = metric_fn_dict[m](rgb_gt, rgb)
            # can be tuple (for lpips) or single tensor

        else:
            # batchify to save memory
            all_batches_scores = []
            batch_sizes = []

            batch_num = rgb.shape[0] // iter_batch_size + int(rgb.shape[0] % iter_batch_size != 0)
            for i in tqdm(range(0, rgb.shape[0], iter_batch_size), disable=batch_num < 20,
                          desc=f"Computing {m} in batches"):
                bs = min(iter_batch_size, rgb.shape[0] - i)
                rgb_batch = rgb[i:i + bs].to("cuda")
                rgb_gt_batch = rgb_gt[i:i + bs].to("cuda")
                batch_scores = metric_fn_dict[m](rgb_gt_batch, rgb_batch)
                # can be tuple (for lpips) or single tensor

                all_batches_scores.append(batch_scores)
                batch_sizes.append(bs)

            assert len(all_batches_scores) > 0, "No batch scores computed."

            # Use weighted mean to avoid bias when the last batch is smaller than iter_batch_size.
            # Each batch score is the mean over `bs` images, so we weight by bs to recover the
            # true per-image mean across all N images.
            weights = torch.tensor(batch_sizes, dtype=torch.float32)
            total = weights.sum()
            first = all_batches_scores[0]

            # Case 1: scalar tensors
            if isinstance(first, torch.Tensor):
                vals = torch.stack(all_batches_scores).cpu()
                score = (vals * weights).sum() / total

            # Case 2: tuples of tensors
            elif isinstance(first, tuple):
                n = len(first)
                cols = [torch.stack([batch[i] for batch in all_batches_scores]).cpu() for i in range(n)]
                score = tuple((col * weights).sum() / total for col in cols)

            else:
                raise TypeError("Unexpected element type: must be torch.Tensor or tuple of torch.Tensors.")

        # append to scores list
        metric_scores[m] = score

    return metric_scores
