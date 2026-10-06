import torch


def batched_select(data, indices):
    """
    Select data[i, indices[i]] for each batch element i.

    Args:
        data: [B, N, ...] input tensor
        indices: [B, K] indices for each batch element

    """

    assert data.shape[0] == indices.shape[0], f"Batch size mismatch {data.shape[0]} vs {indices.shape[0]}"
    assert indices.dim() == 2, f"indices should be 2D, got {indices.shape}"

    B = data.shape[0]
    batch_idx = torch.arange(B, device=data.device)[:, None]
    return data[batch_idx, indices]
