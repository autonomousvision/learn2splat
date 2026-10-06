"""
Fused KNN Gather + Attention kernel.

Replaces the two-step gather-then-attend pattern in KNNAttention with a single
CUDA kernel that never materializes the [N, K, C] intermediate tensors.

Usage:
    from .fused_knn_attn import fused_knn_attention  # autograd Function

    # Same result as the unfused version but faster and less memory
    out = fused_knn_attention(q, k, v, knn_idx, scale)
"""

import torch
from torch.autograd import Function

# Try to import the compiled CUDA extension; fall back to pure-PyTorch
try:
    import fused_knn_attn_cuda as _C
    FUSED_KNN_ATTN_CUDA_AVAILABLE = True
except ImportError:
    FUSED_KNN_ATTN_CUDA_AVAILABLE = False


class FusedKNNAttentionFunction(Function):
    """Autograd function for fused KNN gather + scaled dot-product attention.

    Forward:
        q, k, v: [N, C]  (query, key, value features for all points)
        idx:     [N, K]  (pre-computed KNN indices, int32)
        scale:   float   (attention scale factor, typically head_dim ** -0.5)

    Returns:
        out: [N, C]
    """

    @staticmethod
    def forward(ctx, q, k, v, idx, scale):
        # Ensure contiguous float32 for the CUDA kernel
        q = q.contiguous().float()
        k = k.contiguous().float()
        v = v.contiguous().float()
        idx = idx.contiguous().int()

        N, C = q.shape
        num_k = idx.shape[1]

        out = torch.empty((N, C), dtype=torch.float32, device=q.device)
        attn_weights = torch.empty((N, num_k), dtype=torch.float32, device=q.device)

        _C.fused_knn_attn_forward_cuda(
            q, k, v, idx, out, attn_weights,
            N, C, num_k, float(scale)
        )

        ctx.save_for_backward(q, k, v, idx, attn_weights)
        ctx.scale = scale
        ctx.N = N
        ctx.C = C
        ctx.num_k = num_k

        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, idx, attn_weights = ctx.saved_tensors
        scale = ctx.scale
        N, C, num_k = ctx.N, ctx.C, ctx.num_k

        grad_out = grad_out.contiguous().float()

        grad_q = torch.zeros((N, C), dtype=torch.float32, device=q.device)
        grad_k = torch.zeros((N, C), dtype=torch.float32, device=q.device)
        grad_v = torch.zeros((N, C), dtype=torch.float32, device=q.device)

        _C.fused_knn_attn_backward_cuda(
            grad_out, q, k, v, idx, attn_weights,
            grad_q, grad_k, grad_v,
            N, C, num_k, float(scale)
        )

        # idx and scale don't need gradients
        return grad_q, grad_k, grad_v, None, None


class FusedKNNAttentionFunctionPyTorch(Function):
    """Pure-PyTorch fallback (same semantics, no CUDA extension required).

    Avoids materializing full [N, K, C] by iterating over K neighbors.
    Still faster than the original due to not creating [N, K, C] tensors,
    but slower than the CUDA kernel.
    """

    @staticmethod
    def forward(ctx, q, k, v, idx, scale):
        N, C = q.shape
        num_k = idx.shape[1]

        # Compute scores by iterating over neighbors (avoids [N, K, C] tensor)
        scores = torch.empty(N, num_k, device=q.device, dtype=q.dtype)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()  # [N]
            k_neighbor = k[neighbor_idx]      # [N, C]
            scores[:, kk] = (q * k_neighbor).sum(dim=-1) * scale

        attn_weights = torch.softmax(scores, dim=-1)  # [N, K]

        # Compute output
        out = torch.zeros(N, C, device=q.device, dtype=q.dtype)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()
            v_neighbor = v[neighbor_idx]       # [N, C]
            out += attn_weights[:, kk:kk+1] * v_neighbor

        ctx.save_for_backward(q, k, v, idx, attn_weights)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, idx, attn_weights = ctx.saved_tensors
        scale = ctx.scale
        N, C = q.shape
        num_k = idx.shape[1]

        # grad_attn[k] = dot(grad_out, V[idx[:, k]])
        grad_attn = torch.empty(N, num_k, device=q.device, dtype=q.dtype)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()
            v_neighbor = v[neighbor_idx]
            grad_attn[:, kk] = (grad_out * v_neighbor).sum(dim=-1)

        # Softmax backward: grad_scores = attn * (grad_attn - sum(attn * grad_attn))
        ds = (attn_weights * grad_attn).sum(dim=-1, keepdim=True)  # [N, 1]
        grad_scores = attn_weights * (grad_attn - ds)  # [N, K]

        # grad_Q
        grad_q = torch.zeros(N, C, device=q.device, dtype=q.dtype)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()
            k_neighbor = k[neighbor_idx]
            grad_q += grad_scores[:, kk:kk+1] * k_neighbor * scale

        # grad_K (scatter add)
        grad_k = torch.zeros_like(k)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()
            contrib = grad_scores[:, kk:kk+1] * q * scale  # [N, C]
            grad_k.index_add_(0, neighbor_idx, contrib)

        # grad_V (scatter add)
        grad_v = torch.zeros_like(v)
        for kk in range(num_k):
            neighbor_idx = idx[:, kk].long()
            contrib = attn_weights[:, kk:kk+1] * grad_out  # [N, C]
            grad_v.index_add_(0, neighbor_idx, contrib)

        return grad_q, grad_k, grad_v, None, None


def fused_knn_attention(q, k, v, idx, scale):
    """Fused KNN gather + attention. Uses CUDA kernel if available, else PyTorch fallback.

    Args:
        q: [N, C] query features
        k: [N, C] key features
        v: [N, C] value features
        idx: [N, K] pre-computed KNN neighbor indices (int32)
        scale: attention scale factor

    Returns:
        out: [N, C] attention output
    """
    if FUSED_KNN_ATTN_CUDA_AVAILABLE and q.is_cuda:
        return FusedKNNAttentionFunction.apply(q, k, v, idx, scale)
    else:
        return FusedKNNAttentionFunctionPyTorch.apply(q, k, v, idx, scale)
