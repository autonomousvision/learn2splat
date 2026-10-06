/*
 * Fused KNN Gather + Scaled Dot-Product Attention CUDA Kernel
 *
 * Fuses the two separate gather operations (for K and V) and the attention
 * computation into a single kernel, avoiding materialization of [N, K, C]
 * intermediate tensors.
 *
 * Forward:
 *   Given Q [N, C], K [N, C], V [N, C], idx [N, num_k]:
 *   For each query point n:
 *     1. Gather K[idx[n, :]] and compute scores = Q[n] . K[neighbor] * scale
 *     2. Softmax over scores
 *     3. Gather V[idx[n, :]] and compute out = sum_k attn[k] * V[neighbor_k]
 *
 * Backward:
 *   Given grad_out [N, C], saved Q, K, V, idx, attn_weights:
 *   Computes grad_Q [N, C], grad_K [N, C], grad_V [N, C]
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#define THREADS_PER_BLOCK 256
#define MAX_K_NEIGHBORS 64

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val, float* shared, int tid, int block_size) {
    int lane = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    val = (tid < num_warps) ? shared[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;  // result valid in thread 0
}

// ============================================================================
// FORWARD KERNEL
// ============================================================================
// One block per query point. Threads cooperate across the C dimension.
// Shared memory: scores[K] + attn[K] + reduction_buf[num_warps]

__global__ void fused_knn_attn_forward_kernel(
    const float* __restrict__ q,            // [N, C]
    const float* __restrict__ k,            // [N, C]
    const float* __restrict__ v,            // [N, C]
    const int*   __restrict__ idx,          // [N, num_k]
    float*       __restrict__ out,          // [N, C]
    float*       __restrict__ attn_weights, // [N, num_k] saved for backward
    const int N,
    const int C,
    const int num_k,
    const float scale
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory: scores[num_k] + attn[num_k] + reduction_buf[ceil(block_size/32)]
    extern __shared__ float smem[];
    float* scores = smem;                           // [num_k]
    float* attn   = scores + num_k;                 // [num_k]
    float* reduce_buf = attn + num_k;               // [ceil(block_size/32)]

    const float* q_n   = q + (long long)n * C;
    const int*   idx_n = idx + (long long)n * num_k;

    // ------- Step 1: Compute attention scores -------
    for (int kk = 0; kk < num_k; kk++) {
        int neighbor = idx_n[kk];
        const float* k_neighbor = k + (long long)neighbor * C;

        // Dot product Q[n] . K[neighbor] over C, distributed across threads
        float partial = 0.0f;
        for (int c = tid; c < C; c += block_size) {
            partial += q_n[c] * k_neighbor[c];
        }

        float dot = block_reduce_sum(partial, reduce_buf, tid, block_size);
        if (tid == 0) {
            scores[kk] = dot * scale;
        }
        __syncthreads();
    }

    // ------- Step 2: Softmax -------
    if (tid == 0) {
        float max_s = -FLT_MAX;
        for (int kk = 0; kk < num_k; kk++) {
            max_s = fmaxf(max_s, scores[kk]);
        }
        float sum_exp = 0.0f;
        for (int kk = 0; kk < num_k; kk++) {
            attn[kk] = expf(scores[kk] - max_s);
            sum_exp += attn[kk];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int kk = 0; kk < num_k; kk++) {
            attn[kk] *= inv_sum;
            attn_weights[n * num_k + kk] = attn[kk];
        }
    }
    __syncthreads();

    // ------- Step 3: Weighted sum of V neighbors -------
    float* out_n = out + (long long)n * C;
    for (int c = tid; c < C; c += block_size) {
        float val = 0.0f;
        for (int kk = 0; kk < num_k; kk++) {
            int neighbor = idx_n[kk];
            val += attn[kk] * v[(long long)neighbor * C + c];
        }
        out_n[c] = val;
    }
}


// ============================================================================
// BACKWARD KERNEL
// ============================================================================
// One block per query point. Computes grad_Q and scatters grad_K, grad_V
// using atomicAdd.
//
// Equations:
//   grad_attn[k] = sum_c grad_out[n,c] * V[idx[n,k], c]
//   ds = sum_k attn[k] * grad_attn[k]
//   grad_scores[k] = attn[k] * (grad_attn[k] - ds)
//   grad_Q[n, c] = sum_k grad_scores[k] * K[idx[n,k], c] * scale
//   grad_K[idx[n,k], c] += grad_scores[k] * Q[n, c] * scale   (atomicAdd)
//   grad_V[idx[n,k], c] += attn[k] * grad_out[n, c]           (atomicAdd)

__global__ void fused_knn_attn_backward_kernel(
    const float* __restrict__ grad_out,     // [N, C]
    const float* __restrict__ q,            // [N, C]
    const float* __restrict__ k,            // [N, C]
    const float* __restrict__ v,            // [N, C]
    const int*   __restrict__ idx,          // [N, num_k]
    const float* __restrict__ attn_weights, // [N, num_k]
    float*       __restrict__ grad_q,       // [N, C]
    float*       __restrict__ grad_k,       // [N, C]
    float*       __restrict__ grad_v,       // [N, C]
    const int N,
    const int C,
    const int num_k,
    const float scale
) {
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Shared memory: grad_attn[num_k] + attn[num_k] + grad_scores[num_k] + reduce_buf[ceil(block_size/32)]
    extern __shared__ float smem[];
    float* s_grad_attn   = smem;                         // [num_k]
    float* s_attn        = s_grad_attn + num_k;          // [num_k]
    float* s_grad_scores = s_attn + num_k;               // [num_k]
    float* reduce_buf    = s_grad_scores + num_k;        // [ceil(block_size/32)]

    const float* grad_out_n = grad_out + (long long)n * C;
    const float* q_n        = q + (long long)n * C;
    const int*   idx_n      = idx + (long long)n * num_k;
    float*       grad_q_n   = grad_q + (long long)n * C;

    // Load attn weights into shared memory
    if (tid < num_k) {
        s_attn[tid] = attn_weights[(long long)n * num_k + tid];
    }
    __syncthreads();

    // ------- Step 1: Compute grad_attn[k] = dot(grad_out[n], V[idx[n,k]]) -------
    for (int kk = 0; kk < num_k; kk++) {
        int neighbor = idx_n[kk];
        const float* v_neighbor = v + (long long)neighbor * C;

        float partial = 0.0f;
        for (int c = tid; c < C; c += block_size) {
            partial += grad_out_n[c] * v_neighbor[c];
        }
        float dot = block_reduce_sum(partial, reduce_buf, tid, block_size);
        if (tid == 0) {
            s_grad_attn[kk] = dot;
        }
        __syncthreads();
    }

    // ------- Step 2: Softmax backward -------
    // ds = sum_k attn[k] * grad_attn[k]
    // grad_scores[k] = attn[k] * (grad_attn[k] - ds)
    if (tid == 0) {
        float ds = 0.0f;
        for (int kk = 0; kk < num_k; kk++) {
            ds += s_attn[kk] * s_grad_attn[kk];
        }
        for (int kk = 0; kk < num_k; kk++) {
            s_grad_scores[kk] = s_attn[kk] * (s_grad_attn[kk] - ds);
        }
    }
    __syncthreads();

    // ------- Step 3: grad_Q[n, c] = sum_k grad_scores[k] * K[idx[n,k], c] * scale -------
    for (int c = tid; c < C; c += block_size) {
        float g = 0.0f;
        for (int kk = 0; kk < num_k; kk++) {
            int neighbor = idx_n[kk];
            g += s_grad_scores[kk] * k[(long long)neighbor * C + c];
        }
        grad_q_n[c] = g * scale;
    }

    // ------- Step 4: Scatter grad_K and grad_V using atomicAdd -------
    for (int kk = 0; kk < num_k; kk++) {
        int neighbor = idx_n[kk];
        float gs = s_grad_scores[kk] * scale;
        float aw = s_attn[kk];

        for (int c = tid; c < C; c += block_size) {
            // grad_K[neighbor, c] += grad_scores[k] * Q[n, c] * scale
            atomicAdd(grad_k + (long long)neighbor * C + c, gs * q_n[c]);
            // grad_V[neighbor, c] += attn[k] * grad_out[n, c]
            atomicAdd(grad_v + (long long)neighbor * C + c, aw * grad_out_n[c]);
        }
    }
}


// ============================================================================
// C++ Launcher Functions
// ============================================================================

void fused_knn_attn_forward_cuda_launcher(
    const float* q, const float* k, const float* v, const int* idx,
    float* out, float* attn_weights,
    int N, int C, int num_k, float scale
) {
    int block_size = THREADS_PER_BLOCK;
    if (C < block_size) {
        // Round up to next power of 2 for efficient reductions
        block_size = 1;
        while (block_size < C) block_size <<= 1;
        if (block_size < 32) block_size = 32;  // min warp size
    }

    int num_warps = (block_size + 31) / 32;
    // smem: scores[num_k] + attn[num_k] + reduce_buf[num_warps]
    int smem_size = (2 * num_k + num_warps) * sizeof(float);

    fused_knn_attn_forward_kernel<<<N, block_size, smem_size>>>(
        q, k, v, idx, out, attn_weights, N, C, num_k, scale
    );
}

void fused_knn_attn_backward_cuda_launcher(
    const float* grad_out, const float* q, const float* k, const float* v,
    const int* idx, const float* attn_weights,
    float* grad_q, float* grad_k, float* grad_v,
    int N, int C, int num_k, float scale
) {
    int block_size = THREADS_PER_BLOCK;
    if (C < block_size) {
        block_size = 1;
        while (block_size < C) block_size <<= 1;
        if (block_size < 32) block_size = 32;
    }

    int num_warps = (block_size + 31) / 32;
    // smem: grad_attn[num_k] + attn[num_k] + grad_scores[num_k] + reduce_buf[num_warps]
    int smem_size = (3 * num_k + num_warps) * sizeof(float);

    fused_knn_attn_backward_kernel<<<N, block_size, smem_size>>>(
        grad_out, q, k, v, idx, attn_weights,
        grad_q, grad_k, grad_v, N, C, num_k, scale
    );
}
