#ifndef _FUSED_KNN_ATTN_KERNEL_H
#define _FUSED_KNN_ATTN_KERNEL_H

#include <torch/extension.h>

// Forward: q [N,C], k [N,C], v [N,C], idx [N,K] -> out [N,C], attn [N,K]
void fused_knn_attn_forward_cuda(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor idx,
    at::Tensor out, at::Tensor attn_weights,
    int N, int C, int num_k, float scale);

// Backward: grad_out [N,C] -> grad_q [N,C], grad_k [N,C], grad_v [N,C]
void fused_knn_attn_backward_cuda(
    at::Tensor grad_out, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor idx, at::Tensor attn_weights,
    at::Tensor grad_q, at::Tensor grad_k, at::Tensor grad_v,
    int N, int C, int num_k, float scale);

// CUDA launcher functions (C++ linkage, called from .cpp, defined in .cu)
void fused_knn_attn_forward_cuda_launcher(
    const float* q, const float* k, const float* v, const int* idx,
    float* out, float* attn_weights,
    int N, int C, int num_k, float scale);

void fused_knn_attn_backward_cuda_launcher(
    const float* grad_out, const float* q, const float* k, const float* v,
    const int* idx, const float* attn_weights,
    float* grad_q, float* grad_k, float* grad_v,
    int N, int C, int num_k, float scale);

#endif
