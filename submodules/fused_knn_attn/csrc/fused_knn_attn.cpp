#include <torch/extension.h>
#include "fused_knn_attn_kernel.h"

void fused_knn_attn_forward_cuda(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor idx,
    at::Tensor out, at::Tensor attn_weights,
    int N, int C, int num_k, float scale
) {
    fused_knn_attn_forward_cuda_launcher(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        idx.data_ptr<int>(),
        out.data_ptr<float>(), attn_weights.data_ptr<float>(),
        N, C, num_k, scale
    );
}

void fused_knn_attn_backward_cuda(
    at::Tensor grad_out, at::Tensor q, at::Tensor k, at::Tensor v,
    at::Tensor idx, at::Tensor attn_weights,
    at::Tensor grad_q, at::Tensor grad_k, at::Tensor grad_v,
    int N, int C, int num_k, float scale
) {
    fused_knn_attn_backward_cuda_launcher(
        grad_out.data_ptr<float>(), q.data_ptr<float>(),
        k.data_ptr<float>(), v.data_ptr<float>(),
        idx.data_ptr<int>(), attn_weights.data_ptr<float>(),
        grad_q.data_ptr<float>(), grad_k.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        N, C, num_k, scale
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_knn_attn_forward_cuda", &fused_knn_attn_forward_cuda);
    m.def("fused_knn_attn_backward_cuda", &fused_knn_attn_backward_cuda);
}
