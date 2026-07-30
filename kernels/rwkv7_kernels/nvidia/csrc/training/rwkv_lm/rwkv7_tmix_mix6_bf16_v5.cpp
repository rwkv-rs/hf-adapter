#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <utility>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_forward_v5_cuda(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> tmix_mix6_backward_v5_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

namespace {

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, name, " must be bf16");
}

void check_vec(const torch::Tensor& x, int64_t c, const char* name) {
    TORCH_CHECK(x.dim() == 1, name, " must have shape [C]");
    TORCH_CHECK(x.size(0) == c, name, " shape mismatch");
}

void check_same_device(
    const torch::Tensor& value,
    const torch::Tensor& reference,
    const char* name) {
    TORCH_CHECK(
        value.device() == reference.device(),
        name,
        " must share x's CUDA device");
}

} // namespace

std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(
        x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
        "x requires non-empty batch, time, and channel dimensions");
    int64_t c = x.size(2);
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>(&x_r, "x_r"),
             std::pair<const torch::Tensor*, const char*>(&x_w, "x_w"),
             std::pair<const torch::Tensor*, const char*>(&x_k, "x_k"),
             std::pair<const torch::Tensor*, const char*>(&x_v, "x_v"),
             std::pair<const torch::Tensor*, const char*>(&x_a, "x_a"),
             std::pair<const torch::Tensor*, const char*>(&x_g, "x_g")}) {
        check_vec(*item.first, c, item.second);
        check_same_device(*item.first, x, item.second);
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    return tmix_mix6_forward_v5_cuda(x, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_bf16_cuda(grad_r, "grad_r");
    check_bf16_cuda(grad_w, "grad_w");
    check_bf16_cuda(grad_k, "grad_k");
    check_bf16_cuda(grad_v, "grad_v");
    check_bf16_cuda(grad_a, "grad_a");
    check_bf16_cuda(grad_g, "grad_g");
    check_bf16_cuda(x, "x");
    check_bf16_cuda(x_r, "x_r");
    check_bf16_cuda(x_w, "x_w");
    check_bf16_cuda(x_k, "x_k");
    check_bf16_cuda(x_v, "x_v");
    check_bf16_cuda(x_a, "x_a");
    check_bf16_cuda(x_g, "x_g");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
    TORCH_CHECK(
        x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
        "x requires non-empty batch, time, and channel dimensions");
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>(&grad_r, "grad_r"),
             std::pair<const torch::Tensor*, const char*>(&grad_w, "grad_w"),
             std::pair<const torch::Tensor*, const char*>(&grad_k, "grad_k"),
             std::pair<const torch::Tensor*, const char*>(&grad_v, "grad_v"),
             std::pair<const torch::Tensor*, const char*>(&grad_a, "grad_a"),
             std::pair<const torch::Tensor*, const char*>(&grad_g, "grad_g")}) {
        TORCH_CHECK(item.first->sizes() == x.sizes(), item.second, " shape mismatch");
        check_same_device(*item.first, x, item.second);
    }
    int64_t c = x.size(2);
    TORCH_CHECK((c % 2) == 0, "tmix_mix6_v5 currently requires even C");
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>(&x_r, "x_r"),
             std::pair<const torch::Tensor*, const char*>(&x_w, "x_w"),
             std::pair<const torch::Tensor*, const char*>(&x_k, "x_k"),
             std::pair<const torch::Tensor*, const char*>(&x_v, "x_v"),
             std::pair<const torch::Tensor*, const char*>(&x_a, "x_a"),
             std::pair<const torch::Tensor*, const char*>(&x_g, "x_g")}) {
        check_vec(*item.first, c, item.second);
        check_same_device(*item.first, x, item.second);
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    return tmix_mix6_backward_v5_cuda(grad_r, grad_w, grad_k, grad_v, grad_a, grad_g, x, x_r, x_w, x_k, x_v, x_a, x_g);
}

TORCH_LIBRARY(rwkv7_tmix_mix6_bf16_v5, m) {
    m.def("forward(Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor x, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_mix6_bf16_v5, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
