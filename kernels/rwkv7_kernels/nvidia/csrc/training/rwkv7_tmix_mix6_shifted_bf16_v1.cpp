#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

std::vector<torch::Tensor> mix6_shifted_forward_cuda(
    torch::Tensor x,
    torch::Tensor shifted,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

std::vector<torch::Tensor> mix6_shifted_backward_cuda(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor shifted,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g);

namespace {

void check_bf16_cuda(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == torch::kBFloat16, name, " must be bf16");
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(value.data_ptr()) % alignof(std::uint32_t)
            == 0,
        name,
        " must be four-byte aligned for packed BF16x2 access");
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

void check_request(
    const torch::Tensor& x,
    const torch::Tensor& shifted,
    const std::vector<std::pair<const torch::Tensor*, const char*>>& mixes) {
    check_bf16_cuda(x, "x");
    check_bf16_cuda(shifted, "shifted");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B,T,C]");
    TORCH_CHECK(
        x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0,
        "x requires non-empty batch, time, and channel dimensions");
    TORCH_CHECK(shifted.sizes() == x.sizes(), "shifted shape mismatch");
    check_same_device(shifted, x, "shifted");
    const int64_t channels = x.size(2);
    TORCH_CHECK((channels % 2) == 0, "Mix6 requires even C");
    TORCH_CHECK(
        x.size(0) <= std::numeric_limits<int64_t>::max() / x.size(1),
        "Mix6 B*T size overflows its flattened CUDA index");
    const int64_t rows = x.size(0) * x.size(1);
    TORCH_CHECK(
        rows <= std::numeric_limits<int64_t>::max() / (channels / 2),
        "Mix6 flattened row/channel size overflows its CUDA index");
    for (const auto& item : mixes) {
        check_bf16_cuda(*item.first, item.second);
        TORCH_CHECK(item.first->dim() == 1, item.second, " must have shape [C]");
        TORCH_CHECK(item.first->size(0) == channels, item.second, " shape mismatch");
        check_same_device(*item.first, x, item.second);
    }
}

}  // namespace

std::vector<torch::Tensor> forward(
    torch::Tensor x,
    torch::Tensor shifted,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_request(
        x,
        shifted,
        {{&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
         {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}});
    const c10::cuda::CUDAGuard device_guard(x.device());
    return mix6_shifted_forward_cuda(
        x, shifted, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> backward(
    torch::Tensor grad_r,
    torch::Tensor grad_w,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_a,
    torch::Tensor grad_g,
    torch::Tensor x,
    torch::Tensor shifted,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    check_request(
        x,
        shifted,
        {{&x_r, "x_r"}, {&x_w, "x_w"}, {&x_k, "x_k"},
         {&x_v, "x_v"}, {&x_a, "x_a"}, {&x_g, "x_g"}});
    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>(&grad_r, "grad_r"),
             std::pair<const torch::Tensor*, const char*>(&grad_w, "grad_w"),
             std::pair<const torch::Tensor*, const char*>(&grad_k, "grad_k"),
             std::pair<const torch::Tensor*, const char*>(&grad_v, "grad_v"),
             std::pair<const torch::Tensor*, const char*>(&grad_a, "grad_a"),
             std::pair<const torch::Tensor*, const char*>(&grad_g, "grad_g")}) {
        check_bf16_cuda(*item.first, item.second);
        TORCH_CHECK(item.first->sizes() == x.sizes(), item.second, " shape mismatch");
        check_same_device(*item.first, x, item.second);
    }
    const c10::cuda::CUDAGuard device_guard(x.device());
    return mix6_shifted_backward_cuda(
        grad_r, grad_w, grad_k, grad_v, grad_a, grad_g,
        x, shifted, x_r, x_w, x_k, x_v, x_a, x_g);
}

TORCH_LIBRARY(rwkv7_tmix_mix6_shifted_bf16_v1, m) {
    m.def("forward(Tensor x, Tensor shifted, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
    m.def("backward(Tensor grad_r, Tensor grad_w, Tensor grad_k, Tensor grad_v, Tensor grad_a, Tensor grad_g, Tensor x, Tensor shifted, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(rwkv7_tmix_mix6_shifted_bf16_v1, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
