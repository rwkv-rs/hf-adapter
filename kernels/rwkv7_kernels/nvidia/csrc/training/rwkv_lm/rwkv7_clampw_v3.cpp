#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <tuple>
#include <utility>

#ifdef _FP32_
    using bf = float;
#else
    #include <cuda_bf16.h>
    using bf = __nv_bfloat16;
#endif

void cuda_forward_v3(int B, int T, int H, bf*r, float*decay, bf*k, bf*v, bf*a, bf*b, bf*y, float*s, float*sa, cudaStream_t stream);

namespace {

#ifdef _FP32_
constexpr at::ScalarType kVectorDtype = torch::kFloat32;
#else
constexpr at::ScalarType kVectorDtype = torch::kBFloat16;
#endif
constexpr int64_t kHeadSize = 64;
constexpr int64_t kChunkLength = 16;

void check_cuda_tensor(
        const torch::Tensor& value,
        const torch::Tensor& reference,
        at::ScalarType dtype,
        const char* name) {
    TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.scalar_type() == dtype, name, " has an invalid dtype");
    TORCH_CHECK(value.device() == reference.device(), name, " must share r's CUDA device");
}

void check_vector_shape(const torch::Tensor& value, const torch::Tensor& r, const char* name) {
    TORCH_CHECK(value.sizes() == r.sizes(), name, " must have the same [B,T,H,N] shape as r");
}

void check_common_inputs(
        const torch::Tensor& r,
        const torch::Tensor& decay,
        const torch::Tensor& k,
        const torch::Tensor& v,
        const torch::Tensor& a,
        const torch::Tensor& b,
        const torch::Tensor& s,
        const torch::Tensor& sa) {
    TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");
    TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
    TORCH_CHECK(r.scalar_type() == kVectorDtype, "r has an invalid dtype");
    TORCH_CHECK(r.dim() == 4, "r must have shape [B,T,H,N]");
    TORCH_CHECK(r.size(0) > 0 && r.size(1) > 0 && r.size(2) > 0,
                "r requires non-empty batch, time, and head dimensions");
    TORCH_CHECK(r.size(3) == kHeadSize, "r head width must equal 64");
    TORCH_CHECK(r.size(1) % kChunkLength == 0,
                "r time dimension must be divisible by the compiled chunk length");
    TORCH_CHECK(r.size(0) <= std::numeric_limits<int>::max()
                    && r.size(1) <= std::numeric_limits<int>::max()
                    && r.size(2) <= std::numeric_limits<int>::max(),
                "r dimensions exceed the CUDA launch ABI");

    for (const auto& item : {
             std::pair<const torch::Tensor*, const char*>(&k, "k"),
             std::pair<const torch::Tensor*, const char*>(&v, "v"),
             std::pair<const torch::Tensor*, const char*>(&a, "a"),
             std::pair<const torch::Tensor*, const char*>(&b, "b")}) {
        check_cuda_tensor(*item.first, r, kVectorDtype, item.second);
        check_vector_shape(*item.first, r, item.second);
    }
    check_cuda_tensor(decay, r, torch::kFloat32, "decay");
    check_vector_shape(decay, r, "decay");

    check_cuda_tensor(sa, r, torch::kFloat32, "sa");
    check_vector_shape(sa, r, "sa");
    check_cuda_tensor(s, r, torch::kFloat32, "s");
    TORCH_CHECK(s.dim() == 5, "s must have shape [B,H,T/chunk,N,N]");
    TORCH_CHECK(
        s.size(0) == r.size(0)
            && s.size(1) == r.size(2)
            && s.size(2) == r.size(1) / kChunkLength
            && s.size(3) == kHeadSize
            && s.size(4) == kHeadSize,
        "s must have shape [B,H,T/chunk,N,N]");
}

} // namespace

void forward(torch::Tensor &r, torch::Tensor &decay, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    check_common_inputs(r, decay, k, v, a, b, s, sa);
    check_cuda_tensor(y, r, kVectorDtype, "y");
    check_vector_shape(y, r, "y");
    const c10::cuda::CUDAGuard device_guard(r.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(r.get_device());
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    cuda_forward_v3(B, T, H, (bf*)r.data_ptr(), (float*)decay.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), stream);
}

void cuda_backward_v3(int B, int T, int H, bf*r, float*decay, bf*k, bf*v, bf*a, bf*b, bf*dy, float*s, float*sa, bf*dr, float*ddecay, bf*dk, bf*dv, bf*da, bf*db, cudaStream_t stream);

void backward(torch::Tensor &r, torch::Tensor &decay, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &b, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dr, torch::Tensor &ddecay, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &da, torch::Tensor &db) {
    check_common_inputs(r, decay, k, v, a, b, s, sa);
    for (const auto& item : {
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&dy, kVectorDtype, "dy"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&dr, kVectorDtype, "dr"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&ddecay, torch::kFloat32, "ddecay"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&dk, kVectorDtype, "dk"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&dv, kVectorDtype, "dv"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&da, kVectorDtype, "da"),
             std::tuple<const torch::Tensor*, at::ScalarType, const char*>(&db, kVectorDtype, "db")}) {
        check_cuda_tensor(*std::get<0>(item), r, std::get<1>(item), std::get<2>(item));
        check_vector_shape(*std::get<0>(item), r, std::get<2>(item));
    }
    const c10::cuda::CUDAGuard device_guard(r.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(r.get_device());
    int B = r.sizes()[0], T = r.sizes()[1], H = r.sizes()[2];
    cuda_backward_v3(B, T, H, (bf*)r.data_ptr(), (float*)decay.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)a.data_ptr(), (bf*)b.data_ptr(), (bf*)dy.data_ptr(),
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dr.data_ptr(), (float*)ddecay.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)da.data_ptr(), (bf*)db.data_ptr(), stream);
}

TORCH_LIBRARY(rwkv7_clampw_v3, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
