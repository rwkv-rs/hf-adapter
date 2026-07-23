#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <limits>
#include <vector>

namespace {

__device__ inline __nv_bfloat162 load_bf16x2(const at::BFloat16* pointer) {
    return *reinterpret_cast<const __nv_bfloat162*>(pointer);
}

__device__ inline void store_bf16x2(
    at::BFloat16* pointer, __nv_bfloat162 value) {
    *reinterpret_cast<__nv_bfloat162*>(pointer) = value;
}

inline int64_t ceil_div(int64_t value, int64_t divisor) {
    // All callers pass positive values.  Spell the division this way so a
    // valid, very large tensor extent cannot overflow while adding
    // ``divisor - 1`` before the division.
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

__global__ void mix6_shifted_forward_kernel(
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ shifted,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ out_r,
    at::BFloat16* __restrict__ out_w,
    at::BFloat16* __restrict__ out_k,
    at::BFloat16* __restrict__ out_v,
    at::BFloat16* __restrict__ out_a,
    at::BFloat16* __restrict__ out_g,
    int64_t rows,
    int64_t channels) {
    const int64_t pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_pairs = rows * (channels / 2);
    if (pair >= total_pairs) {
        return;
    }
    const int64_t channel = (pair % (channels / 2)) * 2;
    const int64_t index = (pair / (channels / 2)) * channels + channel;
    const __nv_bfloat162 current = load_bf16x2(x + index);
    const __nv_bfloat162 delta =
        __hsub2(load_bf16x2(shifted + index), current);
    store_bf16x2(out_r + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_r + channel))));
    store_bf16x2(out_w + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_w + channel))));
    store_bf16x2(out_k + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_k + channel))));
    store_bf16x2(out_v + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_v + channel))));
    store_bf16x2(out_a + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_a + channel))));
    store_bf16x2(out_g + index, __hadd2(current, __hmul2(delta, load_bf16x2(x_g + channel))));
}

// Keep adjacent channel pairs together so every half warp issues coalesced
// loads.  Eight independent row workers share one 128-thread block.  Each
// worker visits a disjoint, monotonically increasing subsequence of a fixed
// 64-row tile; the first worker then merges those eight partials in a fixed
// order.  The result is deterministic without serializing all B*T rows onto
// one thread per channel pair.
constexpr int MIX6_REDUCTION_THREADS = 128;
constexpr int MIX6_CHANNEL_PAIRS_PER_TILE = 16;
constexpr int MIX6_ROW_WORKERS_PER_BLOCK =
    MIX6_REDUCTION_THREADS / MIX6_CHANNEL_PAIRS_PER_TILE;
constexpr int MIX6_ROWS_PER_PARTIAL = 64;
constexpr int MIX6_PARAMETER_GRAD_COMPONENTS = 12;

static_assert(
    MIX6_REDUCTION_THREADS % MIX6_CHANNEL_PAIRS_PER_TILE == 0,
    "Mix6 reduction block must contain complete channel-pair tiles");
static_assert(
    MIX6_ROWS_PER_PARTIAL % MIX6_ROW_WORKERS_PER_BLOCK == 0,
    "Mix6 row tiles must divide evenly across row workers");

__global__ void mix6_shifted_backward_partials_kernel(
    const at::BFloat16* __restrict__ grad_r,
    const at::BFloat16* __restrict__ grad_w,
    const at::BFloat16* __restrict__ grad_k,
    const at::BFloat16* __restrict__ grad_v,
    const at::BFloat16* __restrict__ grad_a,
    const at::BFloat16* __restrict__ grad_g,
    const at::BFloat16* __restrict__ x,
    const at::BFloat16* __restrict__ shifted,
    const at::BFloat16* __restrict__ x_r,
    const at::BFloat16* __restrict__ x_w,
    const at::BFloat16* __restrict__ x_k,
    const at::BFloat16* __restrict__ x_v,
    const at::BFloat16* __restrict__ x_a,
    const at::BFloat16* __restrict__ x_g,
    at::BFloat16* __restrict__ grad_x,
    at::BFloat16* __restrict__ grad_shifted,
    float* __restrict__ parameter_partials,
    int64_t rows,
    int64_t channels,
    int64_t channel_pair_tiles) {
    const int row_worker =
        threadIdx.x / MIX6_CHANNEL_PAIRS_PER_TILE;
    const int channel_pair_in_tile =
        threadIdx.x % MIX6_CHANNEL_PAIRS_PER_TILE;
    const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
    const int64_t partial_index = flat_block / channel_pair_tiles;
    const int64_t channel_pair_tile =
        flat_block - partial_index * channel_pair_tiles;
    const int64_t channel_pair =
        channel_pair_tile * MIX6_CHANNEL_PAIRS_PER_TILE
        + channel_pair_in_tile;
    const bool valid_channel = channel_pair < channels / 2;
    const int64_t channel = channel_pair * 2;
    float ar0 = 0.0f, ar1 = 0.0f, aw0 = 0.0f, aw1 = 0.0f;
    float ak0 = 0.0f, ak1 = 0.0f, av0 = 0.0f, av1 = 0.0f;
    float aa0 = 0.0f, aa1 = 0.0f, ag0 = 0.0f, ag1 = 0.0f;

    if (valid_channel) {
        const __nv_bfloat162 pr = load_bf16x2(x_r + channel);
        const __nv_bfloat162 pw = load_bf16x2(x_w + channel);
        const __nv_bfloat162 pk = load_bf16x2(x_k + channel);
        const __nv_bfloat162 pv = load_bf16x2(x_v + channel);
        const __nv_bfloat162 pa = load_bf16x2(x_a + channel);
        const __nv_bfloat162 pg = load_bf16x2(x_g + channel);
        const int64_t row_begin =
            partial_index * MIX6_ROWS_PER_PARTIAL + row_worker;
        const int64_t partial_end =
            (partial_index + 1) *
            static_cast<int64_t>(MIX6_ROWS_PER_PARTIAL);
        const int64_t row_end = partial_end < rows ? partial_end : rows;
        for (int64_t row = row_begin;
             row < row_end;
             row += MIX6_ROW_WORKERS_PER_BLOCK) {
            const int64_t index = row * channels + channel;
            const __nv_bfloat162 gr = load_bf16x2(grad_r + index);
            const __nv_bfloat162 gw = load_bf16x2(grad_w + index);
            const __nv_bfloat162 gk = load_bf16x2(grad_k + index);
            const __nv_bfloat162 gv = load_bf16x2(grad_v + index);
            const __nv_bfloat162 ga = load_bf16x2(grad_a + index);
            const __nv_bfloat162 gg = load_bf16x2(grad_g + index);

            // Follow the readable autograd graph rather than distributing
            // ``g * (1 - mix)``.  The element gradients therefore preserve the
            // same BF16 edge accumulation order as the canonical expression.
            __nv_bfloat162 direct = __hadd2(gr, gw);
            direct = __hadd2(direct, gk);
            direct = __hadd2(direct, gv);
            direct = __hadd2(direct, ga);
            direct = __hadd2(direct, gg);
            __nv_bfloat162 grad_delta = __hmul2(gr, pr);
            grad_delta = __hadd2(grad_delta, __hmul2(gw, pw));
            grad_delta = __hadd2(grad_delta, __hmul2(gk, pk));
            grad_delta = __hadd2(grad_delta, __hmul2(gv, pv));
            grad_delta = __hadd2(grad_delta, __hmul2(ga, pa));
            grad_delta = __hadd2(grad_delta, __hmul2(gg, pg));
            store_bf16x2(grad_x + index, __hsub2(direct, grad_delta));
            store_bf16x2(grad_shifted + index, grad_delta);

            const float2 difference = __bfloat1622float2(
                __hsub2(
                    load_bf16x2(shifted + index),
                    load_bf16x2(x + index)));
            const float2 fr = __bfloat1622float2(gr);
            const float2 fw = __bfloat1622float2(gw);
            const float2 fk = __bfloat1622float2(gk);
            const float2 fv = __bfloat1622float2(gv);
            const float2 fa = __bfloat1622float2(ga);
            const float2 fg = __bfloat1622float2(gg);
            ar0 += fr.x * difference.x;
            ar1 += fr.y * difference.y;
            aw0 += fw.x * difference.x;
            aw1 += fw.y * difference.y;
            ak0 += fk.x * difference.x;
            ak1 += fk.y * difference.y;
            av0 += fv.x * difference.x;
            av1 += fv.y * difference.y;
            aa0 += fa.x * difference.x;
            aa1 += fa.y * difference.y;
            ag0 += fg.x * difference.x;
            ag1 += fg.y * difference.y;
        }
    }

    __shared__ float reductions
        [MIX6_PARAMETER_GRAD_COMPONENTS]
        [MIX6_ROW_WORKERS_PER_BLOCK]
        [MIX6_CHANNEL_PAIRS_PER_TILE];
    reductions[0][row_worker][channel_pair_in_tile] = ar0;
    reductions[1][row_worker][channel_pair_in_tile] = ar1;
    reductions[2][row_worker][channel_pair_in_tile] = aw0;
    reductions[3][row_worker][channel_pair_in_tile] = aw1;
    reductions[4][row_worker][channel_pair_in_tile] = ak0;
    reductions[5][row_worker][channel_pair_in_tile] = ak1;
    reductions[6][row_worker][channel_pair_in_tile] = av0;
    reductions[7][row_worker][channel_pair_in_tile] = av1;
    reductions[8][row_worker][channel_pair_in_tile] = aa0;
    reductions[9][row_worker][channel_pair_in_tile] = aa1;
    reductions[10][row_worker][channel_pair_in_tile] = ag0;
    reductions[11][row_worker][channel_pair_in_tile] = ag1;
    __syncthreads();

    if (row_worker == 0 && valid_channel) {
        float totals[MIX6_PARAMETER_GRAD_COMPONENTS] = {0.0f};
#pragma unroll
        for (int worker = 0; worker < MIX6_ROW_WORKERS_PER_BLOCK; ++worker) {
#pragma unroll
            for (int component = 0;
                 component < MIX6_PARAMETER_GRAD_COMPONENTS;
                 ++component) {
                totals[component] +=
                    reductions[component][worker][channel_pair_in_tile];
            }
        }
        const int64_t base =
            (partial_index * 6 * channels) + channel;
        parameter_partials[base] = totals[0];
        parameter_partials[base + 1] = totals[1];
        parameter_partials[base + channels] = totals[2];
        parameter_partials[base + channels + 1] = totals[3];
        parameter_partials[base + 2 * channels] = totals[4];
        parameter_partials[base + 2 * channels + 1] = totals[5];
        parameter_partials[base + 3 * channels] = totals[6];
        parameter_partials[base + 3 * channels + 1] = totals[7];
        parameter_partials[base + 4 * channels] = totals[8];
        parameter_partials[base + 4 * channels + 1] = totals[9];
        parameter_partials[base + 5 * channels] = totals[10];
        parameter_partials[base + 5 * channels + 1] = totals[11];
    }
}

__global__ void mix6_shifted_backward_finalize_kernel(
    const float* __restrict__ parameter_partials,
    at::BFloat16* __restrict__ grad_x_r,
    at::BFloat16* __restrict__ grad_x_w,
    at::BFloat16* __restrict__ grad_x_k,
    at::BFloat16* __restrict__ grad_x_v,
    at::BFloat16* __restrict__ grad_x_a,
    at::BFloat16* __restrict__ grad_x_g,
    int64_t partial_count,
    int64_t channels) {
    const int64_t channel_pair =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel_pair >= channels / 2) {
        return;
    }
    const int64_t channel = channel_pair * 2;
    float totals[MIX6_PARAMETER_GRAD_COMPONENTS] = {0.0f};
    for (int64_t partial = 0; partial < partial_count; ++partial) {
        const int64_t base = (partial * 6 * channels) + channel;
        totals[0] += parameter_partials[base];
        totals[1] += parameter_partials[base + 1];
        totals[2] += parameter_partials[base + channels];
        totals[3] += parameter_partials[base + channels + 1];
        totals[4] += parameter_partials[base + 2 * channels];
        totals[5] += parameter_partials[base + 2 * channels + 1];
        totals[6] += parameter_partials[base + 3 * channels];
        totals[7] += parameter_partials[base + 3 * channels + 1];
        totals[8] += parameter_partials[base + 4 * channels];
        totals[9] += parameter_partials[base + 4 * channels + 1];
        totals[10] += parameter_partials[base + 5 * channels];
        totals[11] += parameter_partials[base + 5 * channels + 1];
    }
    store_bf16x2(
        grad_x_r + channel,
        __floats2bfloat162_rn(totals[0], totals[1]));
    store_bf16x2(
        grad_x_w + channel,
        __floats2bfloat162_rn(totals[2], totals[3]));
    store_bf16x2(
        grad_x_k + channel,
        __floats2bfloat162_rn(totals[4], totals[5]));
    store_bf16x2(
        grad_x_v + channel,
        __floats2bfloat162_rn(totals[6], totals[7]));
    store_bf16x2(
        grad_x_a + channel,
        __floats2bfloat162_rn(totals[8], totals[9]));
    store_bf16x2(
        grad_x_g + channel,
        __floats2bfloat162_rn(totals[10], totals[11]));
}

}  // namespace

std::vector<torch::Tensor> mix6_shifted_forward_cuda(
    torch::Tensor x,
    torch::Tensor shifted,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g) {
    std::vector<torch::Tensor> outputs;
    outputs.reserve(6);
    for (int index = 0; index < 6; ++index) {
        outputs.push_back(torch::empty_like(x));
    }
    const int threads = 256;
    const int64_t rows = x.size(0) * x.size(1);
    const int64_t channels = x.size(2);
    TORCH_CHECK(
        rows <= std::numeric_limits<int64_t>::max() / (channels / 2),
        "Mix6 forward tensor is too large for its flattened CUDA index");
    const int64_t block_count = ceil_div(
        rows * (channels / 2), static_cast<int64_t>(threads));
    TORCH_CHECK(
        block_count <= std::numeric_limits<int>::max(),
        "Mix6 forward exceeds the CUDA one-dimensional grid limit");
    const int blocks = static_cast<int>(block_count);
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    mix6_shifted_forward_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<at::BFloat16>(), shifted.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(), x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(), x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(), x_g.data_ptr<at::BFloat16>(),
        outputs[0].data_ptr<at::BFloat16>(), outputs[1].data_ptr<at::BFloat16>(),
        outputs[2].data_ptr<at::BFloat16>(), outputs[3].data_ptr<at::BFloat16>(),
        outputs[4].data_ptr<at::BFloat16>(), outputs[5].data_ptr<at::BFloat16>(),
        rows, channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return outputs;
}

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
    torch::Tensor x_g) {
    auto grad_x = torch::empty_like(x);
    auto grad_shifted = torch::empty_like(shifted);
    std::vector<torch::Tensor> parameter_grads;
    parameter_grads.reserve(6);
    for (int index = 0; index < 6; ++index) {
        parameter_grads.push_back(torch::empty({x.size(2)}, x.options()));
    }
    const int64_t rows = x.size(0) * x.size(1);
    const int64_t channels = x.size(2);
    const int64_t partial_count = ceil_div(
        rows, static_cast<int64_t>(MIX6_ROWS_PER_PARTIAL));
    const int64_t channel_pair_tiles = ceil_div(
        channels / 2,
        static_cast<int64_t>(MIX6_CHANNEL_PAIRS_PER_TILE));
    TORCH_CHECK(
        partial_count <=
            std::numeric_limits<int64_t>::max() / channel_pair_tiles,
        "Mix6 backward partial grid size overflow");
    const int64_t partial_blocks = partial_count * channel_pair_tiles;
    TORCH_CHECK(
        partial_blocks <= std::numeric_limits<int>::max(),
        "Mix6 backward exceeds the CUDA one-dimensional grid limit");
    TORCH_CHECK(
        channels <= std::numeric_limits<int64_t>::max() / 6,
        "Mix6 backward parameter-partial stride overflow");
    const int64_t parameter_partial_stride = 6 * channels;
    TORCH_CHECK(
        partial_count <=
            std::numeric_limits<int64_t>::max() / parameter_partial_stride,
        "Mix6 backward parameter-partial workspace size overflow");
    auto parameter_partials = torch::empty(
        {partial_count, 6, channels},
        x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    mix6_shifted_backward_partials_kernel<<<
        static_cast<int>(partial_blocks),
        MIX6_REDUCTION_THREADS,
        0,
        stream>>>(
        grad_r.data_ptr<at::BFloat16>(), grad_w.data_ptr<at::BFloat16>(),
        grad_k.data_ptr<at::BFloat16>(), grad_v.data_ptr<at::BFloat16>(),
        grad_a.data_ptr<at::BFloat16>(), grad_g.data_ptr<at::BFloat16>(),
        x.data_ptr<at::BFloat16>(), shifted.data_ptr<at::BFloat16>(),
        x_r.data_ptr<at::BFloat16>(), x_w.data_ptr<at::BFloat16>(),
        x_k.data_ptr<at::BFloat16>(), x_v.data_ptr<at::BFloat16>(),
        x_a.data_ptr<at::BFloat16>(), x_g.data_ptr<at::BFloat16>(),
        grad_x.data_ptr<at::BFloat16>(), grad_shifted.data_ptr<at::BFloat16>(),
        parameter_partials.data_ptr<float>(),
        rows,
        channels,
        channel_pair_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int finalize_threads = 256;
    const int64_t finalize_block_count = ceil_div(
        channels / 2, static_cast<int64_t>(finalize_threads));
    TORCH_CHECK(
        finalize_block_count <= std::numeric_limits<int>::max(),
        "Mix6 backward finalize exceeds the CUDA one-dimensional grid limit");
    const int finalize_blocks = static_cast<int>(finalize_block_count);
    mix6_shifted_backward_finalize_kernel<<<
        finalize_blocks,
        finalize_threads,
        0,
        stream>>>(
        parameter_partials.data_ptr<float>(),
        parameter_grads[0].data_ptr<at::BFloat16>(),
        parameter_grads[1].data_ptr<at::BFloat16>(),
        parameter_grads[2].data_ptr<at::BFloat16>(),
        parameter_grads[3].data_ptr<at::BFloat16>(),
        parameter_grads[4].data_ptr<at::BFloat16>(),
        parameter_grads[5].data_ptr<at::BFloat16>(),
        partial_count,
        channels);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {
        grad_x, grad_shifted,
        parameter_grads[0], parameter_grads[1], parameter_grads[2],
        parameter_grads[3], parameter_grads[4], parameter_grads[5]};
}
