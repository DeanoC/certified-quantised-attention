#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/version.h>

namespace {

constexpr int kBlockSize = 16;
constexpr int kDequantTileD = 32;
constexpr int kScoreThreads = 128;
constexpr int kCertThreads = 256;

__global__ void probe_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

__global__ void dequant_keys_to_fp16_t_kernel(
    const int8_t* __restrict__ keys_int8,
    const float* __restrict__ keys_scale,
    const float* __restrict__ keys_zp,
    at::Half* __restrict__ output,
    int tokens,
    int head_dim,
    int num_blocks) {
    __shared__ __half tile[kBlockSize][kDequantTileD];

    const int bid = blockIdx.x;
    const int d_base = blockIdx.y * kDequantTileD;
    const int kvh = blockIdx.z;
    const int tid = threadIdx.x;
    const int tile_elems = kBlockSize * kDequantTileD;

    // Load/dequantize in source-contiguous order: token row, then channel.
    for (int e = tid; e < tile_elems; e += blockDim.x) {
        const int local_t = e / kDequantTileD;
        const int local_d = e - local_t * kDequantTileD;
        const int d = d_base + local_d;
        const int tok = bid * kBlockSize + local_t;
        __half v = __float2half(0.0f);
        if (d < head_dim && tok < tokens) {
            const int key_idx = (kvh * tokens + tok) * head_dim + d;
            const int scale_idx = (kvh * num_blocks + bid) * head_dim + d;
            const float f = static_cast<float>(keys_int8[key_idx]) * keys_scale[scale_idx] + keys_zp[scale_idx];
            v = __float2half(f);
        }
        tile[local_t][local_d] = v;
    }
    __syncthreads();

    // Store in destination-contiguous order: channel row, then token.
    for (int e = tid; e < tile_elems; e += blockDim.x) {
        const int local_d = e / kBlockSize;
        const int local_t = e - local_d * kBlockSize;
        const int d = d_base + local_d;
        const int tok = bid * kBlockSize + local_t;
        if (d < head_dim && tok < tokens) {
            const int out_idx = (kvh * head_dim + d) * tokens + tok;
            reinterpret_cast<__half*>(output)[out_idx] = tile[local_t][local_d];
        }
    }
}

__global__ void score_blocks_kernel(
    const int8_t* __restrict__ keys_int8,
    const float* __restrict__ keys_scale,
    const float* __restrict__ keys_zp,
    const float* __restrict__ q_all,
    float* __restrict__ m_b,
    float* __restrict__ s_b,
    int kv_heads,
    int tokens,
    int head_dim,
    int num_blocks,
    int num_q_heads,
    int gqa_group,
    float q_scale) {
    __shared__ float scores[kBlockSize];
    __shared__ float reduce[kScoreThreads];

    const int bid = blockIdx.x;
    const int qh = blockIdx.y;
    const int tid = threadIdx.x;
    if (bid >= num_blocks || qh >= num_q_heads) {
        return;
    }
    const int kvh = min(qh / gqa_group, kv_heads - 1);

    float partial = 0.0f;
    const int token_lane = tid & (kBlockSize - 1);
    const int dim_start = tid >> 4;
    if (tid < kScoreThreads) {
        const int tok = bid * kBlockSize + token_lane;
        const int key_base = (kvh * tokens + tok) * head_dim;
        const int scale_base = (kvh * num_blocks + bid) * head_dim;
        const int q_base = qh * head_dim;
        for (int d = dim_start; d < head_dim; d += (kScoreThreads >> 4)) {
            const float q = q_all[q_base + d];
            const float k = static_cast<float>(keys_int8[key_base + d]) * keys_scale[scale_base + d]
                + keys_zp[scale_base + d];
            partial += q * k;
        }
    }
    reduce[tid] = partial;
    __syncthreads();

    // Reduce the 8 dimension-lanes belonging to each token.
    if (tid < kBlockSize) {
        float sum = 0.0f;
        for (int lane = 0; lane < (kScoreThreads >> 4); ++lane) {
            sum += reduce[(lane << 4) + tid];
        }
        scores[tid] = sum * q_scale;
    }
    __syncthreads();

    if (tid == 0) {
        float m = -INFINITY;
        #pragma unroll
        for (int i = 0; i < kBlockSize; ++i) {
            m = fmaxf(m, scores[i]);
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < kBlockSize; ++i) {
            s += expf(scores[i] - m);
        }
        const int out = qh * num_blocks + bid;
        m_b[out] = m;
        s_b[out] = s;
    }
}

__global__ void certify_kernel(
    const float* __restrict__ m_b,
    const float* __restrict__ s_b,
    const float* __restrict__ correction,
    int32_t* __restrict__ skip_i32,
    int num_blocks,
    int num_q_heads,
    int gqa_group,
    float block_epsilon) {
    __shared__ float scratch[kCertThreads];

    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    if (qh >= num_q_heads) {
        return;
    }
    const int kvh = qh / gqa_group;
    const int base = qh * num_blocks;
    const int corr_base = kvh * num_blocks;

    float local_max = -INFINITY;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        local_max = fmaxf(local_max, m_b[base + b]);
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float global_max = scratch[0];

    float local_total = 0.0f;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        local_total += s_b[base + b] * correction[corr_base + b] * expf(m_b[base + b] - global_max);
    }
    scratch[tid] = local_total;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float total = fmaxf(scratch[0], 1.0e-30f);

    for (int b = tid; b < num_blocks; b += blockDim.x) {
        const float mass = s_b[base + b] * correction[corr_base + b] * expf(m_b[base + b] - global_max);
        skip_i32[base + b] = (mass / total) < block_epsilon ? 1 : 0;
    }
}

}  // namespace

torch::Tensor cutlass_sm120_probe_launcher(torch::Tensor input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    auto output = torch::empty_like(input);
    const int64_t n = input.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    if (n > 0) {
        probe_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
    }
    return output;
}

torch::Tensor dequant_keys_to_fp16_t_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    int64_t block_size) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    const int kv_heads = keys_int8.size(0);
    const int tokens = keys_int8.size(1);
    const int head_dim = keys_int8.size(2);
    const int num_blocks = keys_scale.size(1);
    auto output = torch::empty(
        {kv_heads, head_dim, tokens},
        torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat16));
    const int threads = 256;
    const dim3 grid(num_blocks, (head_dim + kDequantTileD - 1) / kDequantTileD, kv_heads);
    if (kv_heads > 0 && tokens > 0 && head_dim > 0) {
        dequant_keys_to_fp16_t_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            keys_int8.data_ptr<int8_t>(),
            keys_scale.data_ptr<float>(),
            keys_zero_points.data_ptr<float>(),
            output.data_ptr<at::Half>(),
            tokens,
            head_dim,
            num_blocks);
    }
    return output;
}

std::vector<torch::Tensor> score_certify_sm120_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    torch::Tensor correction,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    double block_epsilon) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    const int kv_heads = keys_int8.size(0);
    const int tokens = keys_int8.size(1);
    const int head_dim = keys_int8.size(2);
    const int num_blocks = keys_scale.size(1);
    const int q_heads = q_all.size(0);

    auto f32_opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat32);
    auto i32_opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kInt32);
    auto m_b = torch::empty({q_heads, num_blocks}, f32_opts);
    auto s_b = torch::empty({q_heads, num_blocks}, f32_opts);
    auto skip = torch::empty({q_heads, num_blocks}, i32_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid_score(num_blocks, q_heads);
    score_blocks_kernel<<<grid_score, kScoreThreads, 0, stream>>>(
        keys_int8.data_ptr<int8_t>(),
        keys_scale.data_ptr<float>(),
        keys_zero_points.data_ptr<float>(),
        q_all.data_ptr<float>(),
        m_b.data_ptr<float>(),
        s_b.data_ptr<float>(),
        kv_heads,
        tokens,
        head_dim,
        num_blocks,
        q_heads,
        static_cast<int>(gqa_group),
        static_cast<float>(q_scale));

    certify_kernel<<<q_heads, kCertThreads, 0, stream>>>(
        m_b.data_ptr<float>(),
        s_b.data_ptr<float>(),
        correction.data_ptr<float>(),
        skip.data_ptr<int32_t>(),
        num_blocks,
        q_heads,
        static_cast<int>(gqa_group),
        static_cast<float>(block_epsilon));

    return {m_b, s_b, skip};
}
