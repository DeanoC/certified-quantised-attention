#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int kNumGroups = 8;
constexpr int kGroupSize = 32;
constexpr int kBlockSize = 16;
constexpr int kWordsPerGroup = 8;
constexpr int kThreads = 256;
constexpr int kMaxTokens = 256;
constexpr int kMaxBlocks = 32;

__device__ __forceinline__ float block_reduce_max(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

template <typename IndexT, typename MaskT>
__global__ void fused_selected_blocks_context_kernel(
    const int32_t* __restrict__ payload,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const IndexT* __restrict__ selected_block_ids,
    const MaskT* __restrict__ valid_mask,
    const float* __restrict__ queries,
    const float* __restrict__ query_group_sums,
    const float* __restrict__ values,
    float* __restrict__ output,
    int selected_block_count,
    int block_count,
    int head_dim,
    float query_scale) {
    __shared__ float s_query[256];
    __shared__ float s_qsum[8];
    __shared__ float s_logits[kMaxTokens];
    __shared__ float s_weights[kMaxTokens];
    __shared__ int s_block_ids[16];
    __shared__ float s_reduce[kThreads];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < 256) {
        s_query[tid] = queries[q_idx * 256 + tid];
    }
    if (tid < 8) {
        s_qsum[tid] = query_group_sums[q_idx * 8 + tid];
    }
    if (tid < selected_block_count) {
        s_block_ids[tid] = static_cast<int>(selected_block_ids[tid]);
    }
    __syncthreads();

    const int token_slot = tid;
    float logit = -INFINITY;
    if (token_slot < selected_block_count * kBlockSize) {
        const int sel_block_idx = token_slot / kBlockSize;
        const int tok_idx = token_slot % kBlockSize;
        const int block_id = s_block_ids[sel_block_idx];
        if (block_id >= 0 && block_id < block_count && static_cast<int>(valid_mask[block_id * kBlockSize + tok_idx]) != 0) {
            float accum = 0.0f;
#pragma unroll
            for (int group_idx = 0; group_idx < kNumGroups; ++group_idx) {
                const float scale = scales[(group_idx * block_count + block_id) * kBlockSize + tok_idx];
                const float bias_term = bias[(group_idx * block_count + block_id) * kBlockSize + tok_idx];
                float group_dot = 0.0f;
#pragma unroll
                for (int word_idx = 0; word_idx < kWordsPerGroup; ++word_idx) {
                    const int32_t packed = payload[((group_idx * block_count + block_id) * kBlockSize + tok_idx) * kWordsPerGroup + word_idx];
#pragma unroll
                    for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
                        const int elem_idx = word_idx * 4 + byte_idx;
                        const int code = (packed >> (byte_idx * 8)) & 0xFF;
                        group_dot += static_cast<float>(code) * s_query[group_idx * kGroupSize + elem_idx];
                    }
                }
                accum += group_dot * scale + s_qsum[group_idx] * bias_term;
            }
            logit = accum * query_scale;
        }
    }
    s_logits[tid] = logit;
    __syncthreads();

    const float max_logit = block_reduce_max(logit, s_reduce);
    float weight = 0.0f;
    if (token_slot < selected_block_count * kBlockSize && isfinite(logit)) {
        weight = expf(logit - max_logit);
    }
    s_weights[tid] = weight;
    __syncthreads();

    const float denom = block_reduce_sum(weight, s_reduce);
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int flat_token_idx = 0; flat_token_idx < selected_block_count * kBlockSize; ++flat_token_idx) {
            if (s_weights[flat_token_idx] == 0.0f) {
                continue;
            }
            const int sel_block_idx = flat_token_idx / kBlockSize;
            const int tok_idx = flat_token_idx % kBlockSize;
            const int block_id = s_block_ids[sel_block_idx];
            acc += s_weights[flat_token_idx] * values[(block_id * kBlockSize + tok_idx) * head_dim + tid];
        }
        output[q_idx * head_dim + tid] = acc / fmaxf(denom, 1e-8f);
    }
}

template <typename IndexT, typename MaskT>
__global__ void fused_selected_blocks_stream_stats_kernel(
    const int32_t* __restrict__ payload,
    const float* __restrict__ scales,
    const float* __restrict__ bias,
    const IndexT* __restrict__ selected_block_ids,
    const MaskT* __restrict__ valid_mask,
    const float* __restrict__ queries,
    const float* __restrict__ query_group_sums,
    const float* __restrict__ values,
    float* __restrict__ h_out,
    float* __restrict__ m_out,
    float* __restrict__ l_out,
    float* __restrict__ block_max_out,
    float* __restrict__ block_mass_num_out,
    int selected_block_count,
    int block_count,
    int head_dim,
    float query_scale) {
    __shared__ float s_query[256];
    __shared__ float s_qsum[8];
    __shared__ float s_logits[kMaxTokens];
    __shared__ float s_weights[kMaxTokens];
    __shared__ int s_block_ids[16];
    __shared__ float s_reduce[kThreads];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (tid < 256) {
        s_query[tid] = queries[q_idx * 256 + tid];
    }
    if (tid < 8) {
        s_qsum[tid] = query_group_sums[q_idx * 8 + tid];
    }
    if (tid < selected_block_count) {
        s_block_ids[tid] = static_cast<int>(selected_block_ids[tid]);
    }
    __syncthreads();

    const int token_slot = tid;
    float logit = -INFINITY;
    if (token_slot < selected_block_count * kBlockSize) {
        const int sel_block_idx = token_slot / kBlockSize;
        const int tok_idx = token_slot % kBlockSize;
        const int block_id = s_block_ids[sel_block_idx];
        if (block_id >= 0 && block_id < block_count && static_cast<int>(valid_mask[block_id * kBlockSize + tok_idx]) != 0) {
            float accum = 0.0f;
#pragma unroll
            for (int group_idx = 0; group_idx < kNumGroups; ++group_idx) {
                const float scale = scales[(group_idx * block_count + block_id) * kBlockSize + tok_idx];
                const float bias_term = bias[(group_idx * block_count + block_id) * kBlockSize + tok_idx];
                float group_dot = 0.0f;
#pragma unroll
                for (int word_idx = 0; word_idx < kWordsPerGroup; ++word_idx) {
                    const int32_t packed = payload[((group_idx * block_count + block_id) * kBlockSize + tok_idx) * kWordsPerGroup + word_idx];
#pragma unroll
                    for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
                        const int elem_idx = word_idx * 4 + byte_idx;
                        const int code = (packed >> (byte_idx * 8)) & 0xFF;
                        group_dot += static_cast<float>(code) * s_query[group_idx * kGroupSize + elem_idx];
                    }
                }
                accum += group_dot * scale + s_qsum[group_idx] * bias_term;
            }
            logit = accum * query_scale;
        }
    }
    s_logits[tid] = logit;
    __syncthreads();

    const float max_logit = block_reduce_max(logit, s_reduce);
    float weight_num = 0.0f;
    if (token_slot < selected_block_count * kBlockSize && isfinite(logit)) {
        weight_num = expf(logit - max_logit);
    }
    s_weights[tid] = weight_num;
    __syncthreads();

    const float denom = block_reduce_sum(weight_num, s_reduce);
    if (tid == 0) {
        m_out[q_idx] = max_logit;
        l_out[q_idx] = denom;
    }
    if (tid < selected_block_count) {
        float block_max = -INFINITY;
        float block_mass = 0.0f;
#pragma unroll
        for (int tok_idx = 0; tok_idx < kBlockSize; ++tok_idx) {
            const int flat_token_idx = tid * kBlockSize + tok_idx;
            block_max = fmaxf(block_max, s_logits[flat_token_idx]);
            block_mass += s_weights[flat_token_idx];
        }
        block_max_out[q_idx * selected_block_count + tid] = block_max;
        block_mass_num_out[q_idx * selected_block_count + tid] = block_mass;
    }
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int flat_token_idx = 0; flat_token_idx < selected_block_count * kBlockSize; ++flat_token_idx) {
            const float weight = s_weights[flat_token_idx];
            if (weight == 0.0f) {
                continue;
            }
            const int sel_block_idx = flat_token_idx / kBlockSize;
            const int tok_idx = flat_token_idx % kBlockSize;
            const int block_id = s_block_ids[sel_block_idx];
            acc += weight * values[(block_id * kBlockSize + tok_idx) * head_dim + tid];
        }
        h_out[q_idx * head_dim + tid] = acc;
    }
}

template <typename ValueT>
__global__ void softmax_value_context_kernel(
    const float* __restrict__ logits,
    const ValueT* __restrict__ values,
    float* __restrict__ output,
    int token_count,
    int head_dim,
    float query_scale) {
    __shared__ float s_reduce[kThreads];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    for (int token_idx = tid; token_idx < token_count; token_idx += blockDim.x) {
        const float scaled = logits[q_idx * token_count + token_idx] * query_scale;
        local_max = fmaxf(local_max, scaled);
    }
    const float row_max = block_reduce_max(local_max, s_reduce);

    float local_denom = 0.0f;
    for (int token_idx = tid; token_idx < token_count; token_idx += blockDim.x) {
        const float scaled = logits[q_idx * token_count + token_idx] * query_scale;
        if (isfinite(scaled)) {
            local_denom += expf(scaled - row_max);
        }
    }
    const float denom = block_reduce_sum(local_denom, s_reduce);

    for (int dim_idx = tid; dim_idx < head_dim; dim_idx += blockDim.x) {
        float acc = 0.0f;
        for (int token_idx = 0; token_idx < token_count; ++token_idx) {
            const float scaled = logits[q_idx * token_count + token_idx] * query_scale;
            if (!isfinite(scaled)) {
                continue;
            }
            const float weight = expf(scaled - row_max);
            acc += weight * static_cast<float>(values[token_idx * head_dim + dim_idx]);
        }
        output[q_idx * head_dim + dim_idx] = acc / fmaxf(denom, 1e-8f);
    }
}

template <typename IndexT, typename ValueT>
__global__ void softmax_value_stream_stats_kernel(
    const float* __restrict__ logits,
    const IndexT* __restrict__ token_block_ids,
    const ValueT* __restrict__ values,
    float* __restrict__ h_out,
    float* __restrict__ m_out,
    float* __restrict__ l_out,
    float* __restrict__ block_max_out,
    float* __restrict__ block_mass_num_out,
    int token_count,
    int block_count,
    int head_dim,
    float query_scale) {
    __shared__ float s_reduce[kThreads];
    __shared__ float s_scaled_logits[kMaxTokens];
    __shared__ float s_weights[kMaxTokens];

    const int q_idx = blockIdx.x;
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    if (tid < token_count) {
        const float scaled = logits[q_idx * token_count + tid] * query_scale;
        s_scaled_logits[tid] = scaled;
        local_max = scaled;
    }
    const float row_max = block_reduce_max(local_max, s_reduce);

    float local_denom = 0.0f;
    if (tid < token_count) {
        const float scaled = s_scaled_logits[tid];
        const float weight = isfinite(scaled) ? expf(scaled - row_max) : 0.0f;
        s_weights[tid] = weight;
        local_denom = weight;
    }
    const float denom = block_reduce_sum(local_denom, s_reduce);

    if (tid == 0) {
        m_out[q_idx] = row_max;
        l_out[q_idx] = denom;
    }
    if (tid < block_count) {
        float block_max = -INFINITY;
        float block_mass = 0.0f;
        for (int token_idx = 0; token_idx < token_count; ++token_idx) {
            if (static_cast<int>(token_block_ids[token_idx]) != tid) {
                continue;
            }
            block_max = fmaxf(block_max, s_scaled_logits[token_idx]);
            block_mass += s_weights[token_idx];
        }
        block_max_out[q_idx * block_count + tid] = block_max;
        block_mass_num_out[q_idx * block_count + tid] = block_mass;
    }
    for (int dim_idx = tid; dim_idx < head_dim; dim_idx += blockDim.x) {
        float acc = 0.0f;
        for (int token_idx = 0; token_idx < token_count; ++token_idx) {
            const float weight = s_weights[token_idx];
            if (weight == 0.0f) {
                continue;
            }
            acc += weight * static_cast<float>(values[token_idx * head_dim + dim_idx]);
        }
        h_out[q_idx * head_dim + dim_idx] = acc;
    }
}

}  // namespace

torch::Tensor fused_selected_blocks_context_cuda_launcher(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale) {
    c10::cuda::CUDAGuard device_guard(payload_words.device());

    const int query_count = static_cast<int>(queries.size(0));
    const int block_count = static_cast<int>(payload_words.size(1));
    const int head_dim = static_cast<int>(values.size(2));
    const int selected_block_count = static_cast<int>(selected_block_ids.numel());

    auto output = torch::zeros({query_count, head_dim}, queries.options().dtype(torch::kFloat32));
    if (selected_block_count == 0) {
        return output;
    }

    const dim3 grid(query_count);
    const dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    if (selected_block_ids.scalar_type() == torch::kInt64) {
        if (valid_mask.scalar_type() == torch::kBool) {
            fused_selected_blocks_context_kernel<int64_t, bool><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int64_t>(),
                valid_mask.data_ptr<bool>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                output.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            fused_selected_blocks_context_kernel<int64_t, int32_t><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int64_t>(),
                valid_mask.data_ptr<int32_t>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                output.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        }
    } else {
        if (valid_mask.scalar_type() == torch::kBool) {
            fused_selected_blocks_context_kernel<int32_t, bool><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int32_t>(),
                valid_mask.data_ptr<bool>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                output.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            fused_selected_blocks_context_kernel<int32_t, int32_t><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int32_t>(),
                valid_mask.data_ptr<int32_t>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                output.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> fused_selected_blocks_stream_stats_cuda_launcher(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale) {
    c10::cuda::CUDAGuard device_guard(payload_words.device());

    const int query_count = static_cast<int>(queries.size(0));
    const int block_count = static_cast<int>(payload_words.size(1));
    const int head_dim = static_cast<int>(values.size(2));
    const int selected_block_count = static_cast<int>(selected_block_ids.numel());

    auto h = torch::zeros({query_count, head_dim}, queries.options().dtype(torch::kFloat32));
    auto m = torch::full({query_count}, -std::numeric_limits<float>::infinity(), queries.options().dtype(torch::kFloat32));
    auto l = torch::zeros({query_count}, queries.options().dtype(torch::kFloat32));
    auto block_max = torch::full({query_count, std::max(selected_block_count, 1)}, -std::numeric_limits<float>::infinity(), queries.options().dtype(torch::kFloat32));
    auto block_mass_num = torch::zeros({query_count, std::max(selected_block_count, 1)}, queries.options().dtype(torch::kFloat32));
    if (selected_block_count == 0) {
        return {h, m, l, block_max, block_mass_num};
    }

    const dim3 grid(query_count);
    const dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    if (selected_block_ids.scalar_type() == torch::kInt64) {
        if (valid_mask.scalar_type() == torch::kBool) {
            fused_selected_blocks_stream_stats_kernel<int64_t, bool><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int64_t>(),
                valid_mask.data_ptr<bool>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            fused_selected_blocks_stream_stats_kernel<int64_t, int32_t><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int64_t>(),
                valid_mask.data_ptr<int32_t>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        }
    } else {
        if (valid_mask.scalar_type() == torch::kBool) {
            fused_selected_blocks_stream_stats_kernel<int32_t, bool><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int32_t>(),
                valid_mask.data_ptr<bool>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            fused_selected_blocks_stream_stats_kernel<int32_t, int32_t><<<grid, block, 0, stream>>>(
                payload_words.data_ptr<int32_t>(),
                scales.data_ptr<float>(),
                bias.data_ptr<float>(),
                selected_block_ids.data_ptr<int32_t>(),
                valid_mask.data_ptr<int32_t>(),
                queries.data_ptr<float>(),
                query_group_sums.data_ptr<float>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                selected_block_count,
                block_count,
                head_dim,
                static_cast<float>(query_scale));
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, m, l, block_max, block_mass_num};
}

torch::Tensor softmax_value_context_cuda_launcher(
    torch::Tensor logits,
    torch::Tensor values,
    double query_scale) {
    c10::cuda::CUDAGuard device_guard(logits.device());

    const int query_count = static_cast<int>(logits.size(0));
    const int token_count = static_cast<int>(logits.size(1));
    const int head_dim = static_cast<int>(values.size(1));

    auto output = torch::zeros({query_count, head_dim}, logits.options().dtype(torch::kFloat32));
    if (query_count == 0 || token_count == 0 || head_dim == 0) {
        return output;
    }

    const dim3 grid(query_count);
    const dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    if (values.scalar_type() == torch::kFloat16) {
        softmax_value_context_kernel<at::Half><<<grid, block, 0, stream>>>(
            logits.data_ptr<float>(),
            values.data_ptr<at::Half>(),
            output.data_ptr<float>(),
            token_count,
            head_dim,
            static_cast<float>(query_scale));
    } else {
        softmax_value_context_kernel<float><<<grid, block, 0, stream>>>(
            logits.data_ptr<float>(),
            values.data_ptr<float>(),
            output.data_ptr<float>(),
            token_count,
            head_dim,
            static_cast<float>(query_scale));
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> softmax_value_stream_stats_cuda_launcher(
    torch::Tensor logits,
    torch::Tensor token_block_ids,
    torch::Tensor values,
    int64_t block_count,
    double query_scale) {
    c10::cuda::CUDAGuard device_guard(logits.device());

    const int query_count = static_cast<int>(logits.size(0));
    const int token_count = static_cast<int>(logits.size(1));
    const int head_dim = static_cast<int>(values.size(1));
    const int block_count_i = static_cast<int>(block_count);

    auto h = torch::zeros({query_count, head_dim}, logits.options().dtype(torch::kFloat32));
    auto m = torch::full({query_count}, -std::numeric_limits<float>::infinity(), logits.options().dtype(torch::kFloat32));
    auto l = torch::zeros({query_count}, logits.options().dtype(torch::kFloat32));
    auto block_max = torch::full({query_count, std::max(block_count_i, 1)}, -std::numeric_limits<float>::infinity(), logits.options().dtype(torch::kFloat32));
    auto block_mass_num = torch::zeros({query_count, std::max(block_count_i, 1)}, logits.options().dtype(torch::kFloat32));
    if (query_count == 0 || token_count == 0 || head_dim == 0) {
        return {h, m, l, block_max, block_mass_num};
    }

    const dim3 grid(query_count);
    const dim3 block(kThreads);
    auto stream = at::cuda::getDefaultCUDAStream();

    if (token_block_ids.scalar_type() == torch::kInt64) {
        if (values.scalar_type() == torch::kFloat16) {
            softmax_value_stream_stats_kernel<int64_t, at::Half><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(),
                token_block_ids.data_ptr<int64_t>(),
                values.data_ptr<at::Half>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                token_count,
                block_count_i,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            softmax_value_stream_stats_kernel<int64_t, float><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(),
                token_block_ids.data_ptr<int64_t>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                token_count,
                block_count_i,
                head_dim,
                static_cast<float>(query_scale));
        }
    } else {
        if (values.scalar_type() == torch::kFloat16) {
            softmax_value_stream_stats_kernel<int32_t, at::Half><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(),
                token_block_ids.data_ptr<int32_t>(),
                values.data_ptr<at::Half>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                token_count,
                block_count_i,
                head_dim,
                static_cast<float>(query_scale));
        } else {
            softmax_value_stream_stats_kernel<int32_t, float><<<grid, block, 0, stream>>>(
                logits.data_ptr<float>(),
                token_block_ids.data_ptr<int32_t>(),
                values.data_ptr<float>(),
                h.data_ptr<float>(),
                m.data_ptr<float>(),
                l.data_ptr<float>(),
                block_max.data_ptr<float>(),
                block_mass_num.data_ptr<float>(),
                token_count,
                block_count_i,
                head_dim,
                static_cast<float>(query_scale));
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, m, l, block_max, block_mass_num};
}
