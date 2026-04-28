#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kBlockSize = 16;

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

template <typename QueryT>
__device__ __forceinline__ float load_query(const QueryT* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_query<at::Half>(const at::Half* ptr, int idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ __forceinline__ float load_query<at::BFloat16>(const at::BFloat16* ptr, int idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

__device__ __forceinline__ float load_half(const at::Half* ptr, int idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <typename T>
__device__ __forceinline__ float load_16(const T* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_16<at::Half>(const at::Half* ptr, int idx) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
}

template <>
__device__ __forceinline__ float load_16<at::BFloat16>(const at::BFloat16* ptr, int idx) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
}

template <typename QueryT, typename KeyT, typename ValueT>
__global__ void mixedv_partial_kernel(
    const int8_t* __restrict__ keys_int8,
    const float* __restrict__ keys_scale,
    const float* __restrict__ keys_zp,
    const KeyT* __restrict__ keys_fp16,
    const int32_t* __restrict__ key_block_slots,
    const int32_t* __restrict__ topk_mask,
    const uint8_t* __restrict__ values_packed,
    const at::Half* __restrict__ values_scales,
    const at::Half* __restrict__ values_zeros,
    const ValueT* __restrict__ values_fp16_scratch,
    const int32_t* __restrict__ value_fp16_mask,
    const int32_t* __restrict__ value_block_slots,
    const QueryT* __restrict__ q_all,
    const float* __restrict__ int8_token_scores,
    float* __restrict__ m_part,
    float* __restrict__ l_part,
    float* __restrict__ acc_part,
    int kv_heads,
    int tokens,
    int head_dim,
    int d_v,
    int groups,
    int fp16_key_tokens,
    int q_heads,
    int blocks,
    int gqa_group,
    int group_size,
    int scratch_tokens,
    int score_blocks,
    int num_splits,
    int blocks_per_split,
    int last_block_valid,
    bool use_score_cache,
    float q_scale) {
    __shared__ float s_scores[kBlockSize];
    __shared__ float s_weights[kBlockSize];
    __shared__ float s_reduce[kThreads];

    const int prog = blockIdx.x;
    const int qh = prog / num_splits;
    const int split = prog - qh * num_splits;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }

    const int kvh = min(qh / gqa_group, kv_heads - 1);
    const int block_start = split * blocks_per_split;
    const int block_end = min(block_start + blocks_per_split, blocks);

    float m = -INFINITY;
    float l = 0.0f;
    float acc = 0.0f;

    for (int bid = block_start; bid < block_end; ++bid) {
        const int score_tok = tid >> 4;  // 16 lanes cooperate per token.
        const int score_lane = tid & 15;
        float dot_part = 0.0f;
        const int use_fp16_key_for_block = topk_mask[qh * blocks + bid] != 0;
        if (score_tok < kBlockSize) {
            const int tok = bid * kBlockSize + score_tok;
            const bool valid_tok = (bid != blocks - 1) || (score_tok < last_block_valid);
            if (valid_tok) {
                const int key_base = (kvh * tokens + tok) * head_dim;
                const int scale_base = (kvh * blocks + bid) * head_dim;
                const int q_base = qh * head_dim;
                const int key_slot = max(key_block_slots[bid], 0);
                const int key_tok = key_slot * kBlockSize + score_tok;
                const int fp16_key_base = (kvh * fp16_key_tokens + key_tok) * head_dim;
                for (int d = score_lane; d < head_dim; d += 16) {
                    const float q = load_query<QueryT>(q_all, q_base + d);
                    float k;
                    if (use_fp16_key_for_block) {
                        k = load_16<KeyT>(keys_fp16, fp16_key_base + d);
                    } else {
                        k = static_cast<float>(keys_int8[key_base + d]) * keys_scale[scale_base + d]
                            + keys_zp[scale_base + d];
                    }
                    dot_part += q * k;
                }
            }
        }
        // Reduce 16 lanes per token within each half-warp. The previous
        // shared-memory reduction synchronized the full 256-thread block on
        // every offset even though each score only spans 16 lanes.
        for (int offset = 8; offset > 0; offset >>= 1) {
            dot_part += __shfl_down_sync(0xffffffff, dot_part, offset, 16);
        }
        if (score_lane == 0 && score_tok < kBlockSize) {
            const bool valid_tok = (bid != blocks - 1) || (score_tok < last_block_valid);
            s_scores[score_tok] = valid_tok ? dot_part * q_scale : -INFINITY;
        }
        __syncthreads();

        float block_max = -INFINITY;
        if (tid < 32) {
            float score = tid < kBlockSize ? s_scores[tid] : -INFINITY;
            for (int offset = 16; offset > 0; offset >>= 1) {
                score = fmaxf(score, __shfl_down_sync(0xffffffff, score, offset));
            }
            if (tid == 0) {
                s_reduce[0] = score;
            }
        }
        __syncthreads();
        block_max = s_reduce[0];
        const float new_m = fmaxf(m, block_max);
        const float alpha = expf(m - new_m);
        if (tid < d_v) {
            acc *= alpha;
        }
        l *= alpha;

        float weight = 0.0f;
        if (tid < kBlockSize && isfinite(s_scores[tid])) {
            weight = expf(s_scores[tid] - new_m);
            s_weights[tid] = weight;
        } else if (tid < kBlockSize) {
            s_weights[tid] = 0.0f;
        }
        __syncthreads();
        float weight_sum = tid < kBlockSize ? weight : 0.0f;
        if (tid < 32) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                weight_sum += __shfl_down_sync(0xffffffff, weight_sum, offset);
            }
            if (tid == 0) {
                s_reduce[0] = weight_sum;
            }
        }
        __syncthreads();
        l += s_reduce[0];

        if (tid < d_v) {
            const int use_fp16_value = value_fp16_mask[qh * blocks + bid] != 0;
            const int slot = max(value_block_slots[bid], 0);
            float block_acc = 0.0f;
            for (int local_tok = 0; local_tok < kBlockSize; ++local_tok) {
                const float w = s_weights[local_tok];
                if (w == 0.0f) {
                    continue;
                }
                float v;
                if (use_fp16_value) {
                    const int idx = (kvh * scratch_tokens + slot * kBlockSize + local_tok) * d_v + tid;
                    v = load_16<ValueT>(values_fp16_scratch, idx);
                } else {
                    const int tok = bid * kBlockSize + local_tok;
                    const int packed_idx = (kvh * tokens + tok) * (d_v / 2) + (tid / 2);
                    const uint8_t packed = values_packed[packed_idx];
                    const int code = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
                    const int group = tid / group_size;
                    const int scale_idx = (kvh * tokens + tok) * groups + group;
                    v = static_cast<float>(code) * load_half(values_scales, scale_idx)
                        + load_half(values_zeros, scale_idx);
                }
                block_acc += w * v;
            }
            acc += block_acc;
        }
        m = new_m;
        __syncthreads();
    }

    const int part = qh * num_splits + split;
    if (tid == 0) {
        m_part[part] = m;
        l_part[part] = l;
    }
    if (tid < d_v) {
        acc_part[part * d_v + tid] = acc;
    }
}

template <typename QueryT>
__global__ void score_blocks_kernel(
    const int8_t* __restrict__ keys_int8,
    const float* __restrict__ keys_scale,
    const float* __restrict__ keys_zp,
    const QueryT* __restrict__ q_all,
    float* __restrict__ m_out,
    float* __restrict__ s_out,
    int kv_heads,
    int tokens,
    int head_dim,
    int q_heads,
    int blocks,
    int gqa_group,
    int blocks_per_chunk,
    float q_scale) {
    __shared__ float s_scores[kBlockSize];

    const int prog = blockIdx.x;
    const int chunks = (blocks + blocks_per_chunk - 1) / blocks_per_chunk;
    const int qh = prog / chunks;
    const int chunk = prog - qh * chunks;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }
    const int kvh = min(qh / gqa_group, kv_heads - 1);
    const int q_base = qh * head_dim;
    const int block_start = chunk * blocks_per_chunk;
    const int block_end = min(block_start + blocks_per_chunk, blocks);

    for (int bid = block_start; bid < block_end; ++bid) {
        const int score_tok = tid >> 4;
        const int score_lane = tid & 15;
        float dot_part = 0.0f;
        if (score_tok < kBlockSize) {
            const int tok = bid * kBlockSize + score_tok;
            const int key_base = (kvh * tokens + tok) * head_dim;
            const int scale_base = (kvh * blocks + bid) * head_dim;
            for (int d = score_lane; d < head_dim; d += 16) {
                const float q = load_query<QueryT>(q_all, q_base + d);
                const float k = static_cast<float>(keys_int8[key_base + d]) * keys_scale[scale_base + d]
                    + keys_zp[scale_base + d];
                dot_part += q * k;
            }
        }
        for (int offset = 8; offset > 0; offset >>= 1) {
            dot_part += __shfl_down_sync(0xffffffff, dot_part, offset, 16);
        }
        if (score_lane == 0 && score_tok < kBlockSize) {
            s_scores[score_tok] = dot_part * q_scale;
        }
        __syncthreads();

        if (tid == 0) {
            float m = -INFINITY;
            for (int i = 0; i < kBlockSize; ++i) {
                m = fmaxf(m, s_scores[i]);
            }
            float s = 0.0f;
            for (int i = 0; i < kBlockSize; ++i) {
                s += expf(s_scores[i] - m);
            }
            const int out = qh * blocks + bid;
            m_out[out] = m;
            s_out[out] = s;
        }
        __syncthreads();
    }
}

__global__ void mixedv_reduce_kernel(
    const float* __restrict__ m_part,
    const float* __restrict__ l_part,
    const float* __restrict__ acc_part,
    float* __restrict__ output,
    int q_heads,
    int d_v,
    int num_splits) {
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }

    float m_global = -INFINITY;
    for (int split = 0; split < num_splits; ++split) {
        m_global = fmaxf(m_global, m_part[qh * num_splits + split]);
    }

    float l_total = 0.0f;
    float acc_total = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
        const int part = qh * num_splits + split;
        const float scale = expf(m_part[part] - m_global);
        l_total += l_part[part] * scale;
        if (tid < d_v) {
            acc_total += acc_part[part * d_v + tid] * scale;
        }
    }
    if (tid < d_v) {
        output[qh * d_v + tid] = acc_total / fmaxf(l_total, 1e-20f);
    }
}

__global__ void adaptive_topk_kernel(
    const float* __restrict__ m_b,
    const float* __restrict__ s_b,
    bool* __restrict__ topk_mask,
    int32_t* __restrict__ k_star,
    float* __restrict__ tail_mass,
    float* __restrict__ tau_actual,
    float* __restrict__ mass_frac,
    int64_t* __restrict__ sorted_idx,
    float* __restrict__ sorted_cumsum,
    int q_heads,
    int blocks,
    int hi,
    int lo,
    float tau_cov) {
    __shared__ float s_reduce[kThreads];
    __shared__ int s_reduce_idx[kThreads];
    __shared__ float s_total;
    __shared__ int s_final_k;

    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    if (qh >= q_heads) {
        return;
    }
    const int row = qh * blocks;
    const int sorted_row = qh * hi;

    float local_max = -INFINITY;
    for (int bid = tid; bid < blocks; bid += blockDim.x) {
        local_max = fmaxf(local_max, m_b[row + bid]);
    }
    const float m_global = block_reduce_max(local_max, s_reduce);

    float local_sum = 0.0f;
    for (int bid = tid; bid < blocks; bid += blockDim.x) {
        const float s = fmaxf(s_b[row + bid], 1.0e-30f);
        const float mass = expf(logf(s) + m_b[row + bid] - m_global);
        mass_frac[row + bid] = mass;
        topk_mask[row + bid] = false;
        local_sum += mass;
    }
    const float total = fmaxf(block_reduce_sum(local_sum, s_reduce), 1.0e-30f);
    if (tid == 0) {
        s_total = total;
    }
    __syncthreads();

    for (int bid = tid; bid < blocks; bid += blockDim.x) {
        mass_frac[row + bid] = mass_frac[row + bid] / s_total;
    }
    __syncthreads();

    float running = 0.0f;
    int first_reach = 0;
    for (int k = 0; k < hi; ++k) {
        float best_val = -INFINITY;
        int best_idx = -1;
        for (int bid = tid; bid < blocks; bid += blockDim.x) {
            if (!topk_mask[row + bid]) {
                const float value = mass_frac[row + bid];
                if (value > best_val || (value == best_val && (best_idx < 0 || bid < best_idx))) {
                    best_val = value;
                    best_idx = bid;
                }
            }
        }
        s_reduce[tid] = best_val;
        s_reduce_idx[tid] = best_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_val = s_reduce[tid + stride];
                const int other_idx = s_reduce_idx[tid + stride];
                if (other_val > s_reduce[tid]
                    || (other_val == s_reduce[tid]
                        && other_idx >= 0
                        && (s_reduce_idx[tid] < 0 || other_idx < s_reduce_idx[tid]))) {
                    s_reduce[tid] = other_val;
                    s_reduce_idx[tid] = other_idx;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int idx = s_reduce_idx[0];
            const float value = fmaxf(s_reduce[0], 0.0f);
            topk_mask[row + idx] = true;
            sorted_idx[sorted_row + k] = static_cast<int64_t>(idx);
            running += value;
            sorted_cumsum[sorted_row + k] = running;
            if (first_reach == 0 && running >= tau_cov) {
                first_reach = k + 1;
            }
        }
        __syncthreads();
        if (tid == 0 && k == hi - 1) {
            int ks = first_reach == 0 ? hi : first_reach;
            ks = max(lo, min(ks, hi));
            s_final_k = ks;
            k_star[qh] = ks;
            const float tau = sorted_cumsum[sorted_row + ks - 1];
            tau_actual[qh] = tau;
            tail_mass[qh] = fmaxf(1.0f - tau, 0.0f);
        }
        __syncthreads();
    }

    for (int k = tid + s_final_k; k < hi; k += blockDim.x) {
        const int idx = static_cast<int>(sorted_idx[sorted_row + k]);
        topk_mask[row + idx] = false;
    }
}

template <typename QueryT, typename KeyT, typename ValueT>
void launch_partial(
    const torch::Tensor& keys_int8,
    const torch::Tensor& keys_scale,
    const torch::Tensor& keys_zero_points,
    const torch::Tensor& keys_fp16,
    const torch::Tensor& key_block_slots,
    const torch::Tensor& topk_mask,
    const torch::Tensor& values_int4_packed,
    const torch::Tensor& values_int4_scales,
    const torch::Tensor& values_int4_zeros,
    const torch::Tensor& values_fp16_scratch,
    const torch::Tensor& value_fp16_mask,
    const torch::Tensor& value_block_slots,
    const torch::Tensor& q_all,
    const torch::Tensor& int8_token_scores,
    torch::Tensor& m_part,
    torch::Tensor& l_part,
    torch::Tensor& acc_part,
    int gqa_group,
    int group_size,
    int last_block_valid,
    int num_splits,
    bool use_score_cache,
    float q_scale,
    cudaStream_t stream) {
    const int kv_heads = keys_int8.size(0);
    const int tokens = keys_int8.size(1);
    const int head_dim = keys_int8.size(2);
    const int d_v = values_int4_packed.size(2) * 2;
    const int groups = values_int4_scales.size(2);
    const int q_heads = q_all.size(0);
    const int blocks = keys_scale.size(1);
    const int fp16_key_tokens = keys_fp16.size(1);
    const int scratch_tokens = values_fp16_scratch.size(1);
    const int score_blocks = use_score_cache ? int8_token_scores.size(1) : 0;
    const int blocks_per_split = (blocks + num_splits - 1) / num_splits;

    mixedv_partial_kernel<QueryT, KeyT, ValueT><<<q_heads * num_splits, kThreads, 0, stream>>>(
        keys_int8.data_ptr<int8_t>(),
        keys_scale.data_ptr<float>(),
        keys_zero_points.data_ptr<float>(),
        keys_fp16.data_ptr<KeyT>(),
        key_block_slots.data_ptr<int32_t>(),
        topk_mask.data_ptr<int32_t>(),
        values_int4_packed.data_ptr<uint8_t>(),
        values_int4_scales.data_ptr<at::Half>(),
        values_int4_zeros.data_ptr<at::Half>(),
        values_fp16_scratch.data_ptr<ValueT>(),
        value_fp16_mask.data_ptr<int32_t>(),
        value_block_slots.data_ptr<int32_t>(),
        q_all.data_ptr<QueryT>(),
        int8_token_scores.data_ptr<float>(),
        m_part.data_ptr<float>(),
        l_part.data_ptr<float>(),
        acc_part.data_ptr<float>(),
        kv_heads,
        tokens,
        head_dim,
        d_v,
        groups,
        fp16_key_tokens,
        q_heads,
        blocks,
        gqa_group,
        group_size,
        scratch_tokens,
        score_blocks,
        num_splits,
        blocks_per_split,
        last_block_valid,
        use_score_cache,
        q_scale);
}

template <typename QueryT>
void launch_score_blocks(
    const torch::Tensor& keys_int8,
    const torch::Tensor& keys_scale,
    const torch::Tensor& keys_zero_points,
    const torch::Tensor& q_all,
    torch::Tensor& m_out,
    torch::Tensor& s_out,
    int gqa_group,
    int blocks_per_chunk,
    float q_scale,
    cudaStream_t stream) {
    const int kv_heads = keys_int8.size(0);
    const int tokens = keys_int8.size(1);
    const int head_dim = keys_int8.size(2);
    const int q_heads = q_all.size(0);
    const int blocks = keys_scale.size(1);
    const int chunks = (blocks + blocks_per_chunk - 1) / blocks_per_chunk;
    score_blocks_kernel<QueryT><<<q_heads * chunks, kThreads, 0, stream>>>(
        keys_int8.data_ptr<int8_t>(),
        keys_scale.data_ptr<float>(),
        keys_zero_points.data_ptr<float>(),
        q_all.data_ptr<QueryT>(),
        m_out.data_ptr<float>(),
        s_out.data_ptr<float>(),
        kv_heads,
        tokens,
        head_dim,
        q_heads,
        blocks,
        gqa_group,
        blocks_per_chunk,
        q_scale);
}

template <typename QueryT>
void dispatch_key_value_partial(
    const torch::Tensor& keys_int8,
    const torch::Tensor& keys_scale,
    const torch::Tensor& keys_zero_points,
    const torch::Tensor& keys_fp16,
    const torch::Tensor& key_block_slots,
    const torch::Tensor& topk_mask,
    const torch::Tensor& values_int4_packed,
    const torch::Tensor& values_int4_scales,
    const torch::Tensor& values_int4_zeros,
    const torch::Tensor& values_fp16_scratch,
    const torch::Tensor& value_fp16_mask,
    const torch::Tensor& value_block_slots,
    const torch::Tensor& q_all,
    const torch::Tensor& int8_token_scores,
    torch::Tensor& m_part,
    torch::Tensor& l_part,
    torch::Tensor& acc_part,
    int gqa_group,
    int group_size,
    int last_block_valid,
    int num_splits,
    bool use_score_cache,
    float q_scale,
    cudaStream_t stream) {
    if (keys_fp16.scalar_type() == torch::kFloat16 && values_fp16_scratch.scalar_type() == torch::kFloat16) {
        launch_partial<QueryT, at::Half, at::Half>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            gqa_group, group_size, last_block_valid, num_splits,
            use_score_cache, q_scale, stream);
    } else if (keys_fp16.scalar_type() == torch::kFloat16) {
        launch_partial<QueryT, at::Half, at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            gqa_group, group_size, last_block_valid, num_splits,
            use_score_cache, q_scale, stream);
    } else if (values_fp16_scratch.scalar_type() == torch::kFloat16) {
        launch_partial<QueryT, at::BFloat16, at::Half>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            gqa_group, group_size, last_block_valid, num_splits,
            use_score_cache, q_scale, stream);
    } else {
        launch_partial<QueryT, at::BFloat16, at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            gqa_group, group_size, last_block_valid, num_splits,
            use_score_cache, q_scale, stream);
    }
}

}  // namespace

std::vector<torch::Tensor> score_blocks_cuda_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    int64_t blocks_per_chunk) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    TORCH_CHECK(block_size == 16, "native Blackwell score v1 requires block_size=16");
    const int q_heads = q_all.size(0);
    const int blocks = keys_scale.size(1);
    auto opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat32);
    auto m_out = torch::empty({q_heads, blocks}, opts);
    auto s_out = torch::empty({q_heads, blocks}, opts);
    const int bpc = std::max<int>(1, static_cast<int>(blocks_per_chunk));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (q_all.scalar_type() == torch::kFloat32) {
        launch_score_blocks<float>(
            keys_int8, keys_scale, keys_zero_points, q_all, m_out, s_out,
            static_cast<int>(gqa_group), bpc, static_cast<float>(q_scale), stream);
    } else if (q_all.scalar_type() == torch::kFloat16) {
        launch_score_blocks<at::Half>(
            keys_int8, keys_scale, keys_zero_points, q_all, m_out, s_out,
            static_cast<int>(gqa_group), bpc, static_cast<float>(q_scale), stream);
    } else {
        launch_score_blocks<at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, q_all, m_out, s_out,
            static_cast<int>(gqa_group), bpc, static_cast<float>(q_scale), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {m_out, s_out};
}

std::vector<torch::Tensor> adaptive_topk_cuda_launcher(
    torch::Tensor m_b,
    torch::Tensor s_b,
    double tau_cov,
    int64_t k_min,
    int64_t k_max) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(m_b));
    const int q_heads = m_b.size(0);
    const int blocks = m_b.size(1);
    const int hi = std::min<int>(static_cast<int>(k_max), blocks);
    const int lo = std::min<int>(static_cast<int>(k_min), hi);

    auto f32_opts = torch::TensorOptions().device(m_b.device()).dtype(torch::kFloat32);
    auto i32_opts = torch::TensorOptions().device(m_b.device()).dtype(torch::kInt32);
    auto i64_opts = torch::TensorOptions().device(m_b.device()).dtype(torch::kInt64);
    auto bool_opts = torch::TensorOptions().device(m_b.device()).dtype(torch::kBool);

    auto topk_mask = torch::empty({q_heads, blocks}, bool_opts);
    auto k_star = torch::empty({q_heads}, i32_opts);
    auto tail_mass = torch::empty({q_heads}, f32_opts);
    auto tau_actual = torch::empty({q_heads}, f32_opts);
    auto mass_frac = torch::empty({q_heads, blocks}, f32_opts);
    auto sorted_idx = torch::empty({q_heads, hi}, i64_opts);
    auto sorted_cumsum = torch::empty({q_heads, hi}, f32_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    adaptive_topk_kernel<<<q_heads, kThreads, 0, stream>>>(
        m_b.data_ptr<float>(),
        s_b.data_ptr<float>(),
        topk_mask.data_ptr<bool>(),
        k_star.data_ptr<int32_t>(),
        tail_mass.data_ptr<float>(),
        tau_actual.data_ptr<float>(),
        mass_frac.data_ptr<float>(),
        sorted_idx.data_ptr<int64_t>(),
        sorted_cumsum.data_ptr<float>(),
        q_heads,
        blocks,
        hi,
        lo,
        static_cast<float>(tau_cov));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {topk_mask, k_star, tail_mass, tau_actual, mass_frac, sorted_idx, k_star, sorted_cumsum};
}

torch::Tensor hybrid_mixedv_split_k_cuda_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor keys_fp16,
    torch::Tensor key_block_slots,
    torch::Tensor topk_mask,
    torch::Tensor values_int4_packed,
    torch::Tensor values_int4_scales,
    torch::Tensor values_int4_zeros,
    torch::Tensor values_fp16_scratch,
    torch::Tensor value_fp16_mask,
    torch::Tensor value_block_slots,
    torch::Tensor q_all,
    torch::Tensor int8_token_scores,
    bool use_score_cache,
    int64_t gqa_group,
    int64_t block_size,
    int64_t group_size,
    double q_scale,
    int64_t last_block_valid,
    int64_t num_splits) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    const int q_heads = q_all.size(0);
    const int d_v = values_int4_packed.size(2) * 2;
    const int splits = static_cast<int>(num_splits);

    auto opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat32);
    auto m_part = torch::empty({q_heads, splits}, opts);
    auto l_part = torch::empty({q_heads, splits}, opts);
    auto acc_part = torch::empty({q_heads, splits, d_v}, opts);
    auto output = torch::empty({q_heads, d_v}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (q_all.scalar_type() == torch::kFloat32) {
        dispatch_key_value_partial<float>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    } else if (q_all.scalar_type() == torch::kFloat16) {
        dispatch_key_value_partial<at::Half>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    } else {
        dispatch_key_value_partial<at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    mixedv_reduce_kernel<<<q_heads, kThreads, 0, stream>>>(
        m_part.data_ptr<float>(),
        l_part.data_ptr<float>(),
        acc_part.data_ptr<float>(),
        output.data_ptr<float>(),
        q_heads,
        d_v,
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> hybrid_mixedv_split_k_cuda_profile_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor keys_fp16,
    torch::Tensor key_block_slots,
    torch::Tensor topk_mask,
    torch::Tensor values_int4_packed,
    torch::Tensor values_int4_scales,
    torch::Tensor values_int4_zeros,
    torch::Tensor values_fp16_scratch,
    torch::Tensor value_fp16_mask,
    torch::Tensor value_block_slots,
    torch::Tensor q_all,
    torch::Tensor int8_token_scores,
    bool use_score_cache,
    int64_t gqa_group,
    int64_t block_size,
    int64_t group_size,
    double q_scale,
    int64_t last_block_valid,
    int64_t num_splits) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(keys_int8));
    const int q_heads = q_all.size(0);
    const int d_v = values_int4_packed.size(2) * 2;
    const int splits = static_cast<int>(num_splits);

    auto opts = torch::TensorOptions().device(keys_int8.device()).dtype(torch::kFloat32);
    auto m_part = torch::empty({q_heads, splits}, opts);
    auto l_part = torch::empty({q_heads, splits}, opts);
    auto acc_part = torch::empty({q_heads, splits, d_v}, opts);
    auto output = torch::empty({q_heads, d_v}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t ev_start, ev_after_partial, ev_after_reduce;
    C10_CUDA_CHECK(cudaEventCreate(&ev_start));
    C10_CUDA_CHECK(cudaEventCreate(&ev_after_partial));
    C10_CUDA_CHECK(cudaEventCreate(&ev_after_reduce));
    C10_CUDA_CHECK(cudaEventRecord(ev_start, stream));

    if (q_all.scalar_type() == torch::kFloat32) {
        dispatch_key_value_partial<float>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    } else if (q_all.scalar_type() == torch::kFloat16) {
        dispatch_key_value_partial<at::Half>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    } else {
        dispatch_key_value_partial<at::BFloat16>(
            keys_int8, keys_scale, keys_zero_points, keys_fp16, key_block_slots, topk_mask,
            values_int4_packed, values_int4_scales, values_int4_zeros,
            values_fp16_scratch, value_fp16_mask, value_block_slots, q_all,
            int8_token_scores, m_part, l_part, acc_part,
            static_cast<int>(gqa_group), static_cast<int>(group_size),
            static_cast<int>(last_block_valid), splits, use_score_cache,
            static_cast<float>(q_scale), stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(ev_after_partial, stream));

    mixedv_reduce_kernel<<<q_heads, kThreads, 0, stream>>>(
        m_part.data_ptr<float>(),
        l_part.data_ptr<float>(),
        acc_part.data_ptr<float>(),
        output.data_ptr<float>(),
        q_heads,
        d_v,
        splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaEventRecord(ev_after_reduce, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(ev_after_reduce));

    float partial_ms = 0.0f;
    float reduce_ms = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&partial_ms, ev_start, ev_after_partial));
    C10_CUDA_CHECK(cudaEventElapsedTime(&reduce_ms, ev_after_partial, ev_after_reduce));
    C10_CUDA_CHECK(cudaEventDestroy(ev_start));
    C10_CUDA_CHECK(cudaEventDestroy(ev_after_partial));
    C10_CUDA_CHECK(cudaEventDestroy(ev_after_reduce));

    auto timing = torch::empty({2}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
    timing[0] = partial_ms;
    timing[1] = reduce_ms;
    return {output, timing};
}
