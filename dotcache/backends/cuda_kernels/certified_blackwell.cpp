#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> score_blocks_cuda_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    int64_t blocks_per_chunk);

std::vector<torch::Tensor> adaptive_topk_cuda_launcher(
    torch::Tensor m_b,
    torch::Tensor s_b,
    double tau_cov,
    int64_t k_min,
    int64_t k_max);

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
    int64_t num_splits);

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
    int64_t num_splits);

namespace {

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_i32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
}

}  // namespace

std::vector<torch::Tensor> score_blocks_cuda(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    int64_t blocks_per_chunk) {
    for (const auto& item : {
             std::pair<torch::Tensor, const char*>{keys_int8, "keys_int8"},
             {keys_scale, "keys_scale"},
             {keys_zero_points, "keys_zero_points"},
             {q_all, "q_all"},
         }) {
        check_cuda(item.first, item.second);
        check_contiguous(item.first, item.second);
    }
    TORCH_CHECK(keys_int8.scalar_type() == torch::kInt8, "keys_int8 must be int8");
    TORCH_CHECK(keys_scale.scalar_type() == torch::kFloat32, "keys_scale must be float32");
    TORCH_CHECK(keys_zero_points.scalar_type() == torch::kFloat32, "keys_zero_points must be float32");
    TORCH_CHECK(q_all.scalar_type() == torch::kFloat32 || q_all.scalar_type() == torch::kFloat16 || q_all.scalar_type() == torch::kBFloat16,
                "q_all must be float32/float16/bfloat16");
    TORCH_CHECK(keys_int8.dim() == 3, "keys_int8 must have shape [kv_heads, tokens, head_dim]");
    TORCH_CHECK(keys_scale.dim() == 3, "keys_scale must have shape [kv_heads, blocks, head_dim]");
    TORCH_CHECK(keys_zero_points.sizes() == keys_scale.sizes(), "keys_zero_points shape mismatch");
    TORCH_CHECK(q_all.dim() == 2, "q_all must have shape [q_heads, head_dim]");
    TORCH_CHECK(block_size == 16, "native Blackwell score v1 requires block_size=16");
    TORCH_CHECK(keys_int8.size(1) == keys_scale.size(1) * block_size, "token/block mismatch");
    TORCH_CHECK(keys_int8.size(2) == keys_scale.size(2), "head_dim mismatch");
    TORCH_CHECK(q_all.size(1) == keys_int8.size(2), "q head_dim mismatch");
    TORCH_CHECK(gqa_group > 0, "gqa_group must be positive");
    return score_blocks_cuda_launcher(
        keys_int8,
        keys_scale,
        keys_zero_points,
        q_all,
        gqa_group,
        block_size,
        q_scale,
        blocks_per_chunk);
}

std::vector<torch::Tensor> adaptive_topk_cuda(
    torch::Tensor m_b,
    torch::Tensor s_b,
    double tau_cov,
    int64_t k_min,
    int64_t k_max) {
    for (const auto& item : {
             std::pair<torch::Tensor, const char*>{m_b, "m_b"},
             {s_b, "s_b"},
         }) {
        check_cuda(item.first, item.second);
        check_contiguous(item.first, item.second);
    }
    TORCH_CHECK(m_b.scalar_type() == torch::kFloat32, "m_b must be float32");
    TORCH_CHECK(s_b.scalar_type() == torch::kFloat32, "s_b must be float32");
    TORCH_CHECK(m_b.dim() == 2, "m_b must have shape [q_heads, blocks]");
    TORCH_CHECK(s_b.sizes() == m_b.sizes(), "s_b shape mismatch");
    TORCH_CHECK(k_min >= 0, "k_min must be non-negative");
    TORCH_CHECK(k_max > 0, "k_max must be positive");
    TORCH_CHECK(k_max <= m_b.size(1), "k_max must not exceed block count");
    return adaptive_topk_cuda_launcher(m_b, s_b, tau_cov, k_min, k_max);
}

torch::Tensor hybrid_mixedv_split_k_cuda(
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
    for (const auto& item : {
             std::pair<torch::Tensor, const char*>{keys_int8, "keys_int8"},
             {keys_scale, "keys_scale"},
             {keys_zero_points, "keys_zero_points"},
             {keys_fp16, "keys_fp16"},
             {key_block_slots, "key_block_slots"},
             {topk_mask, "topk_mask"},
             {values_int4_packed, "values_int4_packed"},
             {values_int4_scales, "values_int4_scales"},
             {values_int4_zeros, "values_int4_zeros"},
             {values_fp16_scratch, "values_fp16_scratch"},
             {value_fp16_mask, "value_fp16_mask"},
             {value_block_slots, "value_block_slots"},
             {q_all, "q_all"},
             {int8_token_scores, "int8_token_scores"},
         }) {
        check_cuda(item.first, item.second);
        check_contiguous(item.first, item.second);
    }

    TORCH_CHECK(keys_int8.scalar_type() == torch::kInt8, "keys_int8 must be int8");
    TORCH_CHECK(keys_scale.scalar_type() == torch::kFloat32, "keys_scale must be float32");
    TORCH_CHECK(keys_zero_points.scalar_type() == torch::kFloat32, "keys_zero_points must be float32");
    TORCH_CHECK(
        keys_fp16.scalar_type() == torch::kFloat16 || keys_fp16.scalar_type() == torch::kBFloat16,
        "keys_fp16 must be float16 or bfloat16");
    TORCH_CHECK(values_int4_packed.scalar_type() == torch::kUInt8, "values_int4_packed must be uint8");
    TORCH_CHECK(values_int4_scales.scalar_type() == torch::kFloat16, "values_int4_scales must be float16");
    TORCH_CHECK(values_int4_zeros.scalar_type() == torch::kFloat16, "values_int4_zeros must be float16");
    TORCH_CHECK(
        values_fp16_scratch.scalar_type() == torch::kFloat16 || values_fp16_scratch.scalar_type() == torch::kBFloat16,
        "values_fp16_scratch must be float16 or bfloat16");
    TORCH_CHECK(q_all.scalar_type() == torch::kFloat32 || q_all.scalar_type() == torch::kFloat16 || q_all.scalar_type() == torch::kBFloat16,
                "q_all must be float32/float16/bfloat16");
    check_i32(topk_mask, "topk_mask");
    check_i32(key_block_slots, "key_block_slots");
    check_i32(value_fp16_mask, "value_fp16_mask");
    check_i32(value_block_slots, "value_block_slots");
    TORCH_CHECK(int8_token_scores.scalar_type() == torch::kFloat32, "int8_token_scores must be float32");

    TORCH_CHECK(keys_int8.dim() == 3, "keys_int8 must have shape [kv_heads, tokens, head_dim]");
    TORCH_CHECK(keys_scale.dim() == 3, "keys_scale must have shape [kv_heads, blocks, head_dim]");
    TORCH_CHECK(keys_zero_points.sizes() == keys_scale.sizes(), "keys_zero_points shape mismatch");
    TORCH_CHECK(keys_fp16.dim() == 3, "keys_fp16 must have shape [kv_heads, fp16_tokens, head_dim]");
    TORCH_CHECK(values_int4_packed.dim() == 3, "values_int4_packed must have shape [kv_heads, tokens, d_v/2]");
    TORCH_CHECK(values_int4_scales.dim() == 3, "values_int4_scales must have shape [kv_heads, tokens, groups]");
    TORCH_CHECK(values_int4_zeros.sizes() == values_int4_scales.sizes(), "values_int4_zeros shape mismatch");
    TORCH_CHECK(values_fp16_scratch.dim() == 3, "values_fp16_scratch must have shape [kv_heads, scratch_tokens, d_v]");
    TORCH_CHECK(topk_mask.dim() == 2, "topk_mask must have shape [q_heads, blocks]");
    TORCH_CHECK(value_fp16_mask.sizes() == topk_mask.sizes(), "value_fp16_mask shape mismatch");
    TORCH_CHECK(value_block_slots.dim() == 1, "value_block_slots must have shape [blocks]");
    TORCH_CHECK(key_block_slots.dim() == 1, "key_block_slots must have shape [blocks]");
    TORCH_CHECK(q_all.dim() == 2, "q_all must have shape [q_heads, head_dim]");
    TORCH_CHECK(!use_score_cache || int8_token_scores.dim() == 3,
                "int8_token_scores must have shape [q_heads, blocks, block_size]");
    TORCH_CHECK(block_size == 16, "native Blackwell v1 requires block_size=16");
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    TORCH_CHECK(keys_int8.size(2) <= 256, "head_dim must be <= 256");
    TORCH_CHECK(values_int4_packed.size(2) * 2 <= 256, "d_v must be <= 256");
    TORCH_CHECK(keys_int8.size(0) == values_int4_packed.size(0), "KV head mismatch");
    TORCH_CHECK(keys_int8.size(1) == values_int4_packed.size(1), "token mismatch");
    TORCH_CHECK(keys_scale.size(1) == topk_mask.size(1), "block mismatch");
    TORCH_CHECK(keys_scale.size(2) == keys_int8.size(2), "head_dim mismatch");
    TORCH_CHECK(keys_fp16.size(0) == keys_int8.size(0), "keys_fp16 kv head mismatch");
    TORCH_CHECK(keys_fp16.size(2) == keys_int8.size(2), "keys_fp16 head_dim mismatch");
    TORCH_CHECK(q_all.size(0) == topk_mask.size(0), "q head mismatch");
    TORCH_CHECK(q_all.size(1) == keys_int8.size(2), "q head_dim mismatch");
    TORCH_CHECK(!use_score_cache || int8_token_scores.size(0) == q_all.size(0), "score-cache q head mismatch");
    TORCH_CHECK(!use_score_cache || int8_token_scores.size(1) <= topk_mask.size(1), "score-cache block mismatch");
    TORCH_CHECK(!use_score_cache || int8_token_scores.size(2) == block_size, "score-cache block-size mismatch");
    TORCH_CHECK(value_block_slots.size(0) == topk_mask.size(1), "value_block_slots block mismatch");
    TORCH_CHECK(key_block_slots.size(0) == topk_mask.size(1), "key_block_slots block mismatch");

    return hybrid_mixedv_split_k_cuda_launcher(
        keys_int8,
        keys_scale,
        keys_zero_points,
        keys_fp16,
        key_block_slots,
        topk_mask,
        values_int4_packed,
        values_int4_scales,
        values_int4_zeros,
        values_fp16_scratch,
        value_fp16_mask,
        value_block_slots,
        q_all,
        int8_token_scores,
        use_score_cache,
        gqa_group,
        block_size,
        group_size,
        q_scale,
        last_block_valid,
	        num_splits);
}

std::vector<torch::Tensor> hybrid_mixedv_split_k_cuda_profile(
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
    // Profiling-only path. The Python wrapper feeds the same contiguous,
    // validated tensors as the production entrypoint; keeping validation out
    // avoids duplicating the large check block above.
    return hybrid_mixedv_split_k_cuda_profile_launcher(
        keys_int8,
        keys_scale,
        keys_zero_points,
        keys_fp16,
        key_block_slots,
        topk_mask,
        values_int4_packed,
        values_int4_scales,
        values_int4_zeros,
        values_fp16_scratch,
        value_fp16_mask,
        value_block_slots,
        q_all,
        int8_token_scores,
        use_score_cache,
        gqa_group,
        block_size,
        group_size,
        q_scale,
        last_block_valid,
        num_splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_blocks_cuda", &score_blocks_cuda, "Blackwell native INT8 block scorer");
    m.def("adaptive_topk_cuda", &adaptive_topk_cuda, "Blackwell native adaptive top-k selector");
    m.def("hybrid_mixedv_split_k_cuda", &hybrid_mixedv_split_k_cuda, "Blackwell native mixed INT4/FP16 split-K attention");
    m.def("hybrid_mixedv_split_k_cuda_profile", &hybrid_mixedv_split_k_cuda_profile, "Profiled Blackwell native mixed INT4/FP16 split-K attention");
}
