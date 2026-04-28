#include <torch/extension.h>

#include <vector>

torch::Tensor fused_selected_blocks_context_cuda_launcher(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale);

std::vector<torch::Tensor> fused_selected_blocks_stream_stats_cuda_launcher(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale);

torch::Tensor softmax_value_context_cuda_launcher(
    torch::Tensor logits,
    torch::Tensor values,
    double query_scale);

std::vector<torch::Tensor> softmax_value_stream_stats_cuda_launcher(
    torch::Tensor logits,
    torch::Tensor token_block_ids,
    torch::Tensor values,
    int64_t block_count,
    double query_scale);

namespace {

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

void check_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor fused_selected_blocks_context_cuda(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale) {
    check_cuda(payload_words, "payload_words");
    check_cuda(scales, "scales");
    check_cuda(bias, "bias");
    check_cuda(selected_block_ids, "selected_block_ids");
    check_cuda(valid_mask, "valid_mask");
    check_cuda(queries, "queries");
    check_cuda(query_group_sums, "query_group_sums");
    check_cuda(values, "values");

    auto payload = payload_words.contiguous();
    auto scales_c = scales.contiguous();
    auto bias_c = bias.contiguous();
    auto selected = selected_block_ids.contiguous();
    auto valid = valid_mask.contiguous();
    auto queries_c = queries.contiguous();
    auto sums_c = query_group_sums.contiguous();
    auto values_c = values.contiguous();

    TORCH_CHECK(payload.scalar_type() == torch::kInt32, "payload_words must be int32");
    TORCH_CHECK(scales_c.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(bias_c.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(queries_c.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(sums_c.scalar_type() == torch::kFloat32, "query_group_sums must be float32");
    TORCH_CHECK(values_c.scalar_type() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(selected.scalar_type() == torch::kInt64 || selected.scalar_type() == torch::kInt32, "selected_block_ids must be int32 or int64");
    TORCH_CHECK(valid.scalar_type() == torch::kBool || valid.scalar_type() == torch::kInt || valid.scalar_type() == torch::kInt32, "valid_mask must be bool/int32");

    TORCH_CHECK(payload.dim() == 4, "payload_words must have shape [num_groups, block_count, block_size, words_per_group]");
    TORCH_CHECK(scales_c.dim() == 3, "scales must have shape [num_groups, block_count, block_size]");
    TORCH_CHECK(bias_c.dim() == 3, "bias must have shape [num_groups, block_count, block_size]");
    TORCH_CHECK(valid.dim() == 2, "valid_mask must have shape [block_count, block_size]");
    TORCH_CHECK(queries_c.dim() == 3, "queries must have shape [query_count, num_groups, group_size]");
    TORCH_CHECK(sums_c.dim() == 2, "query_group_sums must have shape [query_count, num_groups]");
    TORCH_CHECK(values_c.dim() == 3, "values must have shape [block_count, block_size, head_dim]");

    TORCH_CHECK(payload.size(0) == 8, "payload_words num_groups must be 8");
    TORCH_CHECK(payload.size(2) == 16, "payload_words block_size must be 16");
    TORCH_CHECK(payload.size(3) == 8, "payload_words words_per_group must be 8");
    TORCH_CHECK(queries_c.size(1) == 8, "queries num_groups must be 8");
    TORCH_CHECK(queries_c.size(2) == 32, "queries group_size must be 32");
    TORCH_CHECK(values_c.size(2) <= 256, "values head_dim must be <= 256");
    TORCH_CHECK(selected.numel() <= 16, "selected_block_ids length must be <= 16 for native Stage 9 kernel");

    TORCH_CHECK(scales_c.size(0) == payload.size(0) && scales_c.size(1) == payload.size(1) && scales_c.size(2) == payload.size(2), "scales shape mismatch");
    TORCH_CHECK(bias_c.sizes() == scales_c.sizes(), "bias shape mismatch");
    TORCH_CHECK(valid.size(0) == payload.size(1) && valid.size(1) == payload.size(2), "valid_mask shape mismatch");
    TORCH_CHECK(values_c.size(0) == payload.size(1) && values_c.size(1) == payload.size(2), "values shape mismatch");
    TORCH_CHECK(sums_c.size(0) == queries_c.size(0) && sums_c.size(1) == queries_c.size(1), "query_group_sums shape mismatch");

    return fused_selected_blocks_context_cuda_launcher(
        payload,
        scales_c,
        bias_c,
        selected,
        valid,
        queries_c,
        sums_c,
        values_c,
        query_scale);
}

std::vector<torch::Tensor> fused_selected_blocks_stream_stats_cuda(
    torch::Tensor payload_words,
    torch::Tensor scales,
    torch::Tensor bias,
    torch::Tensor selected_block_ids,
    torch::Tensor valid_mask,
    torch::Tensor queries,
    torch::Tensor query_group_sums,
    torch::Tensor values,
    double query_scale) {
    check_cuda(payload_words, "payload_words");
    check_cuda(scales, "scales");
    check_cuda(bias, "bias");
    check_cuda(selected_block_ids, "selected_block_ids");
    check_cuda(valid_mask, "valid_mask");
    check_cuda(queries, "queries");
    check_cuda(query_group_sums, "query_group_sums");
    check_cuda(values, "values");

    auto payload = payload_words.contiguous();
    auto scales_c = scales.contiguous();
    auto bias_c = bias.contiguous();
    auto selected = selected_block_ids.contiguous();
    auto valid = valid_mask.contiguous();
    auto queries_c = queries.contiguous();
    auto sums_c = query_group_sums.contiguous();
    auto values_c = values.contiguous();

    TORCH_CHECK(payload.scalar_type() == torch::kInt32, "payload_words must be int32");
    TORCH_CHECK(scales_c.scalar_type() == torch::kFloat32, "scales must be float32");
    TORCH_CHECK(bias_c.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(queries_c.scalar_type() == torch::kFloat32, "queries must be float32");
    TORCH_CHECK(sums_c.scalar_type() == torch::kFloat32, "query_group_sums must be float32");
    TORCH_CHECK(values_c.scalar_type() == torch::kFloat32, "values must be float32");
    TORCH_CHECK(selected.scalar_type() == torch::kInt64 || selected.scalar_type() == torch::kInt32, "selected_block_ids must be int32 or int64");
    TORCH_CHECK(valid.scalar_type() == torch::kBool || valid.scalar_type() == torch::kInt || valid.scalar_type() == torch::kInt32, "valid_mask must be bool/int32");

    TORCH_CHECK(payload.dim() == 4, "payload_words must have shape [num_groups, block_count, block_size, words_per_group]");
    TORCH_CHECK(scales_c.dim() == 3, "scales must have shape [num_groups, block_count, block_size]");
    TORCH_CHECK(bias_c.dim() == 3, "bias must have shape [num_groups, block_count, block_size]");
    TORCH_CHECK(valid.dim() == 2, "valid_mask must have shape [block_count, block_size]");
    TORCH_CHECK(queries_c.dim() == 3, "queries must have shape [query_count, num_groups, group_size]");
    TORCH_CHECK(sums_c.dim() == 2, "query_group_sums must have shape [query_count, num_groups]");
    TORCH_CHECK(values_c.dim() == 3, "values must have shape [block_count, block_size, head_dim]");

    TORCH_CHECK(payload.size(0) == 8, "payload_words num_groups must be 8");
    TORCH_CHECK(payload.size(2) == 16, "payload_words block_size must be 16");
    TORCH_CHECK(payload.size(3) == 8, "payload_words words_per_group must be 8");
    TORCH_CHECK(queries_c.size(1) == 8, "queries num_groups must be 8");
    TORCH_CHECK(queries_c.size(2) == 32, "queries group_size must be 32");
    TORCH_CHECK(values_c.size(2) <= 256, "values head_dim must be <= 256");
    TORCH_CHECK(selected.numel() <= 16, "selected_block_ids length must be <= 16 for native Stage 9 kernel");

    TORCH_CHECK(scales_c.size(0) == payload.size(0) && scales_c.size(1) == payload.size(1) && scales_c.size(2) == payload.size(2), "scales shape mismatch");
    TORCH_CHECK(bias_c.sizes() == scales_c.sizes(), "bias shape mismatch");
    TORCH_CHECK(valid.size(0) == payload.size(1) && valid.size(1) == payload.size(2), "valid_mask shape mismatch");
    TORCH_CHECK(values_c.size(0) == payload.size(1) && values_c.size(1) == payload.size(2), "values shape mismatch");
    TORCH_CHECK(sums_c.size(0) == queries_c.size(0) && sums_c.size(1) == queries_c.size(1), "query_group_sums shape mismatch");

    return fused_selected_blocks_stream_stats_cuda_launcher(
        payload,
        scales_c,
        bias_c,
        selected,
        valid,
        queries_c,
        sums_c,
        values_c,
        query_scale);
}

torch::Tensor softmax_value_context_cuda(
    torch::Tensor logits,
    torch::Tensor values,
    double query_scale) {
    check_cuda(logits, "logits");
    check_cuda(values, "values");

    auto logits_c = logits.contiguous();
    auto values_c = values.contiguous();

    TORCH_CHECK(logits_c.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(
        values_c.scalar_type() == torch::kFloat32
            || values_c.scalar_type() == torch::kFloat16,
        "values must be float32/float16");
    TORCH_CHECK(logits_c.dim() == 2, "logits must have shape [query_count, token_count]");
    TORCH_CHECK(values_c.dim() == 2, "values must have shape [token_count, head_dim]");
    TORCH_CHECK(logits_c.size(1) == values_c.size(0), "logits/value token dimension mismatch");
    TORCH_CHECK(values_c.size(1) <= 256, "values head_dim must be <= 256");

    return softmax_value_context_cuda_launcher(
        logits_c,
        values_c,
        query_scale);
}

std::vector<torch::Tensor> softmax_value_stream_stats_cuda(
    torch::Tensor logits,
    torch::Tensor token_block_ids,
    torch::Tensor values,
    int64_t block_count,
    double query_scale) {
    check_cuda(logits, "logits");
    check_cuda(token_block_ids, "token_block_ids");
    check_cuda(values, "values");

    auto logits_c = logits.contiguous();
    auto token_block_ids_c = token_block_ids.contiguous();
    auto values_c = values.contiguous();

    TORCH_CHECK(logits_c.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(
        token_block_ids_c.scalar_type() == torch::kInt64
            || token_block_ids_c.scalar_type() == torch::kInt32,
        "token_block_ids must be int32/int64");
    TORCH_CHECK(
        values_c.scalar_type() == torch::kFloat32
            || values_c.scalar_type() == torch::kFloat16,
        "values must be float32/float16");
    TORCH_CHECK(logits_c.dim() == 2, "logits must have shape [query_count, token_count]");
    TORCH_CHECK(token_block_ids_c.dim() == 1, "token_block_ids must have shape [token_count]");
    TORCH_CHECK(values_c.dim() == 2, "values must have shape [token_count, head_dim]");
    TORCH_CHECK(logits_c.size(1) == token_block_ids_c.size(0), "logits/token_block_ids token dimension mismatch");
    TORCH_CHECK(logits_c.size(1) == values_c.size(0), "logits/value token dimension mismatch");
    TORCH_CHECK(values_c.size(1) <= 256, "values head_dim must be <= 256");
    TORCH_CHECK(logits_c.size(1) <= 256, "logits token_count must be <= 256");
    TORCH_CHECK(block_count > 0 && block_count <= 32, "block_count must be in [1, 32]");

    return softmax_value_stream_stats_cuda_launcher(
        logits_c,
        token_block_ids_c,
        values_c,
        block_count,
        query_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_selected_blocks_context_cuda", &fused_selected_blocks_context_cuda, "Fused selected-block direct-M0 CUDA");
    m.def("fused_selected_blocks_stream_stats_cuda", &fused_selected_blocks_stream_stats_cuda, "Fused selected-block direct-M0 CUDA stream stats");
    m.def("softmax_value_context_cuda", &softmax_value_context_cuda, "Fused softmax/value reduction CUDA");
    m.def("softmax_value_stream_stats_cuda", &softmax_value_stream_stats_cuda, "Fused softmax/value/block-stats CUDA");
}
