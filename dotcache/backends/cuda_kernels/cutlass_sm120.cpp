#include <torch/extension.h>

#include <sstream>
#include <string>

#include <cutlass/version.h>

torch::Tensor cutlass_sm120_probe_launcher(torch::Tensor input);
torch::Tensor dequant_keys_to_fp16_t_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    int64_t block_size);
std::vector<torch::Tensor> score_certify_sm120_launcher(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    torch::Tensor correction,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    double block_epsilon);

std::string cutlass_sm120_metadata() {
    std::ostringstream out;
    out << "cutlass=" << CUTLASS_MAJOR << "." << CUTLASS_MINOR << "." << CUTLASS_PATCH
        << " target=sm120";
    return out.str();
}

torch::Tensor cutlass_sm120_probe(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    return cutlass_sm120_probe_launcher(input);
}

torch::Tensor dequant_keys_to_fp16_t(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    int64_t block_size) {
    for (const auto& item : {
             std::pair<torch::Tensor, const char*>{keys_int8, "keys_int8"},
             {keys_scale, "keys_scale"},
             {keys_zero_points, "keys_zero_points"},
         }) {
        TORCH_CHECK(item.first.is_cuda(), item.second, " must be a CUDA tensor");
        TORCH_CHECK(item.first.is_contiguous(), item.second, " must be contiguous");
    }
    TORCH_CHECK(keys_int8.scalar_type() == torch::kInt8, "keys_int8 must be int8");
    TORCH_CHECK(keys_scale.scalar_type() == torch::kFloat32, "keys_scale must be float32");
    TORCH_CHECK(keys_zero_points.scalar_type() == torch::kFloat32, "keys_zero_points must be float32");
    TORCH_CHECK(keys_int8.dim() == 3, "keys_int8 must have shape [kv_heads, tokens, head_dim]");
    TORCH_CHECK(keys_scale.dim() == 3, "keys_scale must have shape [kv_heads, blocks, head_dim]");
    TORCH_CHECK(keys_zero_points.sizes() == keys_scale.sizes(), "keys_zero_points shape mismatch");
    TORCH_CHECK(block_size == 16, "dequant_keys_to_fp16_t requires block_size=16");
    TORCH_CHECK(keys_int8.size(1) == keys_scale.size(1) * block_size, "block count mismatch");
    TORCH_CHECK(keys_scale.size(2) == keys_int8.size(2), "head_dim mismatch");
    return dequant_keys_to_fp16_t_launcher(keys_int8, keys_scale, keys_zero_points, block_size);
}

std::vector<torch::Tensor> score_certify_sm120(
    torch::Tensor keys_int8,
    torch::Tensor keys_scale,
    torch::Tensor keys_zero_points,
    torch::Tensor q_all,
    torch::Tensor correction,
    int64_t gqa_group,
    int64_t block_size,
    double q_scale,
    double block_epsilon) {
    for (const auto& item : {
             std::pair<torch::Tensor, const char*>{keys_int8, "keys_int8"},
             {keys_scale, "keys_scale"},
             {keys_zero_points, "keys_zero_points"},
             {q_all, "q_all"},
             {correction, "correction"},
         }) {
        TORCH_CHECK(item.first.is_cuda(), item.second, " must be a CUDA tensor");
        TORCH_CHECK(item.first.is_contiguous(), item.second, " must be contiguous");
    }
    TORCH_CHECK(keys_int8.scalar_type() == torch::kInt8, "keys_int8 must be int8");
    TORCH_CHECK(keys_scale.scalar_type() == torch::kFloat32, "keys_scale must be float32");
    TORCH_CHECK(keys_zero_points.scalar_type() == torch::kFloat32, "keys_zero_points must be float32");
    TORCH_CHECK(q_all.scalar_type() == torch::kFloat32, "q_all must be float32");
    TORCH_CHECK(correction.scalar_type() == torch::kFloat32, "correction must be float32");
    TORCH_CHECK(keys_int8.dim() == 3, "keys_int8 must have shape [kv_heads, tokens, head_dim]");
    TORCH_CHECK(keys_scale.dim() == 3, "keys_scale must have shape [kv_heads, blocks, head_dim]");
    TORCH_CHECK(keys_zero_points.sizes() == keys_scale.sizes(), "keys_zero_points shape mismatch");
    TORCH_CHECK(q_all.dim() == 2, "q_all must have shape [q_heads, head_dim]");
    TORCH_CHECK(correction.dim() == 2, "correction must have shape [kv_heads, blocks]");
    TORCH_CHECK(block_size == 16, "score_certify_sm120 v0 requires block_size=16");
    TORCH_CHECK(keys_int8.size(2) == q_all.size(1), "head_dim mismatch");
    TORCH_CHECK(keys_scale.size(2) == keys_int8.size(2), "scale head_dim mismatch");
    TORCH_CHECK(keys_int8.size(1) / block_size == keys_scale.size(1), "block count mismatch");
    TORCH_CHECK(correction.size(0) == keys_int8.size(0), "correction KV head mismatch");
    TORCH_CHECK(correction.size(1) == keys_scale.size(1), "correction block mismatch");
    TORCH_CHECK(gqa_group > 0, "gqa_group must be positive");

    return score_certify_sm120_launcher(
        keys_int8,
        keys_scale,
        keys_zero_points,
        q_all,
        correction,
        gqa_group,
        block_size,
        q_scale,
        block_epsilon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_sm120_metadata", &cutlass_sm120_metadata, "CUTLASS SM120 backend metadata");
    m.def("cutlass_sm120_probe", &cutlass_sm120_probe, "CUTLASS SM120 CUDA ABI probe");
    m.def("dequant_keys_to_fp16_t", &dequant_keys_to_fp16_t, "Dequantize asymmetric INT8 keys to [kv, head_dim, tokens] FP16");
    m.def("score_certify_sm120", &score_certify_sm120, "SM120 score/certify phase-1 prototype");
}
