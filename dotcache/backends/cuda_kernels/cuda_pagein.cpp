#include <torch/extension.h>

#include <cstdint>

int64_t page_in_fp16_blocks_cuda_launcher(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor loaded_blocks_cpu,
    torch::Tensor loaded_slots_cpu,
    torch::Tensor evicted_blocks_cpu,
    torch::Tensor loaded_blocks_gpu,
    torch::Tensor loaded_slots_gpu,
    torch::Tensor evicted_blocks_gpu,
    torch::Tensor slot_table_gpu,
    int64_t block_size,
    int64_t active_tokens);

int64_t page_in_fp16_blocks_packed_cuda_launcher(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor stage_cpu,
    torch::Tensor stage_gpu,
    torch::Tensor loaded_blocks_cpu,
    torch::Tensor loaded_slots_cpu,
    torch::Tensor evicted_blocks_cpu,
    torch::Tensor loaded_blocks_gpu,
    torch::Tensor loaded_slots_gpu,
    torch::Tensor evicted_blocks_gpu,
    torch::Tensor slot_table_gpu,
    int64_t block_size,
    int64_t active_tokens);

void page_in_fp16_blocks_by_slots_cuda_launcher(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor block_slots_gpu,
    int64_t block_size,
    int64_t active_tokens,
    int64_t n_blocks);

namespace {

void check_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

int64_t page_in_fp16_blocks_cuda(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor loaded_blocks_cpu,
    torch::Tensor loaded_slots_cpu,
    torch::Tensor evicted_blocks_cpu,
    torch::Tensor loaded_blocks_gpu,
    torch::Tensor loaded_slots_gpu,
    torch::Tensor evicted_blocks_gpu,
    torch::Tensor slot_table_gpu,
    int64_t block_size,
    int64_t active_tokens) {
    TORCH_CHECK(!src_cpu.is_cuda(), "src_cpu must be a CPU tensor");
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(!loaded_blocks_cpu.is_cuda(), "loaded_blocks_cpu must be a CPU tensor");
    TORCH_CHECK(!loaded_slots_cpu.is_cuda(), "loaded_slots_cpu must be a CPU tensor");
    TORCH_CHECK(!evicted_blocks_cpu.is_cuda(), "evicted_blocks_cpu must be a CPU tensor");
    TORCH_CHECK(loaded_blocks_gpu.is_cuda(), "loaded_blocks_gpu must be a CUDA tensor");
    TORCH_CHECK(loaded_slots_gpu.is_cuda(), "loaded_slots_gpu must be a CUDA tensor");
    TORCH_CHECK(evicted_blocks_gpu.is_cuda(), "evicted_blocks_gpu must be a CUDA tensor");
    TORCH_CHECK(slot_table_gpu.is_cuda(), "slot_table_gpu must be a CUDA tensor");
    check_contiguous(src_cpu, "src_cpu");
    check_contiguous(dst_gpu, "dst_gpu");
    check_contiguous(loaded_blocks_cpu, "loaded_blocks_cpu");
    check_contiguous(loaded_slots_cpu, "loaded_slots_cpu");
    check_contiguous(evicted_blocks_cpu, "evicted_blocks_cpu");
    check_contiguous(loaded_blocks_gpu, "loaded_blocks_gpu");
    check_contiguous(loaded_slots_gpu, "loaded_slots_gpu");
    check_contiguous(evicted_blocks_gpu, "evicted_blocks_gpu");
    check_contiguous(slot_table_gpu, "slot_table_gpu");

    TORCH_CHECK(
        src_cpu.scalar_type() == torch::kFloat16 || src_cpu.scalar_type() == torch::kBFloat16,
        "src_cpu must be float16 or bfloat16");
    TORCH_CHECK(dst_gpu.scalar_type() == src_cpu.scalar_type(), "dst_gpu dtype must match src_cpu");
    TORCH_CHECK(loaded_blocks_cpu.scalar_type() == torch::kInt64, "loaded_blocks_cpu must be int64");
    TORCH_CHECK(loaded_slots_cpu.scalar_type() == torch::kInt32, "loaded_slots_cpu must be int32");
    TORCH_CHECK(evicted_blocks_cpu.scalar_type() == torch::kInt64, "evicted_blocks_cpu must be int64");
    TORCH_CHECK(loaded_blocks_gpu.scalar_type() == torch::kInt64, "loaded_blocks_gpu must be int64");
    TORCH_CHECK(loaded_slots_gpu.scalar_type() == torch::kInt32, "loaded_slots_gpu must be int32");
    TORCH_CHECK(evicted_blocks_gpu.scalar_type() == torch::kInt64, "evicted_blocks_gpu must be int64");
    TORCH_CHECK(slot_table_gpu.scalar_type() == torch::kInt32, "slot_table_gpu must be int32");
    TORCH_CHECK(src_cpu.dim() == 3, "src_cpu must have shape [kv_heads, tokens, dim]");
    TORCH_CHECK(dst_gpu.dim() == 3, "dst_gpu must have shape [kv_heads, scratch_tokens, dim]");
    TORCH_CHECK(src_cpu.size(0) == dst_gpu.size(0), "kv head mismatch");
    TORCH_CHECK(src_cpu.size(2) == dst_gpu.size(2), "feature dim mismatch");
    TORCH_CHECK(loaded_blocks_cpu.dim() == 1, "loaded_blocks_cpu must be 1-D");
    TORCH_CHECK(loaded_slots_cpu.dim() == 1, "loaded_slots_cpu must be 1-D");
    TORCH_CHECK(evicted_blocks_cpu.dim() == 1, "evicted_blocks_cpu must be 1-D");
    TORCH_CHECK(loaded_blocks_gpu.dim() == 1, "loaded_blocks_gpu must be 1-D");
    TORCH_CHECK(loaded_slots_gpu.dim() == 1, "loaded_slots_gpu must be 1-D");
    TORCH_CHECK(evicted_blocks_gpu.dim() == 1, "evicted_blocks_gpu must be 1-D");
    TORCH_CHECK(loaded_blocks_cpu.numel() == loaded_slots_cpu.numel(), "loaded block/slot count mismatch");
    TORCH_CHECK(loaded_blocks_gpu.numel() >= loaded_blocks_cpu.numel(), "loaded_blocks_gpu too small");
    TORCH_CHECK(loaded_slots_gpu.numel() >= loaded_slots_cpu.numel(), "loaded_slots_gpu too small");
    TORCH_CHECK(evicted_blocks_gpu.numel() >= evicted_blocks_cpu.numel(), "evicted_blocks_gpu too small");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(active_tokens >= 0 && active_tokens <= src_cpu.size(1), "active_tokens out of range");

    return page_in_fp16_blocks_cuda_launcher(
        src_cpu,
        dst_gpu,
        loaded_blocks_cpu,
        loaded_slots_cpu,
        evicted_blocks_cpu,
        loaded_blocks_gpu,
        loaded_slots_gpu,
        evicted_blocks_gpu,
        slot_table_gpu,
        block_size,
        active_tokens);
}

int64_t page_in_fp16_blocks_packed_cuda(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor stage_cpu,
    torch::Tensor stage_gpu,
    torch::Tensor loaded_blocks_cpu,
    torch::Tensor loaded_slots_cpu,
    torch::Tensor evicted_blocks_cpu,
    torch::Tensor loaded_blocks_gpu,
    torch::Tensor loaded_slots_gpu,
    torch::Tensor evicted_blocks_gpu,
    torch::Tensor slot_table_gpu,
    int64_t block_size,
    int64_t active_tokens) {
    TORCH_CHECK(!src_cpu.is_cuda(), "src_cpu must be a CPU tensor");
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(!stage_cpu.is_cuda(), "stage_cpu must be a CPU tensor");
    TORCH_CHECK(stage_gpu.is_cuda(), "stage_gpu must be a CUDA tensor");
    TORCH_CHECK(!loaded_blocks_cpu.is_cuda(), "loaded_blocks_cpu must be a CPU tensor");
    TORCH_CHECK(!loaded_slots_cpu.is_cuda(), "loaded_slots_cpu must be a CPU tensor");
    TORCH_CHECK(!evicted_blocks_cpu.is_cuda(), "evicted_blocks_cpu must be a CPU tensor");
    TORCH_CHECK(loaded_blocks_gpu.is_cuda(), "loaded_blocks_gpu must be a CUDA tensor");
    TORCH_CHECK(loaded_slots_gpu.is_cuda(), "loaded_slots_gpu must be a CUDA tensor");
    TORCH_CHECK(evicted_blocks_gpu.is_cuda(), "evicted_blocks_gpu must be a CUDA tensor");
    TORCH_CHECK(slot_table_gpu.is_cuda(), "slot_table_gpu must be a CUDA tensor");
    check_contiguous(src_cpu, "src_cpu");
    check_contiguous(dst_gpu, "dst_gpu");
    check_contiguous(stage_cpu, "stage_cpu");
    check_contiguous(stage_gpu, "stage_gpu");
    check_contiguous(loaded_blocks_cpu, "loaded_blocks_cpu");
    check_contiguous(loaded_slots_cpu, "loaded_slots_cpu");
    check_contiguous(evicted_blocks_cpu, "evicted_blocks_cpu");
    check_contiguous(loaded_blocks_gpu, "loaded_blocks_gpu");
    check_contiguous(loaded_slots_gpu, "loaded_slots_gpu");
    check_contiguous(evicted_blocks_gpu, "evicted_blocks_gpu");
    check_contiguous(slot_table_gpu, "slot_table_gpu");

    TORCH_CHECK(
        src_cpu.scalar_type() == torch::kFloat16 || src_cpu.scalar_type() == torch::kBFloat16,
        "src_cpu must be float16 or bfloat16");
    TORCH_CHECK(dst_gpu.scalar_type() == src_cpu.scalar_type(), "dst_gpu dtype must match src_cpu");
    TORCH_CHECK(stage_cpu.scalar_type() == src_cpu.scalar_type(), "stage_cpu dtype must match src_cpu");
    TORCH_CHECK(stage_gpu.scalar_type() == src_cpu.scalar_type(), "stage_gpu dtype must match src_cpu");
    TORCH_CHECK(loaded_blocks_cpu.scalar_type() == torch::kInt64, "loaded_blocks_cpu must be int64");
    TORCH_CHECK(loaded_slots_cpu.scalar_type() == torch::kInt32, "loaded_slots_cpu must be int32");
    TORCH_CHECK(evicted_blocks_cpu.scalar_type() == torch::kInt64, "evicted_blocks_cpu must be int64");
    TORCH_CHECK(loaded_blocks_gpu.scalar_type() == torch::kInt64, "loaded_blocks_gpu must be int64");
    TORCH_CHECK(loaded_slots_gpu.scalar_type() == torch::kInt32, "loaded_slots_gpu must be int32");
    TORCH_CHECK(evicted_blocks_gpu.scalar_type() == torch::kInt64, "evicted_blocks_gpu must be int64");
    TORCH_CHECK(slot_table_gpu.scalar_type() == torch::kInt32, "slot_table_gpu must be int32");
    TORCH_CHECK(src_cpu.dim() == 3, "src_cpu must have shape [kv_heads, tokens, dim]");
    TORCH_CHECK(dst_gpu.dim() == 3, "dst_gpu must have shape [kv_heads, scratch_tokens, dim]");
    TORCH_CHECK(stage_cpu.dim() == 3, "stage_cpu must have shape [kv_heads, stage_tokens, dim]");
    TORCH_CHECK(stage_gpu.sizes() == stage_cpu.sizes(), "stage_gpu shape must match stage_cpu");
    TORCH_CHECK(src_cpu.size(0) == dst_gpu.size(0), "kv head mismatch");
    TORCH_CHECK(src_cpu.size(0) == stage_cpu.size(0), "stage kv head mismatch");
    TORCH_CHECK(src_cpu.size(2) == dst_gpu.size(2), "feature dim mismatch");
    TORCH_CHECK(src_cpu.size(2) == stage_cpu.size(2), "stage feature dim mismatch");
    TORCH_CHECK(loaded_blocks_cpu.dim() == 1, "loaded_blocks_cpu must be 1-D");
    TORCH_CHECK(loaded_slots_cpu.dim() == 1, "loaded_slots_cpu must be 1-D");
    TORCH_CHECK(evicted_blocks_cpu.dim() == 1, "evicted_blocks_cpu must be 1-D");
    TORCH_CHECK(loaded_blocks_gpu.dim() == 1, "loaded_blocks_gpu must be 1-D");
    TORCH_CHECK(loaded_slots_gpu.dim() == 1, "loaded_slots_gpu must be 1-D");
    TORCH_CHECK(evicted_blocks_gpu.dim() == 1, "evicted_blocks_gpu must be 1-D");
    TORCH_CHECK(loaded_blocks_cpu.numel() == loaded_slots_cpu.numel(), "loaded block/slot count mismatch");
    TORCH_CHECK(loaded_blocks_gpu.numel() >= loaded_blocks_cpu.numel(), "loaded_blocks_gpu too small");
    TORCH_CHECK(loaded_slots_gpu.numel() >= loaded_slots_cpu.numel(), "loaded_slots_gpu too small");
    TORCH_CHECK(evicted_blocks_gpu.numel() >= evicted_blocks_cpu.numel(), "evicted_blocks_gpu too small");
    TORCH_CHECK(stage_cpu.size(1) >= loaded_blocks_cpu.numel() * block_size, "stage tensors too small");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(active_tokens >= 0 && active_tokens <= src_cpu.size(1), "active_tokens out of range");

    return page_in_fp16_blocks_packed_cuda_launcher(
        src_cpu,
        dst_gpu,
        stage_cpu,
        stage_gpu,
        loaded_blocks_cpu,
        loaded_slots_cpu,
        evicted_blocks_cpu,
        loaded_blocks_gpu,
        loaded_slots_gpu,
        evicted_blocks_gpu,
        slot_table_gpu,
        block_size,
        active_tokens);
}

void page_in_fp16_blocks_by_slots_cuda(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor block_slots_gpu,
    int64_t block_size,
    int64_t active_tokens,
    int64_t n_blocks) {
    TORCH_CHECK(!src_cpu.is_cuda(), "src_cpu must be a CPU tensor");
    TORCH_CHECK(dst_gpu.is_cuda(), "dst_gpu must be a CUDA tensor");
    TORCH_CHECK(block_slots_gpu.is_cuda(), "block_slots_gpu must be a CUDA tensor");
    check_contiguous(src_cpu, "src_cpu");
    check_contiguous(dst_gpu, "dst_gpu");
    check_contiguous(block_slots_gpu, "block_slots_gpu");
    TORCH_CHECK(
        src_cpu.scalar_type() == torch::kFloat16 || src_cpu.scalar_type() == torch::kBFloat16,
        "src_cpu must be float16 or bfloat16");
    TORCH_CHECK(dst_gpu.scalar_type() == src_cpu.scalar_type(), "dst_gpu dtype must match src_cpu");
    TORCH_CHECK(block_slots_gpu.scalar_type() == torch::kInt32, "block_slots_gpu must be int32");
    TORCH_CHECK(src_cpu.dim() == 3, "src_cpu must have shape [kv_heads, tokens, dim]");
    TORCH_CHECK(dst_gpu.dim() == 3, "dst_gpu must have shape [kv_heads, scratch_tokens, dim]");
    TORCH_CHECK(src_cpu.size(0) == dst_gpu.size(0), "kv head mismatch");
    TORCH_CHECK(src_cpu.size(2) == dst_gpu.size(2), "feature dim mismatch");
    TORCH_CHECK(block_slots_gpu.dim() == 1, "block_slots_gpu must be 1-D");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(active_tokens >= 0 && active_tokens <= src_cpu.size(1), "active_tokens out of range");
    TORCH_CHECK(n_blocks >= 0 && n_blocks <= block_slots_gpu.numel(), "n_blocks out of range");

    page_in_fp16_blocks_by_slots_cuda_launcher(
        src_cpu,
        dst_gpu,
        block_slots_gpu,
        block_size,
        active_tokens,
        n_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("page_in_fp16_blocks_cuda", &page_in_fp16_blocks_cuda, "Batched CPU-pinned FP16 block page-in");
    m.def("page_in_fp16_blocks_packed_cuda", &page_in_fp16_blocks_packed_cuda, "Packed CPU-pinned FP16 block page-in");
    m.def("page_in_fp16_blocks_by_slots_cuda", &page_in_fp16_blocks_by_slots_cuda, "GPU slot-table CPU-pinned FP16 block page-in");
}
