#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

__global__ void update_slot_table_kernel(
    const int64_t* __restrict__ loaded_blocks,
    const int32_t* __restrict__ loaded_slots,
    int64_t n_loaded,
    const int64_t* __restrict__ evicted_blocks,
    int64_t n_evicted,
    int32_t* __restrict__ slot_table,
    int64_t table_size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_evicted) {
        const int64_t bid = evicted_blocks[idx];
        if (bid >= 0 && bid < table_size) {
            slot_table[bid] = -1;
        }
    }
    if (idx < n_loaded) {
        const int64_t bid = loaded_blocks[idx];
        if (bid >= 0 && bid < table_size) {
            slot_table[bid] = loaded_slots[idx];
        }
    }
}

__global__ void scatter_packed_stage_kernel(
    const char* __restrict__ stage,
    char* __restrict__ dst,
    const int32_t* __restrict__ loaded_slots,
    int64_t n_loaded,
    int64_t kv_heads,
    int64_t dst_tokens,
    int64_t dim,
    int64_t elem_size,
    int64_t block_size,
    int64_t max_slot_blocks) {
    const int64_t elements = kv_heads * n_loaded * block_size * dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= elements) {
        return;
    }
    int64_t tmp = linear;
    const int64_t d = tmp % dim;
    tmp /= dim;
    const int64_t local_tok = tmp % block_size;
    tmp /= block_size;
    const int64_t load_idx = tmp % n_loaded;
    const int64_t kvh = tmp / n_loaded;
    const int64_t slot = static_cast<int64_t>(loaded_slots[load_idx]);
    if (slot < 0 || slot >= max_slot_blocks) {
        return;
    }
    const int64_t src_elem = ((kvh * n_loaded * block_size + load_idx * block_size + local_tok) * dim + d);
    const int64_t dst_elem = ((kvh * dst_tokens + slot * block_size + local_tok) * dim + d);
    const char* src_ptr = stage + src_elem * elem_size;
    char* dst_ptr = dst + dst_elem * elem_size;
    if (elem_size == 2) {
        *reinterpret_cast<uint16_t*>(dst_ptr) = *reinterpret_cast<const uint16_t*>(src_ptr);
    } else {
        for (int64_t b = 0; b < elem_size; ++b) {
            dst_ptr[b] = src_ptr[b];
        }
    }
}

__global__ void page_in_by_slots_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    const int32_t* __restrict__ block_slots,
    int64_t kv_heads,
    int64_t src_tokens,
    int64_t dst_tokens,
    int64_t dim,
    int64_t elem_size,
    int64_t block_size,
    int64_t active_tokens,
    int64_t n_blocks,
    int64_t max_slot_blocks) {
    const int64_t total = kv_heads * n_blocks * block_size * dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    int64_t tmp = linear;
    const int64_t d = tmp % dim;
    tmp /= dim;
    const int64_t local_tok = tmp % block_size;
    tmp /= block_size;
    const int64_t bid = tmp % n_blocks;
    const int64_t kvh = tmp / n_blocks;

    const int64_t slot = static_cast<int64_t>(block_slots[bid]);
    if (slot < 0 || slot >= max_slot_blocks) {
        return;
    }

    const int64_t src_tok = bid * block_size + local_tok;
    const int64_t dst_tok = slot * block_size + local_tok;
    if (src_tok >= active_tokens || src_tok >= src_tokens || dst_tok >= dst_tokens) {
        return;
    }

    const int64_t src_elem = ((kvh * src_tokens + src_tok) * dim + d);
    const int64_t dst_elem = ((kvh * dst_tokens + dst_tok) * dim + d);
    const char* src_ptr = src + src_elem * elem_size;
    char* dst_ptr = dst + dst_elem * elem_size;
    if (elem_size == 2) {
        *reinterpret_cast<uint16_t*>(dst_ptr) = *reinterpret_cast<const uint16_t*>(src_ptr);
    } else {
        for (int64_t b = 0; b < elem_size; ++b) {
            dst_ptr[b] = src_ptr[b];
        }
    }
}

}  // namespace

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
    int64_t active_tokens) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(dst_gpu));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t n_loaded = loaded_blocks_cpu.numel();
    const int64_t n_evicted = evicted_blocks_cpu.numel();
    const int64_t kv_heads = src_cpu.size(0);
    const int64_t src_tokens = src_cpu.size(1);
    const int64_t dst_tokens = dst_gpu.size(1);
    const int64_t dim = src_cpu.size(2);
    const int64_t elem_size = src_cpu.element_size();
    const int64_t max_slot_blocks = dst_tokens / block_size;
    int64_t bytes = 0;

    const auto* loaded_blocks = loaded_blocks_cpu.data_ptr<int64_t>();
    const auto* loaded_slots = loaded_slots_cpu.data_ptr<int32_t>();
    const char* src_base = static_cast<const char*>(src_cpu.data_ptr());
    char* dst_base = static_cast<char*>(dst_gpu.data_ptr());
    const size_t src_pitch = static_cast<size_t>(src_tokens * dim * elem_size);
    const size_t dst_pitch = static_cast<size_t>(dst_tokens * dim * elem_size);

    for (int64_t i = 0; i < n_loaded;) {
        const int64_t bid = loaded_blocks[i];
        const int64_t slot = static_cast<int64_t>(loaded_slots[i]);
        if (bid < 0 || slot < 0 || slot >= max_slot_blocks) {
            ++i;
            continue;
        }
        const int64_t start = bid * block_size;
        if (start >= active_tokens) {
            ++i;
            continue;
        }
        int64_t run_blocks = 1;
        while (i + run_blocks < n_loaded) {
            const int64_t next_bid = loaded_blocks[i + run_blocks];
            const int64_t next_slot = static_cast<int64_t>(loaded_slots[i + run_blocks]);
            if (next_bid != bid + run_blocks || next_slot != slot + run_blocks) {
                break;
            }
            if (next_slot < 0 || next_slot >= max_slot_blocks) {
                break;
            }
            const int64_t next_start = next_bid * block_size;
            if (next_start >= active_tokens) {
                break;
            }
            ++run_blocks;
        }
        const int64_t width_tokens = std::min<int64_t>(run_blocks * block_size, active_tokens - start);
        if (width_tokens <= 0) {
            ++i;
            continue;
        }
        const size_t width_bytes = static_cast<size_t>(width_tokens * dim * elem_size);
        const size_t width_full_run = static_cast<size_t>(run_blocks * block_size * dim * elem_size);
        const char* src = src_base + static_cast<size_t>(start * dim * elem_size);
        char* dst = dst_base + static_cast<size_t>(slot * block_size * dim * elem_size);
        C10_CUDA_CHECK(cudaMemcpy2DAsync(
            dst,
            dst_pitch,
            src,
            src_pitch,
            width_bytes,
            static_cast<size_t>(kv_heads),
            cudaMemcpyHostToDevice,
            stream));
        if (width_bytes < width_full_run) {
            for (int64_t h = 0; h < kv_heads; ++h) {
                char* tail = dst + static_cast<size_t>(h) * dst_pitch + width_bytes;
                C10_CUDA_CHECK(cudaMemsetAsync(
                    tail,
                    0,
                    width_full_run - width_bytes,
                    stream));
            }
        }
        bytes += static_cast<int64_t>(width_bytes) * kv_heads;
        i += run_blocks;
    }

    if (n_loaded > 0 || n_evicted > 0) {
        if (n_loaded > 0) {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                loaded_blocks_gpu.data_ptr<int64_t>(),
                loaded_blocks_cpu.data_ptr<int64_t>(),
                static_cast<size_t>(n_loaded * sizeof(int64_t)),
                cudaMemcpyHostToDevice,
                stream));
            C10_CUDA_CHECK(cudaMemcpyAsync(
                loaded_slots_gpu.data_ptr<int32_t>(),
                loaded_slots_cpu.data_ptr<int32_t>(),
                static_cast<size_t>(n_loaded * sizeof(int32_t)),
                cudaMemcpyHostToDevice,
                stream));
        }
        if (n_evicted > 0) {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                evicted_blocks_gpu.data_ptr<int64_t>(),
                evicted_blocks_cpu.data_ptr<int64_t>(),
                static_cast<size_t>(n_evicted * sizeof(int64_t)),
                cudaMemcpyHostToDevice,
                stream));
        }
        const int threads = 256;
        const int64_t n = std::max<int64_t>(n_loaded, n_evicted);
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        update_slot_table_kernel<<<blocks, threads, 0, stream>>>(
            loaded_blocks_gpu.data_ptr<int64_t>(),
            loaded_slots_gpu.data_ptr<int32_t>(),
            n_loaded,
            evicted_blocks_gpu.data_ptr<int64_t>(),
            n_evicted,
            slot_table_gpu.data_ptr<int32_t>(),
            slot_table_gpu.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return bytes;
}

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
    int64_t active_tokens) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(dst_gpu));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t n_loaded = loaded_blocks_cpu.numel();
    const int64_t n_evicted = evicted_blocks_cpu.numel();
    const int64_t kv_heads = src_cpu.size(0);
    const int64_t src_tokens = src_cpu.size(1);
    const int64_t dst_tokens = dst_gpu.size(1);
    const int64_t dim = src_cpu.size(2);
    const int64_t elem_size = src_cpu.element_size();
    const int64_t max_slot_blocks = dst_tokens / block_size;
    int64_t bytes = 0;

    if (n_loaded > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            loaded_blocks_gpu.data_ptr<int64_t>(),
            loaded_blocks_cpu.data_ptr<int64_t>(),
            static_cast<size_t>(n_loaded * sizeof(int64_t)),
            cudaMemcpyHostToDevice,
            stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(
            loaded_slots_gpu.data_ptr<int32_t>(),
            loaded_slots_cpu.data_ptr<int32_t>(),
            static_cast<size_t>(n_loaded * sizeof(int32_t)),
            cudaMemcpyHostToDevice,
            stream));
    }
    if (n_evicted > 0) {
        C10_CUDA_CHECK(cudaMemcpyAsync(
            evicted_blocks_gpu.data_ptr<int64_t>(),
            evicted_blocks_cpu.data_ptr<int64_t>(),
            static_cast<size_t>(n_evicted * sizeof(int64_t)),
            cudaMemcpyHostToDevice,
            stream));
    }

    const auto* loaded_blocks = loaded_blocks_cpu.data_ptr<int64_t>();
    const auto* loaded_slots = loaded_slots_cpu.data_ptr<int32_t>();
    const char* src_base = static_cast<const char*>(src_cpu.data_ptr());
    char* stage_base = static_cast<char*>(stage_cpu.data_ptr());
    const size_t block_bytes = static_cast<size_t>(block_size * dim * elem_size);
    const size_t src_pitch = static_cast<size_t>(src_tokens * dim * elem_size);
    const size_t stage_pitch = static_cast<size_t>(n_loaded * block_size * dim * elem_size);

    for (int64_t i = 0; i < n_loaded; ++i) {
        const int64_t bid = loaded_blocks[i];
        const int64_t slot = static_cast<int64_t>(loaded_slots[i]);
        for (int64_t h = 0; h < kv_heads; ++h) {
            char* dst = stage_base + static_cast<size_t>(h) * stage_pitch
                + static_cast<size_t>(i * block_size * dim * elem_size);
            if (bid < 0 || slot < 0 || slot >= max_slot_blocks) {
                std::memset(dst, 0, block_bytes);
                continue;
            }
            const int64_t start = bid * block_size;
            if (start >= active_tokens) {
                std::memset(dst, 0, block_bytes);
                continue;
            }
            const int64_t width_tokens = std::min<int64_t>(block_size, active_tokens - start);
            const size_t width_bytes = static_cast<size_t>(width_tokens * dim * elem_size);
            const char* src = src_base + static_cast<size_t>(h) * src_pitch
                + static_cast<size_t>(start * dim * elem_size);
            std::memcpy(dst, src, width_bytes);
            if (width_bytes < block_bytes) {
                std::memset(dst + width_bytes, 0, block_bytes - width_bytes);
            }
            bytes += static_cast<int64_t>(width_bytes);
        }
    }

    if (n_loaded > 0) {
        const size_t packed_bytes = static_cast<size_t>(kv_heads * n_loaded * block_size * dim * elem_size);
        C10_CUDA_CHECK(cudaMemcpyAsync(
            stage_gpu.data_ptr(),
            stage_cpu.data_ptr(),
            packed_bytes,
            cudaMemcpyHostToDevice,
            stream));

        const int threads = 256;
        const int64_t elements = kv_heads * n_loaded * block_size * dim;
        const int blocks = static_cast<int>((elements + threads - 1) / threads);
        scatter_packed_stage_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(stage_gpu.data_ptr()),
            static_cast<char*>(dst_gpu.data_ptr()),
            loaded_slots_gpu.data_ptr<int32_t>(),
            n_loaded,
            kv_heads,
            dst_tokens,
            dim,
            elem_size,
            block_size,
            max_slot_blocks);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (n_loaded > 0 || n_evicted > 0) {
        const int threads = 256;
        const int64_t n = std::max<int64_t>(n_loaded, n_evicted);
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        update_slot_table_kernel<<<blocks, threads, 0, stream>>>(
            loaded_blocks_gpu.data_ptr<int64_t>(),
            loaded_slots_gpu.data_ptr<int32_t>(),
            n_loaded,
            evicted_blocks_gpu.data_ptr<int64_t>(),
            n_evicted,
            slot_table_gpu.data_ptr<int32_t>(),
            slot_table_gpu.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return bytes;
}

void page_in_fp16_blocks_by_slots_cuda_launcher(
    torch::Tensor src_cpu,
    torch::Tensor dst_gpu,
    torch::Tensor block_slots_gpu,
    int64_t block_size,
    int64_t active_tokens,
    int64_t n_blocks) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(dst_gpu));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t kv_heads = src_cpu.size(0);
    const int64_t src_tokens = src_cpu.size(1);
    const int64_t dst_tokens = dst_gpu.size(1);
    const int64_t dim = src_cpu.size(2);
    const int64_t elem_size = src_cpu.element_size();
    const int64_t max_slot_blocks = dst_tokens / block_size;
    if (n_blocks <= 0 || max_slot_blocks <= 0) {
        return;
    }

    const int threads = 256;
    const int64_t total = kv_heads * n_blocks * block_size * dim;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    page_in_by_slots_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const char*>(src_cpu.data_ptr()),
        static_cast<char*>(dst_gpu.data_ptr()),
        block_slots_gpu.data_ptr<int32_t>(),
        kv_heads,
        src_tokens,
        dst_tokens,
        dim,
        elem_size,
        block_size,
        active_tokens,
        n_blocks,
        max_slot_blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
