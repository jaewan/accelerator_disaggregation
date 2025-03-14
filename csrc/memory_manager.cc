#include "memory_manager.h"
#include <mutex>
#include <unordered_map>
#include "flat_hash_map"
#include <algorithm>
#include <numeric>
#include <list>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace memory_manager {

namespace {
    // Global state
    std::mutex g_mutex;
    std::unordered_map<void*, at::Tensor> g_remote_tensors;
    MemoryConfig g_config;
    bool g_initialized = false;

    // Memory pool implementation
    struct MemoryBlock {
        void* ptr;
        size_t size;
        std::chrono::steady_clock::time_point last_used;

        MemoryBlock(void* p, size_t s)
            : ptr(p), size(s), last_used(std::chrono::steady_clock::now()) {}
    };

    // Structure to track memory allocations by size
    struct MemoryPool {
        // Free blocks organized by size buckets
        std::unordered_map<size_t, std::list<MemoryBlock>> free_blocks;

        // Total memory allocated
        size_t total_allocated = 0;
        size_t peak_allocated = 0;

        // Statistics
        size_t transfer_bytes_to_remote = 0;
        size_t transfer_bytes_from_remote = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

    std::unique_ptr<MemoryPool> g_memory_pool;

    // Round size up to nearest power of 2 for better memory reuse
    size_t round_size_up(size_t size) {
        size_t rounded = 1;
        while (rounded < size) {
            rounded <<= 1;
        }
        return rounded;
    }
}

// Initialize memory management
void init(const MemoryConfig& config) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_config = config;

    if (!g_memory_pool) {
        g_memory_pool = std::make_unique<MemoryPool>();
    }

    g_initialized = true;
}

// Tensor registration and tracking
void register_tensor(void* data_ptr, const at::Tensor& tensor) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_remote_tensors[data_ptr] = tensor;
}

void unregister_tensor(void* data_ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_remote_tensors.erase(data_ptr);
}

bool is_remote_tensor(void* data_ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_remote_tensors.find(data_ptr) != g_remote_tensors.end();
}

at::Tensor get_tensor(void* data_ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_remote_tensors.find(data_ptr);
    if (it == g_remote_tensors.end()) {
        throw std::runtime_error("Tensor not found in remote tensor registry");
    }
    return it->second;
}

// Memory pool management
void* allocate(size_t size, rpc_client::Error* error) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        init();
    }

    if (!g_config.use_memory_pool) {
        return rpc_client::alloc(size, error);
    }

    // Round size up for better reuse
    size_t rounded_size = round_size_up(size);

    // Try to find a free block of suitable size
    if (!g_memory_pool->free_blocks.empty()) {
        auto& blocks = g_memory_pool->free_blocks;

        // Find the smallest bucket that can fit this size
        auto it = blocks.lower_bound(rounded_size);
        if (it != blocks.end() && !it->second.empty()) {
            // Reuse an existing block
            MemoryBlock block = it->second.front();
            it->second.pop_front();

            // If the bucket is now empty, remove it
            if (it->second.empty()) {
                blocks.erase(it);
            }

            g_memory_pool->cache_hits++;
            if (error) *error = rpc_client::Error::ok();
            return block.ptr;
        }
    }

    // No suitable free block found, allocate a new one
    g_memory_pool->cache_misses++;
    rpc_client::Error alloc_error;
    void* ptr = rpc_client::alloc(rounded_size, &alloc_error);

    if (alloc_error.is_ok()) {
        g_memory_pool->total_allocated += rounded_size;
        g_memory_pool->peak_allocated = std::max(g_memory_pool->peak_allocated,
                                               g_memory_pool->total_allocated);
    }

    if (error) *error = alloc_error;
    return ptr;
}

void free(void* ptr) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized || !g_config.use_memory_pool) {
        rpc_client::free(ptr);
        return;
    }

    // Find the tensor to get its size
    auto it = g_remote_tensors.find(ptr);
    if (it == g_remote_tensors.end()) {
        // We don't know its size, so just free it directly
        rpc_client::free(ptr);
        return;
    }

    // Get the tensor size
    at::Tensor& tensor = it->second;
    size_t tensor_size = tensor.nbytes();
    size_t rounded_size = round_size_up(tensor_size);

    // Add to the free list if pool isn't too large
    size_t current_pool_size = std::accumulate(
        g_memory_pool->free_blocks.begin(),
        g_memory_pool->free_blocks.end(),
        0ULL,
        [](size_t sum, const auto& pair) {
            return sum + pair.first * pair.second.size();
        }
    );

    if (current_pool_size + rounded_size <= g_config.max_pool_size) {
        g_memory_pool->free_blocks[rounded_size].emplace_front(ptr, rounded_size);
    } else {
        // Pool would be too large, just free the memory
        rpc_client::free(ptr);
    }

    // Remove from tensor map
    g_remote_tensors.erase(it);
}

void clear_cache() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return;
    }

    // Free all blocks in the memory pool
    for (auto& pair : g_memory_pool->free_blocks) {
        for (auto& block : pair.second) {
            rpc_client::free(block.ptr);
        }
    }

    g_memory_pool->free_blocks.clear();
}

void clear_memory_pool() {
    clear_cache();
}

// Tensor movement
at::Tensor to_remote(const at::Tensor& tensor, int device_index, rpc_client::Error* error) {
    // If already on remote device, return as is
    if (tensor.device().type() == c10::DeviceType::PrivateUse1) {
        if (tensor.device().index() == device_index) {
            if (error) *error = rpc_client::Error::ok();
            return tensor;
        } else {
            // TODO: Implement device-to-device transfer
            // For now, just go through CPU
        }
    }

    // First copy to CPU if not already there
    at::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.to(at::kCPU);

    // Allocate memory on remote device
    rpc_client::Error alloc_error;
    void* remote_ptr = allocate(tensor.nbytes(), &alloc_error);

    if (alloc_error) {
        if (error) *error = alloc_error;
        return at::Tensor();
    }

    // Copy data to remote device
    rpc_client::Error upload_error = rpc_client::upload_tensor_data(
        remote_ptr, cpu_tensor.data_ptr(), cpu_tensor.nbytes());

    if (upload_error) {
        free(remote_ptr);
        if (error) *error = upload_error;
        return at::Tensor();
    }

    // Update statistics
    std::lock_guard<std::mutex> lock(g_mutex);
    g_memory_pool->transfer_bytes_to_remote += tensor.nbytes();

    // Create tensor that points to remote memory
    auto options = at::TensorOptions()
        .dtype(tensor.scalar_type())
        .device(c10::Device(c10::DeviceType::PrivateUse1, device_index));

    at::Tensor remote_tensor = at::from_blob(
        remote_ptr,
        tensor.sizes().vec(),
        tensor.strides().vec(),
        [remote_ptr](void*) {
            free(remote_ptr);
        },
        options
    );

    // Copy other tensor attributes
    if (tensor.requires_grad()) {
        remote_tensor.set_requires_grad(true);
    }

    // Register in our tracking map
    register_tensor(remote_ptr, remote_tensor);

    if (error) *error = rpc_client::Error::ok();
    return remote_tensor;
}

at::Tensor to_cpu(const at::Tensor& tensor, rpc_client::Error* error) {
    // If already on CPU, return as is
    if (tensor.device().is_cpu()) {
        if (error) *error = rpc_client::Error::ok();
        return tensor;
    }

    // If not a remote tensor, use standard PyTorch
    if (tensor.device().type() != c10::DeviceType::PrivateUse1) {
        at::Tensor cpu_tensor = tensor.to(at::kCPU);
        if (error) *error = rpc_client::Error::ok();
        return cpu_tensor;
    }

    // Allocate CPU tensor
    at::Tensor cpu_tensor = at::empty(
        tensor.sizes().vec(),
        at::TensorOptions().dtype(tensor.scalar_type()).device(at::kCPU)
    );

    // Use pinned memory for potentially faster transfers
    at::Tensor transfer_tensor = cpu_tensor;
    if (g_config.use_pinned_memory) {
        transfer_tensor = at::empty_pinned(
            tensor.sizes().vec(),
            at::TensorOptions().dtype(tensor.scalar_type())
        );
    }

    // Copy data from remote to CPU
    rpc_client::Error download_error = rpc_client::download_tensor_data(
        tensor.data_ptr(), transfer_tensor.data_ptr(), tensor.nbytes());

    if (download_error) {
        if (error) *error = download_error;
        return at::Tensor();
    }

    // Copy from pinned memory to regular CPU memory if needed
    if (g_config.use_pinned_memory) {
        cpu_tensor.copy_(transfer_tensor);
    }

    // Update statistics
    std::lock_guard<std::mutex> lock(g_mutex);
    g_memory_pool->transfer_bytes_from_remote += tensor.nbytes();

    // Copy other tensor attributes
    if (tensor.requires_grad()) {
        cpu_tensor.set_requires_grad(true);
    }

    if (error) *error = rpc_client::Error::ok();
    return cpu_tensor;
}

// Statistics and diagnostics
MemoryStats get_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return {0, 0, 0, 0, 0, 0, 0};
    }

    size_t cache_size = std::accumulate(
        g_memory_pool->free_blocks.begin(),
        g_memory_pool->free_blocks.end(),
        0ULL,
        [](size_t sum, const auto& pair) {
            return sum + pair.first * pair.second.size();
        }
    );

    return {
        g_memory_pool->total_allocated,
        g_memory_pool->peak_allocated,
        cache_size,
        g_memory_pool->total_allocated - cache_size,
        g_memory_pool->transfer_bytes_to_remote,
        g_memory_pool->transfer_bytes_from_remote,
        static_cast<int>(g_remote_tensors.size())
    };
}

void reset_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return;
    }

    g_memory_pool->transfer_bytes_to_remote = 0;
    g_memory_pool->transfer_bytes_from_remote = 0;
    g_memory_pool->cache_hits = 0;
    g_memory_pool->cache_misses = 0;

    // Don't reset total_allocated since that's the current state, not just a stat
}

void print_stats() {
    MemoryStats stats = get_stats();

    std::cout << "\n===== Memory Manager Statistics =====\n";
    std::cout << "Total allocated: " << stats.total_allocated / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Peak allocated: " << stats.peak_allocated / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Cache size: " << stats.cache_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Pool size: " << stats.pool_size / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Transfer to remote: " << stats.transfer_bytes_to_remote / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Transfer from remote: " << stats.transfer_bytes_from_remote / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Active tensors: " << stats.active_tensors << "\n";
    std::cout << "=====================================\n";
}

} // namespace memory_manager
