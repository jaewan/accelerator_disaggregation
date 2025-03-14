#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <unordered_map>
#include <memory>
#include <functional>
#include "rpc_client.h"

namespace memory_manager {

// Configuration for memory management
struct MemoryConfig {
    // Memory pool settings
    bool use_memory_pool = true;
    size_t initial_pool_size = 1024 * 1024 * 1024;  // 1 GB
    size_t max_pool_size = 8ULL * 1024 * 1024 * 1024;  // 8 GB

    // Caching settings
    bool enable_tensor_cache = true;
    size_t max_cache_size = 512 * 1024 * 1024;  // 512 MB

    // Transfer optimization
    bool use_pinned_memory = true;
    bool use_async_transfer = true;
    size_t chunk_size_for_large_transfers = 16 * 1024 * 1024;  // 16 MB
};

// Initialize memory management
void init(const MemoryConfig& config = MemoryConfig());

// Tensor registration and tracking
void register_tensor(void* data_ptr, const at::Tensor& tensor);
void unregister_tensor(void* data_ptr);
bool is_remote_tensor(void* data_ptr);
at::Tensor get_tensor(void* data_ptr);

// Tensor movement with error reporting
at::Tensor to_remote(const at::Tensor& tensor, int device_index = 0, rpc_client::Error* error = nullptr);
at::Tensor to_cpu(const at::Tensor& tensor, rpc_client::Error* error = nullptr);

// Memory pool management
void* allocate(size_t size, rpc_client::Error* error = nullptr);
void free(void* ptr);
void clear_cache();
void clear_memory_pool();

// Statistics and diagnostics
struct MemoryStats {
    size_t total_allocated;
    size_t peak_allocated;
    size_t cache_size;
    size_t pool_size;
    size_t transfer_bytes_to_remote;
    size_t transfer_bytes_from_remote;
    int active_tensors;
};

MemoryStats get_stats();
void reset_stats();
void print_stats();

} // namespace memory_manager
