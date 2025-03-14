#pragma once

#include "remote_device.h"

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <c10/core/DispatchKey.h>

namespace remote_cuda {

// Create a dispatch key for our device
// Must change TORCH_LIBBRARY_IMPL in remote_dispatch.cc if this changes
// as that Macro requires hardcoded type, not variable. cannot use this variable
constexpr c10::DispatchKey REMOTE_CUDA_KEY = c10::DispatchKey::PrivateUse1;

void register_dispatch_keys();

// Handle specific operation types
at::Tensor handle_empty_strided(c10::IntArrayRef size, 
		c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, 
		c10::optional<c10::Layout> layout_opt, c10::optional<c10::Device> device_opt, 
		c10::optional<bool> pin_memory_opt);
//at::Tensor handle_binary_op(const char* op_name, const at::Tensor& self, const at::Tensor& other);
//at::Tensor handle_unary_op(const char* op_name, const at::Tensor& self);
//at::Tensor handle_view_op(const char* op_name, const at::Tensor& self, c10::ArrayRef<int64_t> sizes);

} // namespace remote_cuda
