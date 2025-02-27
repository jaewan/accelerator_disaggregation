#ifndef REMOTE_GPU_EXTENSION_H
#define REMOTE_GPU_EXTENSION_H

#pragma once 

#include <torch/extension.h>
/*
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>
*/

namespace remote_gpu {

// Define a custom device type for disaggregation
constexpr c10::DeviceType kDisaggDeviceType = static_cast<c10::DeviceType>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES + 1);

// Function to register the device
void register_remote_gpu_device();

// Device Operations - these will be registered via TORCH_LIBRARY_INIT
TORCH_API at::Tensor forward_add(const at::Tensor& a, const at::Tensor& b);

// Device Management
//TORCH_API void initialize_remote_connection(const std::string& address);
//TORCH_API void synchronize();

} // namespace remote_gpu

#endif // REMOTE_GPU_EXTENSION_H
