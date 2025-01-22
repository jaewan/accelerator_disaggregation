#ifndef REMOTE_GPU_EXTENSION_H
#define REMOTE_GPU_EXTENSION_H

#include <torch/extension.h>
#include <torch/csrc/Device.h>

namespace remote_gpu {

// Function to register the device
void register_remote_gpu_device();

// Example function to demonstrate forwarding (replace with your logic)
at::Tensor forward_add(const at::Tensor& a, const at::Tensor& b);

} // namespace remote_gpu

#endif // REMOTE_GPU_EXTENSION_H
