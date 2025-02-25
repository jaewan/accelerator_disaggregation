#include "remote_gpu_extension.h"

/*
#include <torch/csrc/Device.h>
#include <torch/csrc/utils/device_guard.h>
#include <torch/csrc/utils/device.h>
*/

#include <iostream>

namespace remote_gpu {

// Placeholder for your remote server communication functions
void send_tensor_to_server(const at::Tensor& tensor) {
  std::cout << "Sending tensor to server..." << std::endl;
  // Implement your tensor serialization and network sending logic here
}

at::Tensor receive_tensor_from_server() {
  std::cout << "Receiving tensor from server..." << std::endl;
  // Implement your tensor receiving and deserialization logic here
  // For now, return a dummy tensor
  return at::zeros({2, 2}, at::kFloat);
}

at::Tensor receive_tensor_from_server() {
	at::Tensor received_tensor;
	return received_tensor;
}

// Example function to demonstrate forwarding (replace with your logic)
at::Tensor forward_add(const at::Tensor& a, const at::Tensor& b) {
  // 1. Send tensors to the server
  send_tensor_to_server(a);
  send_tensor_to_server(b);

  // 2. Send the operation to be performed (e.g., "add")
  std::cout << "Sending 'add' operation to server..." << std::endl;

  // 3. Receive the result from the server
  at::Tensor result = receive_tensor_from_server();

  return result;
}

void register_remote_gpu_device() {
    // Get the current number of GPUs.
    // This method will be used to determine the index of our custom device.
    auto num_gpus = at::cuda::getNumGPUs();

    // Define our custom device type.
    // Our device type name must satisfy the regular expression [a-z]+[a-z0-9_]*
    // i.e., it must match the pattern '[a-z]+[a-z0-9_]*'.
    const std::string device_type = "remote";

    // Set the device index to be the current number of GPUs.
    const auto device_index = num_gpus;

    // Get the current number of registered devices.
    // This will be used to determine the index of our custom device.
    const int device_count = c10::Device::MaxDeviceType;

    // Create a new device guard.
    // This will be used to register our custom device.
    auto* device_guard = new c10::SafeTypeRegistry<c10::Device::Type, c10::Device>(
        c10::Device::Type::CUDA, device_count);

    // Register our custom device.
    c10::Device::registerType(device_type, device_guard);

    // Print the registered device information.
    std::cout << "Registered device: " << c10::Device(device_type, device_index) << std::endl;
}

// Register the device when the extension is loaded
TORCH_LIBRARY_INIT(remote_gpu_ops, m) {
    register_remote_gpu_device();
    m.def("forward_add", &forward_add);
}

} // namespace remote_gpu
