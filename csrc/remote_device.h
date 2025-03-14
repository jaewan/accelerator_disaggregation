#pragma once

#include <torch/extension.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Device.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>


namespace remote_cuda {
// Use PrivateUse1 for our custom device type for prototyping.
// For production, register a unique device type
constexpr c10::DeviceType REMOTE_CUDA_TYPE = c10::DeviceType::PrivateUse1;

// Device guard implementation
// Implement what is in venv/lib/python3.10/site-packages/torch/include/c10/core/impl/DeviceGuardImplInterface.h
class RemoteCUDAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
 public:
  RemoteCUDAGuardImpl() = default;
  ~RemoteCUDAGuardImpl() override = default;

  // Return our device type
  c10::DeviceType type() const override {
    return REMOTE_CUDA_TYPE;
  }

  // Exchange the current device with the specified device
  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_CHECK(d.type() == REMOTE_CUDA_TYPE, "Invalid device type");
    c10::Device current_device = getDevice();
    setDevice(d);
    return current_device;
  }

  // Get the current device
  c10::Device getDevice() const override {
    // For now, always return device index 0
    return c10::Device(REMOTE_CUDA_TYPE, 0);
  }

  // Set the current device
  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.type() == REMOTE_CUDA_TYPE, "Invalid device type");
    // Add your device-specific logic here
  }

  // Set the current device without checks
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    // Add your device-specific logic here
  }

  // Get the default stream for the device
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  // Exchange the current stream
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return s;
  }

  // Get the number of devices - must be noexcept as per the error message
  c10::DeviceIndex deviceCount() const noexcept override {
    return 1;
  }
};

// Function to register the device
void register_device();

}  // namespace remote_cuda
