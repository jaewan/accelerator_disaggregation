#pragma once

#include <torch/csrc/Device.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorImpl.h>

// Define a custom device type for disaggregation
constexpr c10::DeviceType kDisaggDeviceType = static_cast<c10::DeviceType>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES + 1);

class DisaggDevice {
public:
    DisaggDevice();

    // Example method to offload tensor operation
    torch::Tensor offloadToRemote(const torch::Tensor& input);
};
