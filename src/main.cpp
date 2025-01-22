#include "disagg_device.h"
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Starting AI Accelerator Disaggregation Framework..." << std::endl;

    DisaggDevice device;

    // Create a sample tensor
    auto tensor = torch::rand({3, 3}, torch::kFloat);

    std::cout << "Original Tensor:" << std::endl;
    std::cout << tensor << std::endl;

    // Offload tensor to remote device
    auto result = device.offloadToRemote(tensor);

    std::cout << "Processed Tensor:" << std::endl;
    std::cout << result << std::endl;

    return 0;
}
