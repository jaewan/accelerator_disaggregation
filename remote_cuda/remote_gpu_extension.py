import torch
from remote_gpu_extension_binding import *
from torch.utils.cpp_extension import load

# Load the extension
remote_gpu_ops = load(
    name="remote_gpu_ops",
    sources=["src/remote_gpu_extension.cpp"],
    extra_cflags=["-std=c++17", "-O3"], # Add optimization flags
    extra_ldflags=["-L/path/to/your/remote_gpu_extension", "-lremote_gpu_extension"], # Add linking flags
    is_python_module=True
    verbose=True,
)

class RemoteGPUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Use the registered device
        with torch.device("remote:0"):
            return remote_gpu_ops.forward_add(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass (if needed)
        return None, None

def init_remote_device(address: str):
    remote_gpu_ops.initialize_remote_connection(address)

class RemoteGPUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return RemoteGPUFunction.apply(a, b)

# Example usage
if __name__ == "__main__":
    # Initialize remote device
    init_remote_device("remote_gpu_address")

    # Create tensors
    a = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)

    # Use the custom module
    model = RemoteGPUModule()
    result = model(a, b)

    print("Input a:", a)
    print("Input b:", b)
    print("Result:", result)
