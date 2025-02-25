import torch
import sys
import os

# Add the path to the 'python' directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

# Now you can import your module
import remote_gpu_extension

class RemoteGPUModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return remote_gpu_extension.RemoteGPUFunction.apply(a, b)

def main():
    # Create tensors
    a = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)

    # Use the custom module
    model = RemoteGPUModule()
    result = model(a, b)

    print("Input a:", a)
    print("Input b:", b)
    print("Result:", result)

if __name__ == "__main__":
    main()
