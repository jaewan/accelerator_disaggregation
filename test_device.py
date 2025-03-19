import torch
import remote_cuda
import unittest

class TestRemoteCUDA():
    def __init__(self):
        self.device = remote_cuda.REMOTE_CUDA

    def test_basic_operations(self):
        # Test tensor creation
        a = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        print("1")
        b = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        print("2")

        # Test basic arithmetic
        c = a + b  # Should intercept add
        d = a * b  # Should intercept multiply

        # Test more complex operations
        e = torch.matmul(a, b)  # Should intercept matmul
        f = torch.nn.functional.relu(a)  # Should intercept relu
    def A(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        print("1", a.device)
        b = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        print("2")

        # Test basic arithmetic
        c = a + b  # Should intercept add

    def B(self):
        a = torch.tensor([1.0, 2.0, 3.0], device=self.device)

    def test_tensor_create(self):
        # Instead of creating tensors directly on the device:
        # a = torch.tensor([1.0, 2.0, 3.0], device=self.device)

        # Create on CPU first, then move to your device:
        a = torch.tensor([1.0, 2.0, 3.0])
        a = a.to(self.device)

        b = torch.tensor([4.0, 5.0, 6.0])
        b = b.to(self.device)

        # Test operations as before
        c = a + b
        # ...

    def test_local_operations(self):
        # Test operations that should run locally
        a = torch.tensor([1.0], device=self.device)
        size = a.size()  # Should be in kLocalOps
        device = a.device  # Should be in kLocalOps

    def test_device_transfer(self):
        # Test device transfer handling
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        remote_tensor = cpu_tensor.to(self.device)

test = TestRemoteCUDA()
test.test_basic_operations()
