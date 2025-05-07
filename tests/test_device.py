import torch
import remote_cuda
import unittest

class TestRemoteCUDA(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.device_remote = remote_cuda.REMOTE_CUDA
        cls.device_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_add(self):
        a_cuda = torch.tensor([1.0, 2.0, 3.0], device=self.device_cuda)
        b_cuda = torch.tensor([4.0, 5.0, 6.0], device=self.device_cuda)
        
        a_remote = torch.tensor([1.0, 2.0, 3.0], device=self.device_remote)
        b_remote = torch.tensor([4.0, 5.0, 6.0], device=self.device_remote)
        
        c_cuda = a_cuda + b_cuda
        c_remote = a_remote + b_remote
        self.assertTrue(c_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(c_remote.device.type, self.device_remote.type))
        self.assertTrue(torch.equal(c_cuda, c_remote), "Addition mismatch")
        
    def test_basic_operations(self):
        # Test tensor creation
        a_cuda = torch.tensor([1.0, 2.0, 3.0], device=self.device_cuda)
        b_cuda = torch.tensor([4.0, 5.0, 6.0], device=self.device_cuda)
        
        a_remote = torch.tensor([1.0, 2.0, 3.0], device=self.device_remote)
        b_remote = torch.tensor([4.0, 5.0, 6.0], device=self.device_remote)
        
        # Test basic arithmetic
        c_cuda = a_cuda + b_cuda
        c_remote = a_remote + b_remote # Should intercept add
        self.assertTrue(c_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(c_remote.device.type, self.device_remote.type))
        self.assertTrue(torch.equal(c_cuda, c_remote), "Addition mismatch")
        
        d_cuda = a_cuda * b_cuda
        d_remote = a_remote * b_remote # Should intercept multiply
        self.assertTrue(d_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(c_remote.device.type, self.device_remote.type))
        self.assertTrue(torch.equal(d_cuda, d_remote), "Multiplication mismatch")
        
        # Test more complex operations
        # e_cuda = torch.matmul(a_cuda, b_cuda)
        # e_remote = torch.matmul(a_remote, b_remote) # Should intercept matmul
        # self.assertTrue(e_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(e_remote.device.type, self.device_remote.type))
        # self.assertTrue(torch.equal(e_cuda, e_remote), "Matmul mismatch")
        
        # f_cuda = torch.nn.functional.relu(a_cuda)
        # f_remote = torch.nn.functional.relu(a_remote) # Should intercept relu
        # self.assertTrue(f_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(f_remote.device.type, self.device_remote.type))
        # self.assertTrue(torch.equal(f_cuda, f_remote), "ReLU mismatch")
    
    def test_tensor_create(self):
        # Instead of creating tensors directly on the device:
        # a = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        # Create tensor on CPU first and move to devices
        a_cpu = torch.tensor([1.0, 2.0, 3.0])
        a_cuda = a_cpu.to(self.device_cuda)
        a_remote = a_cpu.to(self.device_remote)
        self.assertTrue(a_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(a_remote.device.type, self.device_remote.type))

        b_cpu = torch.tensor([4.0, 5.0, 6.0])
        b_cuda = b_cpu.to(self.device_cuda)
        b_remote = b_cpu.to(self.device_remote)
        self.assertTrue(b_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(b_remote.device.type, self.device_remote.type))

        c_cuda = a_cuda + b_cuda
        c_remote = a_remote + b_remote
        self.assertTrue(c_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(c_remote.device.type, self.device_remote.type))
        self.assertTrue(torch.equal(c_cuda, c_remote), "Tensor creation or addition mismatch")
    
    def test_local_operations(self):
        # Test operations that should run locally
        a_remote = torch.tensor([1.0], device=self.device_remote)
        self.assertTrue(a_remote.device.type == self.device_remote.type, "Device type mismatch {} {}".format(a_remote.device.type, self.device_remote.type))
        self.assertEqual(a_remote.size(), torch.Size([1]), "Size mismatch")  # Should be in kLocalOps
        self.assertEqual(a_remote.item(), 1.0, "Item mismatch")  # item() should return scalar value
        self.assertTrue(torch.equal(a_remote.clone(), a_remote), "Clone mismatch")  # Cloned tensor should be identical

    def test_device_transfer(self):
        # Test device transfer handling
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
        remote_tensor = cpu_tensor.to(self.device_remote)
        self.assertTrue(remote_tensor.device.type == self.device_remote.type, "Device type mismatch {} {}".format(remote_tensor.device.type, self.device_remote.type))
        cuda_tensor = cpu_tensor.to(self.device_cuda)
        self.assertTrue(torch.equal(cuda_tensor, remote_tensor), "Device transfer mismatch")

if __name__ == "__main__":
    unittest.main()

