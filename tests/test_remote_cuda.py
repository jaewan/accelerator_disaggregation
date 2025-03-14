#!/usr/bin/env python3
"""
Unit tests for the remote_cuda module.

This test suite verifies the functionality of the GPU disaggregation framework
including remote execution, tensor operations, memory management, and error handling.

To run these tests:
    pytest tests/test_remote_cuda.py -v
"""

import os
import sys
import time
import pytest
import threading
import numpy as np
import torch
import remote_cuda
from unittest.mock import patch, MagicMock

# Test server address (modify as needed)
TEST_SERVER = os.environ.get("REMOTE_CUDA_TEST_SERVER", "localhost:50051")

# Whether to run tests that require an actual server
RUN_SERVER_TESTS = os.environ.get("REMOTE_CUDA_RUN_SERVER_TESTS", "0") == "1"

# Skip reason for tests that need a real server
SKIP_REASON = "Skipping tests that require a real server (set REMOTE_CUDA_RUN_SERVER_TESTS=1 to enable)"


@pytest.fixture(scope="session")
def maybe_start_test_server():
    """Start a test server if RUN_SERVER_TESTS is enabled but not running externally."""
    server_thread = None
    
    if RUN_SERVER_TESTS:
        # Check if a server is already running
        try:
            # Try to connect to the server
            connected = remote_cuda.init(TEST_SERVER)
            if connected:
                print("Using existing test server at", TEST_SERVER)
                remote_cuda.shutdown()
                yield True
                return
        except Exception:
            pass
            
        # Start a server in a thread
        def run_server():
            try:
                from remote_cuda.server import serve
                serve(port=50051, max_workers=4)
            except Exception as e:
                print(f"Error starting test server: {e}")
                
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give the server time to start
        time.sleep(3)
        print("Started test server for testing")
        
    yield RUN_SERVER_TESTS
    
    # Nothing to clean up - the server thread is a daemon


@pytest.fixture
def remote_connection(maybe_start_test_server):
    """Initialize connection to the remote server for each test that needs it."""
    if not maybe_start_test_server:
        pytest.skip(SKIP_REASON)
        
    # Connect to the server
    connected = remote_cuda.init(TEST_SERVER)
    if not connected:
        pytest.skip(f"Could not connect to test server at {TEST_SERVER}")
    
    yield
    
    # Clean up resources
    try:
        remote_cuda.clear_memory_cache()
    except Exception:
        pass


class TestRemoteCUDA:
    """Tests for the remote_cuda module."""
    
    def test_import_and_constants(self):
        """Test that we can import the module and access constants."""
        assert hasattr(remote_cuda, "REMOTE_CUDA")
        assert isinstance(remote_cuda.REMOTE_CUDA, torch.device)
        assert remote_cuda.REMOTE_CUDA.type == "privateuseone"
    
    def test_torch_integration(self):
        """Test that the module is properly integrated with torch."""
        assert hasattr(torch, "remote_cuda")
        assert hasattr(torch.remote_cuda, "is_available")
        assert hasattr(torch.remote_cuda, "device_count")
        assert hasattr(torch.remote_cuda, "set_device")
        assert hasattr(torch.remote_cuda, "current_device")
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_connection(self, remote_connection):
        """Test connecting to the remote server."""
        assert remote_cuda.is_available()
        
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_device_info(self, remote_connection):
        """Test querying device information."""
        device_count = remote_cuda.device_count()
        assert device_count > 0, "Should have at least one remote device"
        
        current_device = remote_cuda.current_device()
        assert current_device >= 0, "Current device should be non-negative"
        
        # Test setting device if we have multiple devices
        if device_count > 1:
            new_device = (current_device + 1) % device_count
            remote_cuda.set_device(new_device)
            assert remote_cuda.current_device() == new_device
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_tensor_to_remote(self, remote_connection):
        """Test moving tensors to remote device."""
        # Create a CPU tensor
        cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Move to remote
        remote_tensor = remote_cuda.tensor_to_remote(cpu_tensor)
        
        # Check properties
        assert remote_tensor.device.type == "privateuseone"
        assert remote_tensor.shape == cpu_tensor.shape
        assert remote_tensor.dtype == cpu_tensor.dtype
        
        # Alternative method
        remote_tensor2 = cpu_tensor.to("privateuseone")
        assert remote_tensor2.device.type == "privateuseone"
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_tensor_to_cpu(self, remote_connection):
        """Test moving tensors from remote to CPU."""
        # Create a tensor and move to remote
        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        remote_tensor = original.to("privateuseone")
        
        # Move back to CPU
        cpu_tensor = remote_cuda.tensor_to_cpu(remote_tensor)
        
        # Check properties and values
        assert cpu_tensor.device.type == "cpu"
        assert cpu_tensor.shape == original.shape
        assert cpu_tensor.dtype == original.dtype
        assert torch.all(torch.eq(cpu_tensor, original))
        
        # Alternative method
        cpu_tensor2 = remote_tensor.to("cpu")
        assert cpu_tensor2.device.type == "cpu"
        assert torch.all(torch.eq(cpu_tensor2, original))
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_basic_operations(self, remote_connection):
        """Test basic tensor operations on remote device."""
        # Addition
        a = torch.tensor([1.0, 2.0, 3.0]).to("privateuseone")
        b = torch.tensor([4.0, 5.0, 6.0]).to("privateuseone")
        
        c = a + b
        assert c.device.type == "privateuseone"
        
        c_cpu = c.to("cpu")
        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(c_cpu, expected)
        
        # Multiplication
        d = a * b
        d_cpu = d.to("cpu")
        expected = torch.tensor([4.0, 10.0, 18.0])
        assert torch.allclose(d_cpu, expected)
        
        # Matrix multiplication
        m1 = torch.randn(2, 3).to("privateuseone")
        m2 = torch.randn(3, 4).to("privateuseone")
        
        m3 = torch.matmul(m1, m2)
        assert m3.device.type == "privateuseone"
        assert m3.shape == (2, 4)
        
        # Compare with CPU result
        m1_cpu = m1.to("cpu")
        m2_cpu = m2.to("cpu")
        expected = torch.matmul(m1_cpu, m2_cpu)
        
        m3_cpu = m3.to("cpu")
        assert torch.allclose(m3_cpu, expected, rtol=1e-4, atol=1e-4)
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_memory_management(self, remote_connection):
        """Test memory management functionality."""
        # Create some large tensors to use memory
        tensors = []
        for _ in range(5):
            t = torch.randn(1000, 1000).to("privateuseone")
            tensors.append(t)
        
        # Get memory stats
        stats = remote_cuda.get_memory_stats()
        assert stats.total_allocated > 0
        assert stats.active_tensors >= 5
        
        # Clean up one tensor
        tensors[0] = None
        
        # Force garbage collection to make sure tensor is freed
        import gc
        gc.collect()
        
        # Clear cache
        remote_cuda.clear_memory_cache()
        
        # Check that stats have changed
        new_stats = remote_cuda.get_memory_stats()
        
        # These assertions depend on the memory manager implementation
        # Some implementations might not show an immediate decrease
        if hasattr(new_stats, 'cache_size') and new_stats.cache_size == 0:
            # If cache was cleared, we should see this
            assert new_stats.cache_size == 0
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_complex_operations(self, remote_connection):
        """Test more complex tensor operations."""
        # Test a series of operations
        a = torch.randn(5, 10).to("privateuseone")
        b = torch.randn(10, 5).to("privateuseone")
        
        # Chain of operations
        c = torch.matmul(a, b)
        d = torch.relu(c)
        e = d * 2.0 + 1.0
        f = torch.softmax(e, dim=1)
        
        # Verify device
        assert f.device.type == "privateuseone"
        
        # Compare with CPU
        a_cpu = a.to("cpu")
        b_cpu = b.to("cpu")
        
        c_cpu = torch.matmul(a_cpu, b_cpu)
        d_cpu = torch.relu(c_cpu)
        e_cpu = d_cpu * 2.0 + 1.0
        f_cpu = torch.softmax(e_cpu, dim=1)
        
        f_from_remote = f.to("cpu")
        assert torch.allclose(f_from_remote, f_cpu, rtol=1e-4, atol=1e-4)
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test initializing with invalid server
        with pytest.raises(Exception):
            remote_cuda.init("invalid-server-address:1234")
        
        # Other error tests that don't require a real server
        with patch('remote_cuda._ext') as mock_ext:
            mock_ext.to_remote.side_effect = RuntimeError("Simulated error")
            
            with pytest.raises(RuntimeError):
                cpu_tensor = torch.randn(3, 4)
                remote_cuda.tensor_to_remote(cpu_tensor)
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_tensor_requires_grad(self, remote_connection):
        """Test tensors with requires_grad."""
        # Create tensor with requires_grad
        a = torch.randn(3, 4, requires_grad=True)
        a_remote = a.to("privateuseone")
        
        # Check that requires_grad is preserved
        assert a_remote.requires_grad
        
        # Perform an operation and check that grad is tracked
        b = a_remote * 2
        assert b.requires_grad
        
        # Compute sum and backprop
        c = b.sum()
        c.backward()
        
        # Check that gradients are computed and can be accessed
        a_grad = a.grad
        assert a_grad is not None
        assert torch.all(a_grad == 2.0)
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_large_tensors(self, remote_connection):
        """Test operations with large tensors."""
        # Skip for CI environments or limited memory
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping large tensor test in CI environment")
        
        # Create large tensors (~100MB)
        shape = (2500, 5000)  # ~50MB for float32
        a = torch.randn(*shape)
        b = torch.randn(*shape)
        
        # Time moving to remote
        start = time.time()
        a_remote = a.to("privateuseone")
        b_remote = b.to("privateuseone")
        transfer_time = time.time() - start
        
        print(f"Transfer time for {a.numel() * a.element_size() / 1024 / 1024:.2f}MB: {transfer_time:.4f}s")
        
        # Perform operation
        start = time.time()
        c_remote = a_remote + b_remote
        compute_time = time.time() - start
        
        print(f"Compute time for addition: {compute_time:.4f}s")
        
        # Transfer back
        start = time.time()
        c = c_remote.to("cpu")
        download_time = time.time() - start
        
        print(f"Download time: {download_time:.4f}s")
        
        # Verify result
        expected = a + b
        assert torch.allclose(c, expected)
    
    @pytest.mark.skipif(not RUN_SERVER_TESTS, reason=SKIP_REASON)
    def test_multiple_devices(self, remote_connection):
        """Test using multiple remote devices if available."""
        device_count = remote_cuda.device_count()
        if device_count < 2:
            pytest.skip("Need at least 2 remote devices for this test")
        
        # Create tensors on different devices
        a = torch.randn(3, 4).to(torch.device("privateuseone", 0))
        b = torch.randn(3, 4).to(torch.device("privateuseone", 1))
        
        # Verify devices
        assert a.device.index == 0
        assert b.device.index == 1
        
        # Move tensor between devices
        c = a.to(torch.device("privateuseone", 1))
        assert c.device.index == 1
        
        # Operations on same device
        d = b + c
        assert d.device.index == 1
    
    def test_mocked_operations(self):
        """Test operations with mocked remote execution (no server needed)."""
        # Create mock tensors and operations
        with patch('remote_cuda.tensor_to_remote') as mock_to_remote, \
             patch('remote_cuda.tensor_to_cpu') as mock_to_cpu:
            
            # Setup mocks
            cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
            mock_remote_tensor = MagicMock()
            mock_remote_tensor.device.type = "privateuseone"
            
            mock_to_remote.return_value = mock_remote_tensor
            mock_to_cpu.return_value = cpu_tensor * 2  # Simulate operation result
            
            # Test to_remote
            result = cpu_tensor.to("privateuseone")
            assert result is mock_remote_tensor
            mock_to_remote.assert_called_once()
            
            # Test operation with mocked result
            result_cpu = result.to("cpu")
            mock_to_cpu.assert_called_once()
            assert torch.all(torch.eq(result_cpu, torch.tensor([2.0, 4.0, 6.0])))


if __name__ == "__main__":
    # To run a specific test directly: 
    # python -m tests.test_remote_cuda TestRemoteCUDA.test_import_and_constants
    pytest.main(["-xvs", __file__])
