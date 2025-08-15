#!/usr/bin/env python3
"""
Integration tests for Stage 4: Persistent Servers & Port Pool

Tests verify that servers are reused across trials and state is reset properly.
"""

import unittest
from unittest.mock import Mock, patch, call
import sys
import os
from pathlib import Path
import tempfile
import shutil
import time

# Add the experiment directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_driver import ServerPool, _reset_server_state, ProcessHandle


class TestServerPool(unittest.TestCase):
    """Test ServerPool context manager functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_server_pool_initialization(self):
        """Test that ServerPool correctly assigns ports to modes"""
        modes = ["naive", "remote_cache", "sys_simulated"]
        pool = ServerPool(modes, "test-model", "127.0.0.1", 29500)
        
        # Check port assignments
        self.assertEqual(pool.get_port("naive"), "29500")
        self.assertEqual(pool.get_port("remote_cache"), "29510")
        self.assertEqual(pool.get_port("sys_simulated"), "29520")
        
        # Local mode should get base port
        self.assertEqual(pool.get_port("local"), "29500")

    def test_server_pool_excludes_local_mode(self):
        """Test that ServerPool doesn't start servers for local mode"""
        modes = ["local", "naive"]
        pool = ServerPool(modes, "test-model", "127.0.0.1", 29500)
        
        # Should only have naive mode in ports (local is excluded)
        self.assertIn("naive", pool.ports)
        self.assertNotIn("local", pool.ports)

    @patch('experiment_driver._start_rpc_server')
    def test_server_pool_context_manager(self, mock_start_server):
        """Test that ServerPool properly starts and terminates servers"""
        # Mock server handles
        mock_server_naive = Mock(spec=ProcessHandle)
        mock_server_cache = Mock(spec=ProcessHandle)
        mock_start_server.side_effect = [mock_server_naive, mock_server_cache]
        
        modes = ["naive", "remote_cache"]
        
        with ServerPool(modes, "test-model", "127.0.0.1", 29500) as pool:
            # Verify servers were started
            self.assertEqual(mock_start_server.call_count, 2)
            
            # Check the calls
            calls = mock_start_server.call_args_list
            self.assertEqual(calls[0], call("test-model", "127.0.0.1", "29500"))
            self.assertEqual(calls[1], call("test-model", "127.0.0.1", "29510"))
        
        # Verify servers were terminated
        mock_server_naive.terminate.assert_called_once()
        mock_server_cache.terminate.assert_called_once()

    @patch('experiment_driver._start_rpc_server')
    def test_server_pool_handles_start_failure(self, mock_start_server):
        """Test that ServerPool handles server start failures gracefully"""
        # Make the first server start succeed, second fail
        mock_server_naive = Mock(spec=ProcessHandle)
        mock_start_server.side_effect = [mock_server_naive, Exception("Server start failed")]
        
        modes = ["naive", "remote_cache"]
        
        with self.assertRaises(Exception):
            with ServerPool(modes, "test-model", "127.0.0.1", 29500) as pool:
                pass
        
        # The successfully started server should still be terminated
        mock_server_naive.terminate.assert_called_once()


class TestServerStateReset(unittest.TestCase):
    """Test server state reset functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_reset_server_state_local_mode(self):
        """Test that reset_server_state skips local mode"""
        # This should complete without errors and without trying to connect to RPC
        _reset_server_state("local", "29500")
        # No assertion needed - just verify it doesn't raise an exception

    @patch('experiment_driver.rpc')
    @patch('experiment_driver.rpc_server')
    @patch('experiment_driver.os.environ', {})
    def test_reset_server_state_rpc_success(self, mock_rpc_server, mock_rpc):
        """Test successful RPC server state reset"""
        # Mock RPC operations
        mock_rpc.init_rpc = Mock()
        mock_rpc.rpc_sync = Mock()
        mock_rpc.shutdown = Mock()
        
        # Call the function
        _reset_server_state("naive", "29500")
        
        # Verify RPC was initialized
        mock_rpc.init_rpc.assert_called_once()
        
        # Verify reset_state was called
        mock_rpc.rpc_sync.assert_called_once_with("GPU_WORKER", mock_rpc_server.reset_state)
        
        # Verify RPC was shutdown
        mock_rpc.shutdown.assert_called_once_with(graceful=False)

    @patch('experiment_driver.rpc')
    @patch('experiment_driver.rpc_server')
    @patch('experiment_driver.os.environ', {})
    def test_reset_server_state_rpc_failure(self, mock_rpc_server, mock_rpc):
        """Test that RPC failures are handled gracefully"""
        # Mock RPC to raise an exception
        mock_rpc.init_rpc = Mock()
        mock_rpc.rpc_sync = Mock(side_effect=Exception("RPC call failed"))
        mock_rpc.shutdown = Mock()
        
        # Call the function - should not raise an exception
        with patch('builtins.print') as mock_print:
            _reset_server_state("naive", "29500")
        
        # Verify warning was printed
        mock_print.assert_called()
        warning_message = mock_print.call_args[0][0]
        self.assertIn("Warning: Failed to reset server state", warning_message)
        
        # Verify shutdown was still attempted
        mock_rpc.shutdown.assert_called()


class TestServerPoolIntegration(unittest.TestCase):
    """Integration tests for ServerPool with multiple modes"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    @patch('experiment_driver._start_rpc_server')
    def test_server_pool_port_uniqueness(self, mock_start_server):
        """Test that each mode gets a unique port"""
        # Mock server handles
        mock_servers = [Mock(spec=ProcessHandle) for _ in range(3)]
        mock_start_server.side_effect = mock_servers
        
        modes = ["naive", "remote_cache", "sys_simulated"]
        
        with ServerPool(modes, "test-model", "127.0.0.1", 29500) as pool:
            # Verify each mode has a unique port
            ports = [pool.get_port(mode) for mode in modes]
            self.assertEqual(len(set(ports)), len(ports))  # All ports should be unique
            
            # Verify expected port values
            self.assertEqual(pool.get_port("naive"), "29500")
            self.assertEqual(pool.get_port("remote_cache"), "29510") 
            self.assertEqual(pool.get_port("sys_simulated"), "29520")

    @patch('experiment_driver._start_rpc_server')
    def test_server_pool_reuse_across_trials(self, mock_start_server):
        """Test that servers can be reused across multiple trials"""
        # Mock server handles
        mock_server = Mock(spec=ProcessHandle)
        mock_start_server.return_value = mock_server
        
        modes = ["naive"]
        
        with ServerPool(modes, "test-model", "127.0.0.1", 29500) as pool:
            # Simulate multiple trials using the same server
            for trial in range(3):
                port = pool.get_port("naive")
                self.assertEqual(port, "29500")
        
        # Server should be started only once
        mock_start_server.assert_called_once_with("test-model", "127.0.0.1", "29500")
        
        # Server should be terminated only once (at the end)
        mock_server.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main() 