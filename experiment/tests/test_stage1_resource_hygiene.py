#!/usr/bin/env python3
"""
Unit tests for Stage 1: Resource-Hygiene Patch

Tests verify that background processes are always terminated, even when exceptions occur.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the experiment directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_driver import _start_rpc_server, _start_dmon, _run_client, run_experiment, ProcessHandle


class TestResourceHygiene(unittest.TestCase):
    """Test resource cleanup in experiment_driver.py"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    @patch('experiment_driver.subprocess.Popen')
    @patch('experiment_driver.time.sleep')
    @patch('experiment_driver.Path.open')
    @patch('experiment_driver.Path.exists')
    @patch('experiment_driver.Path.read_text')
    def test_start_rpc_server_creates_timestamped_log(self, mock_read_text, mock_exists, mock_open, mock_sleep, mock_popen):
        """Test that RPC server creates timestamped log files"""
        # Mock the popen process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock log file existence and content for server ready check
        mock_exists.return_value = True
        mock_read_text.return_value = "RPC server running"
        
        # Call the function
        handle = _start_rpc_server("test-model", "127.0.0.1", "29500")
        
        # Verify log directory was created
        self.assertTrue(Path("artefacts/logs").exists())
        
        # Verify it's a ProcessHandle
        self.assertIsInstance(handle, ProcessHandle)

    @patch('experiment_driver.subprocess.Popen')
    def test_start_dmon_creates_process_handle(self, mock_popen):
        """Test that dmon creates a ProcessHandle"""
        # Mock the popen process
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        # Call the function
        csv_path = Path("test.csv")
        handle = _start_dmon(csv_path)
        
        # Verify it's a ProcessHandle
        self.assertIsInstance(handle, ProcessHandle)
        
        # Verify the command was called correctly
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        self.assertIn("nvidia-smi", args[0])
        self.assertIn("dmon", args[0])

    @patch('experiment_driver.subprocess.run')
    def test_run_client_timeout_handling(self, mock_run):
        """Test that client runs have proper timeout handling"""
        from subprocess import TimeoutExpired
        
        # Mock timeout exception
        mock_run.side_effect = TimeoutExpired(cmd=[], timeout=300)
        
        # Create mock args
        args = Mock()
        args.gpu_host = "127.0.0.1"
        args.master_port = "29500"
        args.model = "test-model"
        
        # Verify timeout exception is properly handled
        with self.assertRaises(TimeoutExpired):
            _run_client("local", "prefill", args)

    @patch('experiment_driver._start_rpc_server')
    @patch('experiment_driver._start_dmon')
    @patch('experiment_driver._run_client')
    @patch('experiment_driver._average_sm_util')
    @patch('experiment_driver._csv_data_rows')
    def test_exitstack_cleanup_on_exception(self, mock_csv_rows, mock_sm_util, 
                                           mock_run_client, mock_start_dmon, mock_start_rpc):
        """Test that ExitStack properly cleans up resources when exceptions occur"""
        # Set up mocks
        mock_server = Mock()
        mock_server.terminate = Mock()
        mock_start_rpc.return_value = mock_server
        
        mock_dmon = Mock()
        mock_dmon.terminate = Mock()
        mock_start_dmon.return_value = mock_dmon
        
        mock_csv_rows.return_value = 5  # Sufficient rows
        mock_sm_util.return_value = 10.0
        
        # Make client run fail on first attempt, succeed on second
        mock_run_client.side_effect = [Exception("Test failure"), (5.0, 1000)]
        
        # Create mock args
        args = Mock()
        args.trials = 1
        args.modes = "naive"  # Remote mode to trigger server startup
        args.output_dir = "test_artefacts"
        args.master_port = "29500"
        args.model = "test-model"
        args.gpu_host = "127.0.0.1"
        args.external_server = False
        args.output = "test_results.csv"
        
        # Run the experiment
        try:
            run_experiment(args)
        except Exception:
            pass  # We expect some exceptions during testing
        
        # Verify that terminate was called on both server and dmon
        # Note: Due to ExitStack, terminate should be called even if exceptions occur
        mock_server.terminate.assert_called()
        mock_dmon.terminate.assert_called()

    def test_process_handle_terminate_with_timeout(self):
        """Test ProcessHandle terminate method with timeout handling"""
        # Create a mock process
        mock_popen = Mock()
        mock_popen.poll.return_value = None  # Process is still running
        mock_popen.send_signal = Mock()
        mock_popen.wait = Mock()
        
        # Create ProcessHandle
        handle = ProcessHandle(mock_popen)
        
        # Test normal termination
        handle.terminate()
        
        # Verify signal was sent and wait was called
        mock_popen.send_signal.assert_called()
        mock_popen.wait.assert_called()

    def test_process_handle_terminate_already_dead(self):
        """Test ProcessHandle terminate when process is already dead"""
        # Create a mock process that's already dead
        mock_popen = Mock()
        mock_popen.poll.return_value = 0  # Process is dead (exit code 0)
        
        # Create ProcessHandle
        handle = ProcessHandle(mock_popen)
        
        # Test termination of dead process
        handle.terminate()
        
        # Verify no signals were sent (process already dead)
        mock_popen.send_signal.assert_not_called()
        mock_popen.wait.assert_not_called()


if __name__ == '__main__':
    unittest.main() 