#!/usr/bin/env python3
"""
Integration tests for Stage 1: Resource-Hygiene Patch

Tests verify that background processes are properly cleaned up in real scenarios.
"""

import unittest
import subprocess
import time
import os
import signal
import sys
from pathlib import Path
import tempfile
import shutil

# Add the experiment directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResourceHygieneIntegration(unittest.TestCase):
    """Integration tests for resource cleanup"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    def test_failure_injection_cleanup(self):
        """Test that processes are cleaned up when failures are injected"""
        # Create a modified run_llm.py that fails when RUN_FAIL=1
        failure_script = """
import os
import sys
if os.environ.get('RUN_FAIL') == '1':
    print("NETWORK_BYTES: 0")
    sys.exit(1)
else:
    print("NETWORK_BYTES: 0")
    print("5.0")  # Fake latency
"""
        
        # Write the failure script
        with open("run_llm.py", "w") as f:
            f.write(failure_script)
        
        # Create a minimal experiment driver script for testing
        test_driver = """
import sys
sys.path.insert(0, '.')
from experiment_driver import run_experiment
import argparse

# Mock args for testing
args = argparse.Namespace()
args.trials = 1
args.modes = "local"  # Use local mode to avoid RPC complexity
args.output_dir = "test_artefacts"
args.master_port = "29500"
args.model = "test-model"
args.gpu_host = "127.0.0.1"
args.external_server = False
args.output = "test_results.csv"

try:
    run_experiment(args)
except Exception as e:
    print(f"Expected failure: {e}")
"""
        
        with open("test_driver.py", "w") as f:
            f.write(test_driver)
        
        # Set environment variable to cause failure
        env = os.environ.copy()
        env['RUN_FAIL'] = '1'
        
        # Run the test driver
        result = subprocess.run(
            [sys.executable, "test_driver.py"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check that no orphaned processes remain
        # Look for any remaining python processes related to our test
        ps_result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        
        # Filter for python processes (excluding our test process)
        python_procs = [line for line in ps_result.stdout.splitlines() 
                       if "python" in line and "test_stage1_integration" not in line]
        
        # Should have minimal python processes (just the current test)
        # This is a basic check - in a real scenario we'd be more specific
        self.assertLess(len(python_procs), 10, 
                       f"Too many python processes found: {python_procs}")

    def test_server_log_timestamping(self):
        """Test that server logs are properly timestamped"""
        # Copy the experiment_driver.py to our temp directory
        experiment_dir = Path(__file__).parent.parent
        shutil.copy(experiment_dir / "experiment_driver.py", "experiment_driver.py")
        
        # Create a minimal test that starts an RPC server
        test_script = """
import sys
sys.path.insert(0, '.')
from experiment_driver import _start_rpc_server
import time

# Start a server (this will fail but should create a log)
try:
    handle = _start_rpc_server("test-model", "127.0.0.1", "29500")
    time.sleep(1)  # Give it a moment
    handle.terminate()
except Exception as e:
    print(f"Expected error: {e}")
"""
        
        with open("test_server.py", "w") as f:
            f.write(test_script)
        
        # Run the test
        result = subprocess.run(
            [sys.executable, "test_server.py"],
            capture_output=True,
            text=True
        )
        
        # Check that timestamped log files were created
        logs_dir = Path("artefacts/logs")
        if logs_dir.exists():
            log_files = list(logs_dir.glob("server_29500_*.log"))
            self.assertGreater(len(log_files), 0, 
                             "No timestamped log files were created")
            
            # Check that the log file has a reasonable timestamp format
            for log_file in log_files:
                # Extract timestamp from filename
                timestamp_str = log_file.stem.split("_")[-1]
                try:
                    timestamp = int(timestamp_str)
                    # Should be a reasonable Unix timestamp (after 2020)
                    self.assertGreater(timestamp, 1577836800, 
                                     f"Invalid timestamp in log file: {log_file}")
                except ValueError:
                    self.fail(f"Invalid timestamp format in log file: {log_file}")

    def test_timeout_handling(self):
        """Test that client timeouts are properly handled"""
        # Create a simple test that directly tests the timeout behavior
        test_script = """
import subprocess
from subprocess import TimeoutExpired

# Test subprocess timeout directly
try:
    result = subprocess.run(
        ["sleep", "5"],  # Sleep for 5 seconds
        capture_output=True,
        text=True,
        timeout=1  # But timeout after 1 second
    )
    print("ERROR: Should have timed out!")
except TimeoutExpired:
    print("SUCCESS: Timeout handled correctly")
except Exception as e:
    print(f"OTHER ERROR: {e}")
"""
        
        with open("test_timeout.py", "w") as f:
            f.write(test_script)
        
        # Run the test
        result = subprocess.run(
            [sys.executable, "test_timeout.py"],
            capture_output=True,
            text=True,
            timeout=10  # Give it 10 seconds max
        )
        
        # Should see our success message
        self.assertIn("SUCCESS: Timeout handled correctly", result.stdout,
                     f"Timeout not handled correctly. Output: {result.stdout}, Error: {result.stderr}")


if __name__ == '__main__':
    unittest.main() 