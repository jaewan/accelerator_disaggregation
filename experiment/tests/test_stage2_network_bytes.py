#!/usr/bin/env python3
"""
Unit tests for Stage 2: True Network-Byte Counting

Tests verify that real RPC counters are used instead of synthetic estimates.
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

import rpc_server
from run_llm import _run_naive_remote, _run_remote_cache, _run_sys_simulated


class TestNetworkByteCounters(unittest.TestCase):
    """Test real network byte counting in RPC scenarios"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_rpc_bytes_with_mock_agent(self, mock_get_agent):
        """Test that get_rpc_bytes() correctly sums outBytes and inBytes"""
        # Mock the RPC agent and its debug info
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        # Mock debug info with specific byte counts
        mock_debug_info = {
            "CLIENT_WORKER": {
                "outBytes": 1000,
                "inBytes": 1500,
                "other_field": "ignored"
            },
            "GPU_WORKER": {
                "outBytes": 500,
                "inBytes": 750,
                "other_field": "ignored"
            },
            "non_dict_entry": "should_be_ignored"
        }
        mock_agent.get_debug_info.return_value = mock_debug_info
        
        # Call the function
        total_bytes = rpc_server.get_rpc_bytes()
        
        # Verify the calculation: (1000+1500) + (500+750) = 3750
        expected_bytes = 1000 + 1500 + 500 + 750
        self.assertEqual(total_bytes, expected_bytes)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_rpc_bytes_agent_exception(self, mock_get_agent):
        """Test that get_rpc_bytes() handles agent exceptions gracefully"""
        # Mock agent access to raise an exception
        mock_get_agent.side_effect = Exception("RPC not initialized")
        
        # Should return 0 on exception
        total_bytes = rpc_server.get_rpc_bytes()
        self.assertEqual(total_bytes, 0)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_rpc_bytes_debug_info_exception(self, mock_get_agent):
        """Test that get_rpc_bytes() handles debug info exceptions gracefully"""
        # Mock agent but make get_debug_info raise an exception
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.get_debug_info.side_effect = Exception("Debug info not available")
        
        # Should return 0 on exception
        total_bytes = rpc_server.get_rpc_bytes()
        self.assertEqual(total_bytes, 0)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_rpc_bytes_empty_debug_info(self, mock_get_agent):
        """Test that get_rpc_bytes() handles empty debug info"""
        # Mock agent with empty debug info
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        mock_agent.get_debug_info.return_value = {}
        
        # Should return 0 for empty debug info
        total_bytes = rpc_server.get_rpc_bytes()
        self.assertEqual(total_bytes, 0)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_rpc_bytes_missing_fields(self, mock_get_agent):
        """Test that get_rpc_bytes() handles missing outBytes/inBytes fields"""
        # Mock agent with partial debug info
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        mock_debug_info = {
            "CLIENT_WORKER": {
                "outBytes": 1000,
                # Missing inBytes
            },
            "GPU_WORKER": {
                # Missing outBytes
                "inBytes": 750,
            },
            "INCOMPLETE_WORKER": {
                "other_field": "no_bytes"
            }
        }
        mock_agent.get_debug_info.return_value = mock_debug_info
        
        # Should handle missing fields gracefully: 1000 + 0 + 0 + 750 = 1750
        total_bytes = rpc_server.get_rpc_bytes()
        expected_bytes = 1000 + 0 + 0 + 750  # Missing fields default to 0
        self.assertEqual(total_bytes, expected_bytes)

    @patch('torch.distributed.rpc.api._get_current_rpc_agent')
    def test_get_network_counters_compatibility(self, mock_get_agent):
        """Test that get_network_counters() still works (for backward compatibility)"""
        # Mock the RPC agent and its debug info
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent
        
        mock_debug_info = {
            "CLIENT_WORKER": {
                "outBytes": 1000,
                "inBytes": 1500,
            }
        }
        mock_agent.get_debug_info.return_value = mock_debug_info
        
        # Call the function
        sent, received = rpc_server.get_network_counters()
        
        # Verify the values
        self.assertEqual(sent, 1000)
        self.assertEqual(received, 1500)

    def test_get_rpc_bytes_returns_integer(self):
        """Test that get_rpc_bytes() always returns an integer"""
        # Test with mocked agent returning float values
        with patch('torch.distributed.rpc.api._get_current_rpc_agent') as mock_get_agent:
            mock_agent = Mock()
            mock_get_agent.return_value = mock_agent
            
            mock_debug_info = {
                "CLIENT_WORKER": {
                    "outBytes": 1000.5,  # Float value
                    "inBytes": 1500.7,   # Float value
                }
            }
            mock_agent.get_debug_info.return_value = mock_debug_info
            
            total_bytes = rpc_server.get_rpc_bytes()
            
            # Should convert to integer
            self.assertIsInstance(total_bytes, int)
            self.assertEqual(total_bytes, int(1000.5 + 1500.7))


class TestClientNetworkByteUsage(unittest.TestCase):
    """Test that client functions use real RPC counters"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    @patch('run_llm._init_rpc')
    @patch('run_llm._shutdown_rpc')
    @patch('run_llm.rpc.rpc_sync')
    @patch('run_llm.AutoTokenizer.from_pretrained')
    @patch('run_llm.AutoModelForCausalLM.from_pretrained')
    def test_naive_remote_uses_rpc_bytes(self, mock_model, mock_tokenizer, mock_rpc_sync, mock_shutdown, mock_init):
        """Test that _run_naive_remote uses get_rpc_bytes instead of manual counting"""
        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_input_ids = Mock()
        mock_tokenizer_instance.return_value = Mock(input_ids=mock_input_ids)
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.state_dict.return_value = {"param1": Mock()}
        
        # Mock RPC calls - the last call should be get_rpc_bytes
        import torch
        mock_logits = torch.tensor([[[1.0, 2.0, 3.0]]])  # Simple tensor for argmax
        mock_kv_cache = Mock()
        mock_rpc_sync.side_effect = [
            (mock_logits, mock_kv_cache),  # run_stateless_forward call
            12345  # get_rpc_bytes call
        ]
        
        # Mock args
        args = Mock()
        args.model = "test-model"
        args.prompt = "test prompt"
        args.phase = "prefill"
        
        # Capture output
        import io
        import contextlib
        captured_output = io.StringIO()
        
        with contextlib.redirect_stdout(captured_output):
            _run_naive_remote(args)
        
        output = captured_output.getvalue()
        
        # Verify that get_rpc_bytes was called
        self.assertEqual(mock_rpc_sync.call_count, 2)
        # Last call should be to get_rpc_bytes
        last_call = mock_rpc_sync.call_args_list[-1]
        self.assertEqual(last_call[0][1], rpc_server.get_rpc_bytes)
        
        # Verify output contains the RPC bytes
        self.assertIn("NETWORK_BYTES: 12345", output)


if __name__ == '__main__':
    unittest.main() 