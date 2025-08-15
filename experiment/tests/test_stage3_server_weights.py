#!/usr/bin/env python3
"""
Unit tests for Stage 3: Client-Side Weight Download Removal

Tests verify that server-side weight loading works correctly and client-side
model loading is removed from remote modes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from pathlib import Path
import tempfile
import shutil
import torch

# Add the experiment directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import rpc_server
from run_llm import _run_naive_remote, _run_remote_cache, _run_sys_simulated


class TestServerSideWeightLoading(unittest.TestCase):
    """Test server-side weight loading functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)

    @patch('rpc_server.AutoModelForCausalLM.from_pretrained')
    def test_download_weights_remote_first_call(self, mock_model_from_pretrained):
        """Test that download_weights_remote loads weights on first call"""
        # Create a mock worker
        worker = rpc_server.RemoteWorker("test-model")
        worker.weights_loaded = False
        
        # Mock the pretrained model
        mock_model = Mock()
        mock_model.half.return_value = mock_model
        mock_model.state_dict.return_value = {"layer.weight": torch.tensor([1.0, 2.0])}
        mock_model_from_pretrained.return_value = mock_model
        
        # Mock the worker's model
        worker.model = Mock()
        worker.model.dtype = torch.float16
        worker.model.load_state_dict = Mock()
        worker.model.parameters = Mock(return_value=[Mock(numel=Mock(return_value=100), element_size=Mock(return_value=4))])
        
        # Call download_weights_remote
        worker.download_weights_remote("test-model")
        
        # Verify that weights were loaded
        self.assertTrue(worker.weights_loaded)
        mock_model_from_pretrained.assert_called_once_with("test-model")
        mock_model.half.assert_called_once()
        worker.model.load_state_dict.assert_called_once()

    @patch('rpc_server.AutoModelForCausalLM.from_pretrained')
    def test_download_weights_remote_already_loaded(self, mock_model_from_pretrained):
        """Test that download_weights_remote skips loading if weights already loaded"""
        # Create a mock worker with weights already loaded
        worker = rpc_server.RemoteWorker("test-model")
        worker.weights_loaded = True
        
        # Call download_weights_remote
        worker.download_weights_remote("test-model")
        
        # Verify that pretrained model was not called
        mock_model_from_pretrained.assert_not_called()

    @patch('rpc_server.AutoModelForCausalLM.from_pretrained')
    def test_download_weights_remote_dtype_conversion(self, mock_model_from_pretrained):
        """Test that download_weights_remote handles different dtypes"""
        # Test float16
        worker = rpc_server.RemoteWorker("test-model")
        worker.weights_loaded = False
        
        mock_model = Mock()
        mock_model.half.return_value = mock_model
        mock_model.state_dict.return_value = {}
        mock_model_from_pretrained.return_value = mock_model
        
        worker.model = Mock()
        worker.model.dtype = torch.float16
        worker.model.load_state_dict = Mock()
        worker.model.parameters = Mock(return_value=[])
        
        worker.download_weights_remote("test-model")
        
        mock_model.half.assert_called_once()
        
        # Test bfloat16
        worker.weights_loaded = False
        mock_model.reset_mock()
        mock_model.to.return_value = mock_model
        worker.model.dtype = torch.bfloat16
        
        worker.download_weights_remote("test-model")
        
        mock_model.to.assert_called_once_with(dtype=torch.bfloat16)

    def test_download_weights_rpc_wrapper(self):
        """Test that the RPC wrapper function calls the worker method"""
        # Mock the global worker
        mock_worker = Mock()
        
        with patch('rpc_server._GLOBAL_WORKER', mock_worker):
            rpc_server.download_weights("test-model")
            mock_worker.download_weights_remote.assert_called_once_with("test-model")

    def test_download_weights_rpc_wrapper_no_worker(self):
        """Test that the RPC wrapper raises error when no worker is initialized"""
        with patch('rpc_server._GLOBAL_WORKER', None):
            with self.assertRaises(RuntimeError) as context:
                rpc_server.download_weights("test-model")
            self.assertIn("Worker not initialised", str(context.exception))


class TestClientSideWeightRemoval(unittest.TestCase):
    """Test that client-side weight loading is removed from remote modes"""

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
    def test_remote_cache_uses_download_weights(self, mock_model, mock_tokenizer, mock_rpc_sync, mock_shutdown, mock_init):
        """Test that _run_remote_cache uses download_weights instead of loading model locally"""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_input_ids = Mock()
        mock_tokenizer_instance.return_value = Mock(input_ids=mock_input_ids)
        
        # Mock RPC calls
        mock_rpc_sync.side_effect = [
            None,  # download_weights call
            (Mock(), "kv_handle"),  # run_prefill_with_handle call
            1000   # get_rpc_bytes call
        ]
        
        # Mock args
        args = Mock()
        args.model = "test-model"
        args.prompt = "test prompt"
        args.phase = "prefill"
        
        # Call the function
        with patch('builtins.print'):
            _run_remote_cache(args)
        
        # Verify that download_weights was called
        download_calls = [call for call in mock_rpc_sync.call_args_list if 'download_weights' in str(call)]
        self.assertEqual(len(download_calls), 1)
        
        # Verify that AutoModelForCausalLM.from_pretrained was NOT called
        mock_model.assert_not_called()

    @patch('run_llm._init_rpc')
    @patch('run_llm._shutdown_rpc')
    @patch('run_llm.rpc.rpc_sync')
    @patch('run_llm.AutoTokenizer.from_pretrained')
    @patch('run_llm.AutoModelForCausalLM.from_pretrained')
    def test_sys_simulated_uses_download_weights(self, mock_model, mock_tokenizer, mock_rpc_sync, mock_shutdown, mock_init):
        """Test that _run_sys_simulated uses download_weights instead of loading model locally"""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_input_ids = Mock()
        mock_tokenizer_instance.return_value = Mock(input_ids=mock_input_ids)
        
        # Mock RPC calls
        mock_rpc_sync.side_effect = [
            None,  # download_weights call
            (b"compressed_logits", "kv_id"),  # run_prefill_semantic call
            1000   # get_rpc_bytes call
        ]
        
        # Mock compression functions
        with patch('run_llm._compress_tensor') as mock_compress, \
             patch('run_llm._decompress_tensor') as mock_decompress:
            
            mock_compress.return_value = b"compressed_data"
            mock_decompress.return_value = Mock()
            
            # Mock args
            args = Mock()
            args.model = "test-model"
            args.prompt = "test prompt"
            args.phase = "prefill"
            
            # Call the function
            with patch('builtins.print'):
                _run_sys_simulated(args)
        
        # Verify that download_weights was called
        download_calls = [call for call in mock_rpc_sync.call_args_list if 'download_weights' in str(call)]
        self.assertEqual(len(download_calls), 1)
        
        # Verify that AutoModelForCausalLM.from_pretrained was NOT called
        mock_model.assert_not_called()

    @patch('run_llm._init_rpc')
    @patch('run_llm._shutdown_rpc')
    @patch('run_llm.rpc.rpc_sync')
    @patch('run_llm.AutoTokenizer.from_pretrained')
    @patch('run_llm.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoConfig.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_config')
    def test_naive_remote_with_skip_weight_upload(self, mock_model_from_config, mock_config, mock_model, mock_tokenizer, mock_rpc_sync, mock_shutdown, mock_init):
        """Test that _run_naive_remote handles skip_weight_upload flag"""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_input_ids = Mock()
        mock_tokenizer_instance.return_value = Mock(input_ids=mock_input_ids)
        
        # Mock config and model from config
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        
        mock_model_instance = Mock()
        mock_model_from_config.return_value = mock_model_instance
        
        # Mock state_dict with some tensors
        mock_tensor = Mock()
        mock_tensor.numel.return_value = 1000
        mock_tensor.element_size.return_value = 4
        mock_model_instance.state_dict.return_value = {"layer.weight": mock_tensor}
        
        # Mock RPC calls
        mock_rpc_sync.side_effect = [
            None,  # download_weights call
            (Mock(), Mock()),  # run_stateless_forward call
            1000   # get_rpc_bytes call
        ]
        
        # Mock args with skip_weight_upload=True
        args = Mock()
        args.model = "test-model"
        args.prompt = "test prompt"
        args.phase = "prefill"
        args.skip_weight_upload = True
        
        # Call the function
        with patch('builtins.print') as mock_print:
            _run_naive_remote(args)
        
        # Verify that download_weights was called
        download_calls = [call for call in mock_rpc_sync.call_args_list if 'download_weights' in str(call)]
        self.assertEqual(len(download_calls), 1)
        
        # Verify that AutoModelForCausalLM.from_pretrained was NOT called
        mock_model.assert_not_called()
        
        # Verify that config-based model creation was used
        mock_config.assert_called_once_with("test-model")
        mock_model_from_config.assert_called_once_with(mock_config_instance)

    @patch('run_llm._init_rpc')
    @patch('run_llm._shutdown_rpc')
    @patch('run_llm.rpc.rpc_sync')
    @patch('run_llm.AutoTokenizer.from_pretrained')
    @patch('run_llm.AutoModelForCausalLM.from_pretrained')
    def test_naive_remote_without_skip_weight_upload(self, mock_model, mock_tokenizer, mock_rpc_sync, mock_shutdown, mock_init):
        """Test that _run_naive_remote still loads model when skip_weight_upload=False"""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_input_ids = Mock()
        mock_tokenizer_instance.return_value = Mock(input_ids=mock_input_ids)
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_model_instance.state_dict.return_value = {"layer.weight": Mock()}
        
        # Mock RPC calls
        mock_rpc_sync.side_effect = [
            (Mock(), Mock()),  # run_stateless_forward call
            1000   # get_rpc_bytes call
        ]
        
        # Mock args with skip_weight_upload=False
        args = Mock()
        args.model = "test-model"
        args.prompt = "test prompt"
        args.phase = "prefill"
        args.skip_weight_upload = False
        
        # Call the function
        with patch('builtins.print') as mock_print:
            _run_naive_remote(args)
        
        # Verify that AutoModelForCausalLM.from_pretrained WAS called
        mock_model.assert_called_once_with("test-model")
        
        # Verify that download_weights was NOT called
        download_calls = [call for call in mock_rpc_sync.call_args_list if 'download_weights' in str(call)]
        self.assertEqual(len(download_calls), 0)


if __name__ == '__main__':
    unittest.main() 