#!/usr/bin/env python3
"""
Inference consistency test: BERT-Base outputs on CPU vs. remote_cuda,
with a temporary HuggingFace cache directory that is removed on exit.
"""
import atexit
import tempfile
import sys
import pytest
import torch
import remote_cuda
from transformers import BertTokenizer, BertModel

# ——— Set up a TemporaryDirectory for HF cache and register cleanup ———
_hf_tempdir = tempfile.TemporaryDirectory()
hf_cache = _hf_tempdir.name
atexit.register(_hf_tempdir.cleanup)

REMOTE_DEVICE = remote_cuda.REMOTE_CUDA
CPU_DEVICE = "cpu"

# Sample sentences for inference
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "PyTorch custom backend should produce identical outputs.",
    "Remote CUDA inference consistency check."
]

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_bert_inference():
    # Load tokenizer and models, pointing cache to our temp dir
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir=hf_cache
    )
    model_cpu = BertModel.from_pretrained(
        'bert-base-uncased',
        cache_dir=hf_cache
    ).to(CPU_DEVICE)
    model_remote = BertModel.from_pretrained(
        'bert-base-uncased',
        cache_dir=hf_cache
    ).to(REMOTE_DEVICE)

    model_cpu.eval()
    model_remote.eval()

    # Tokenize batch
    enc = tokenizer(
        SENTENCES,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=32
    )
    input_ids_cpu = enc['input_ids']
    attention_mask_cpu = enc['attention_mask']

    # Move inputs to remote device
    input_ids_remote = input_ids_cpu.to(REMOTE_DEVICE)
    attention_mask_remote = attention_mask_cpu.to(REMOTE_DEVICE)

    with torch.no_grad():
        # CPU inference
        outputs_cpu = model_cpu(
            input_ids=input_ids_cpu,
            attention_mask=attention_mask_cpu
        ).last_hidden_state

        # Remote inference
        outputs_remote = model_remote(
            input_ids=input_ids_remote,
            attention_mask=attention_mask_remote
        ).last_hidden_state

    # Bring remote outputs back to CPU
    outputs_remote_cpu = outputs_remote.cpu()

    # Compare shapes
    assert outputs_cpu.shape == outputs_remote_cpu.shape, \
        f"Shape mismatch: CPU {outputs_cpu.shape} vs Remote {outputs_remote_cpu.shape}"

    # Compare values
    assert torch.allclose(outputs_cpu, outputs_remote_cpu, rtol=1e-5, atol=1e-5), \
        "BERT inference outputs differ between CPU and remote_cuda"

if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
