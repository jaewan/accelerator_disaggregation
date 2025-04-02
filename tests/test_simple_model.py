#!/usr/bin/env python3
"""
This test verifies that a real workload (a simple feedforward network) can run on the remote_cuda
device.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import remote_cuda
import sys

REMOTE_DEVICE = remote_cuda.REMOTE_CUDA

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_model_workload():
    model_remote = SimpleModel().to(REMOTE_DEVICE)
    model_cpu = SimpleModel().to("cpu")
    model_cpu.load_state_dict(model_remote.cpu().state_dict())

    input_remote = torch.randn(32, 1, 28, 28, device=REMOTE_DEVICE)
    input_cpu = input_remote.to("cpu")
    
    output_remote = model_remote(input_remote)
    output_cpu = model_cpu(input_cpu)
    
    output_from_remote = output_remote.to("cpu")
    
    assert torch.allclose(output_cpu, output_from_remote, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    sys.exit(pytest.main(["-xvs", __file__]))
