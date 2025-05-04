#!/usr/bin/env python3
"""
This test verifies that a real workload (a simple feedforward network) can run on the remote_cuda
device, including backward and several training steps.  Here we explicitly pass a weight tensor
to CrossEntropyLoss so that no optional weight=None is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import remote_cuda
import sys

REMOTE_DEVICE = remote_cuda.REMOTE_CUDA
CPU_DEVICE    = "cpu"
EPOCHS        = 3
BATCH_SIZE    = 32
NUM_CLASSES   = 10

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, NUM_CLASSES)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_model_workload_training():
    # 1. Init models and sync initial weights
    model_remote = SimpleModel().to(REMOTE_DEVICE)
    model_cpu    = SimpleModel().to(CPU_DEVICE)
    model_cpu.load_state_dict(model_remote.cpu().state_dict())

    # 2. Explicit weight tensors for CrossEntropyLoss
    weight_remote = torch.ones(NUM_CLASSES, device=REMOTE_DEVICE)
    weight_cpu    = weight_remote.to(CPU_DEVICE)

    # 3. Loss & optimizers (with explicit weight)
    criterion_r = nn.CrossEntropyLoss(weight=weight_remote)
    criterion_c = nn.CrossEntropyLoss(weight=weight_cpu)
    optim_r     = torch.optim.SGD(model_remote.parameters(), lr=0.01)
    optim_c     = torch.optim.SGD(model_cpu.parameters(),    lr=0.01)

    # 4. Synthetic data loader
    def data_loader():
        for _ in range(10):  # 10 batches per epoch
            x_cpu = torch.randn(BATCH_SIZE, 1, 28, 28)
            y_cpu = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
            yield x_cpu, y_cpu

    # 5. Training loop with forward, backward, step
    model_remote.train()
    model_cpu.train()
    for epoch in range(EPOCHS):
        for x_cpu, y_cpu in data_loader():
            # Move to devices
            x_r, y_r = x_cpu.to(REMOTE_DEVICE), y_cpu.to(REMOTE_DEVICE)

            # Remote forward/backward/step
            optim_r.zero_grad()
            out_r = model_remote(x_r)
            loss_r = criterion_r(out_r, y_r)
            loss_r.backward()
            optim_r.step()

            # CPU forward/backward/step
            optim_c.zero_grad()
            out_c = model_cpu(x_cpu)
            loss_c = criterion_c(out_c, y_cpu)
            loss_c.backward()
            optim_c.step()

    # 6. Compare final parameters
    state_r = model_remote.cpu().state_dict()
    state_c = model_cpu.state_dict()
    for key in state_c:
        assert torch.allclose(
            state_c[key], state_r[key], rtol=1e-6, atol=1e-6
        ), f"Mismatch in {key}: CPU vs Remote"

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
