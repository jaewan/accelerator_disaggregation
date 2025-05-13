#!/usr/bin/env python3
"""
Real-world training tests: multiple models on small CIFAR subsets using remote_cuda.
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import remote_cuda
import sys

REMOTE_DEVICE = remote_cuda.REMOTE_CUDA
CPU_DEVICE = "cpu"

# Shared transform for CIFAR datasets
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_resnet18_cifar10_training():
    # Prepare CIFAR-10 for quick training
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize ResNet-18
    model_remote = torchvision.models.resnet18(num_classes=10).to(REMOTE_DEVICE)
    model_cpu = torchvision.models.resnet18(num_classes=10).to(CPU_DEVICE)
    model_cpu.load_state_dict(model_remote.cpu().state_dict())

    # Loss and optimizer
    criterion_remote = nn.CrossEntropyLoss()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_remote = optim.SGD(model_remote.parameters(), lr=0.01, momentum=0.9)
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01, momentum=0.9)

    model_remote.train(); model_cpu.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= 5:
            break
        inputs_r = inputs.to(REMOTE_DEVICE); targets_r = targets.to(REMOTE_DEVICE)
        # Remote step
        out_r = model_remote(inputs_r)
        loss_r = criterion_remote(out_r, targets_r)
        optimizer_remote.zero_grad(); loss_r.backward(); optimizer_remote.step()
        # CPU step
        out_c = model_cpu(inputs); loss_c = criterion_cpu(out_c, targets)
        optimizer_cpu.zero_grad(); loss_c.backward(); optimizer_cpu.step()

    # Compare parameters
    state_r = model_remote.cpu().state_dict(); state_c = model_cpu.state_dict()
    for key in state_c:
        assert torch.allclose(state_c[key], state_r[key], rtol=1e-3, atol=1e-3), f"Parameter {key} mismatch"

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_vgg16_cifar100_training():
    # Prepare CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize VGG-16
    model_remote = torchvision.models.vgg16(num_classes=100).to(REMOTE_DEVICE)
    model_cpu = torchvision.models.vgg16(num_classes=100).to(CPU_DEVICE)
    model_cpu.load_state_dict(model_remote.cpu().state_dict())

    # Loss and optimizer
    criterion_remote = nn.CrossEntropyLoss()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_remote = optim.SGD(model_remote.parameters(), lr=0.01, momentum=0.9)
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01, momentum=0.9)

    model_remote.train(); model_cpu.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= 5:
            break
        inputs_r = inputs.to(REMOTE_DEVICE); targets_r = targets.to(REMOTE_DEVICE)
        # Remote step
        out_r = model_remote(inputs_r)
        loss_r = criterion_remote(out_r, targets_r)
        optimizer_remote.zero_grad(); loss_r.backward(); optimizer_remote.step()
        # CPU step
        out_c = model_cpu(inputs); loss_c = criterion_cpu(out_c, targets)
        optimizer_cpu.zero_grad(); loss_c.backward(); optimizer_cpu.step()

    # Compare parameters
    state_r = model_remote.cpu().state_dict(); state_c = model_cpu.state_dict()
    for key in state_c:
        assert torch.allclose(state_c[key], state_r[key], rtol=1e-3, atol=1e-3), f"Parameter {key} mismatch"

@pytest.mark.skipif(not remote_cuda.is_available(), reason="Remote CUDA is not available")
def test_densenet121_cifar10_training():
    # Prepare CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    # Initialize DenseNet-121
    model_remote = torchvision.models.densenet121(num_classes=10).to(REMOTE_DEVICE)
    model_cpu = torchvision.models.densenet121(num_classes=10).to(CPU_DEVICE)
    model_cpu.load_state_dict(model_remote.cpu().state_dict())

    # Loss and optimizer
    criterion_remote = nn.CrossEntropyLoss()
    criterion_cpu = nn.CrossEntropyLoss()
    optimizer_remote = optim.SGD(model_remote.parameters(), lr=0.01, momentum=0.9)
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01, momentum=0.9)

    model_remote.train(); model_cpu.train()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= 5:
            break
        inputs_r = inputs.to(REMOTE_DEVICE); targets_r = targets.to(REMOTE_DEVICE)
        # Remote step
        out_r = model_remote(inputs_r)
        loss_r = criterion_remote(out_r, targets_r)
        optimizer_remote.zero_grad(); loss_r.backward(); optimizer_remote.step()
        # CPU step
        out_c = model_cpu(inputs); loss_c = criterion_cpu(out_c, targets)
        optimizer_cpu.zero_grad(); loss_c.backward(); optimizer_cpu.step()

    # Compare parameters
    state_r = model_remote.cpu().state_dict(); state_c = model_cpu.state_dict()
    for key in state_c:
        assert torch.allclose(state_c[key], state_r[key], rtol=1e-3, atol=1e-3), f"Parameter {key} mismatch"

if __name__ == "__main__":
    sys.exit(pytest.main(["-vs", __file__]))
