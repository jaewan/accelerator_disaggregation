#!/usr/bin/env python3
"""
Multi-node environment validation script

Run this script on both CLIENT_HOST and GPU_HOST to ensure
environment consistency for PyTorch RPC experiments.

Usage:
    python scripts/validate_multinode.py [--gpu-host <IP>] [--port <PORT>]
"""

import argparse
import sys
import subprocess
import socket
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("   ❌ ERROR: Python 3.8+ required")
        return False
    print("   ✅ Python version OK")
    return True

def check_dependencies():
    """Check critical dependencies"""
    print("📦 Checking dependencies...")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA devices: {torch.cuda.device_count()}")
        
        import transformers
        print(f"   Transformers: {transformers.__version__}")
        
        # Check if we can import RPC components
        from torch.distributed import rpc
        print("   ✅ PyTorch RPC available")
        
        return True
    except ImportError as e:
        print(f"   ❌ ERROR: Missing dependency: {e}")
        return False

def check_network_connectivity(host, port):
    """Test network connectivity to target host"""
    if not host:
        print("🌐 Network connectivity: Skipped (no host specified)")
        return True
        
    print(f"🌐 Testing connectivity to {host}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print("   ✅ Network connectivity OK")
            return True
        else:
            print("   ❌ Cannot connect to target host")
            print(f"   Ensure {host}:{port} is reachable and firewall allows connections")
            return False
            
    except Exception as e:
        print(f"   ❌ Network error: {e}")
        return False

def check_gpu_monitoring():
    """Check if nvidia-smi is available"""
    print("🖥️  Checking GPU monitoring...")
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print(f"   GPUs found: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"     [{i}] {gpu}")
            print("   ✅ GPU monitoring OK")
            return True
        else:
            print("   ❌ nvidia-smi failed")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ nvidia-smi not found or timeout")
        return False

def check_file_structure():
    """Check if all required files are present"""
    print("📁 Checking file structure...")
    
    required_files = [
        "experiment_driver.py",
        "rpc_server.py", 
        "run_llm.py",
        "requirements.txt"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"   ❌ Missing files: {missing}")
        return False
    
    print("   ✅ All required files present")
    return True

def main():
    parser = argparse.ArgumentParser(description="Validate multi-node setup")
    parser.add_argument("--gpu-host", help="GPU host IP to test connectivity")
    parser.add_argument("--port", type=int, default=29501, help="Port to test")
    args = parser.parse_args()
    
    print("🔍 Multi-Node Environment Validation")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_file_structure(),
        check_gpu_monitoring(),
        check_network_connectivity(args.gpu_host, args.port)
    ]
    
    print("\n📊 Summary:")
    print("=" * 50)
    
    if all(checks):
        print("✅ ALL CHECKS PASSED - Environment ready for multi-node experiments!")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main() 