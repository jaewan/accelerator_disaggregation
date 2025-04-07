import torch
from torch.utils.cpp_extension import load
import importlib.util
import os
import sys
import uuid
from .remote_tensor import RemoteProxyTensor

# Get the directory of the current file
EXTENSION_NAME = "remote_cuda_ext"
torch._C._rename_privateuse1_backend("remote_cuda")
torch.utils.generate_methods_for_privateuse1_backend()
REMOTE_CUDA = torch.device("privateuseone")

# -------------- Try to load the pre-compiled extension -------------- #

def _find_so_file():
    """Find the .so file in Bazel build directories"""
    # Get the directory containing this file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        # Look in the runfiles directory where Bazel puts the .so
        os.path.join(current_dir, f"{EXTENSION_NAME}.so"),

        # Try common Bazel output locations
        os.path.join(os.path.dirname(current_dir), f"bazel-bin/{EXTENSION_NAME}.so"),

        # For when running directly from workspace
        os.path.join(os.path.dirname(current_dir), f"{EXTENSION_NAME}.so"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

try:
    # Try to load pre-built extension first
    so_path = _find_so_file()
    if so_path:
        spec = importlib.util.spec_from_file_location("remote_cuda_ext", so_path)
        _ext = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _ext
        spec.loader.exec_module(_ext)
    else:
        raise FileNotFoundError("[ERROR] Extension Shared library not found")
        # Fallback to JIT compilation. Remove aboce raise Error if needed and edit the files below
        print("[DEBUG] No pre-built extension found, falling back to JIT compilation")
        source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _ext = load(
            name="_remote_cuda_ext",
            sources=[
                os.path.join(source_dir, "csrc/remote_device.cc"),
                os.path.join(source_dir, "csrc/remote_dispatch.cc"),
                os.path.join(source_dir, "csrc/python_bindings.cc")
            ],
            extra_include_paths=[source_dir],
            extra_cflags=["-O3"],
            verbose=True
        )
except Exception as e:
    print(f"Error loading remote_cuda extension: {e}")
    sys.exit(1)


# Initialize with a no-op function for initial build testing
def init(server_address="localhost:50051"):
    """
    Initialize connection to remote GPU server
    
    Args:
        server_address (str): Address of the remote server (default: "localhost:50051")
        **kwargs: Additional configuration options
            - connection_timeout_ms (int): Connection timeout in milliseconds
            - operation_timeout_ms (int): Operation timeout in milliseconds
            - enable_reconnect (bool): Enable automatic reconnection
            - max_reconnect_attempts (int): Maximum number of reconnection attempts
            - use_compression (bool): Enable data compression
    
    Returns:
        bool: True if connection was successful, False otherwise
    """
    return True

def is_available():
    """Check if remote CUDA is available"""
    # Placeholder implementation
    return True

def empty(*size, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Create a remote proxy tensor with uninitialized data."""
    # print(f"[DEBUG] device = {device}, device.type = {device.type if device else 'None'}")
    # if device is None or device.type != "privateuseone":
    #    raise ValueError("Device must be remote_cuda (privateuseone)")
    if device is None or device.type != torch.device("privateuseone").type:
        raise ValueError(f"Device must be remote_cuda (privateuseone), got: {device}, type: {device.type}")

    # Create a metadata-only tensor on CPU first
    cpu_tensor = torch.empty(*size, dtype=dtype, layout=layout, device="cpu", 
                            requires_grad=requires_grad)
    
    # print(f"[DEBUG] Created CPU tensor: {cpu_tensor}")

    # Generate a remote ID
    # remote_id = str(uuid.uuid4())
    
    # Register an empty tensor on the remote server
    # This would call into your C++ extension
    # _ext.register_remote_tensor(remote_id, size, dtype, layout)
    
    # Return a proxy that points to the remote tensor
    mv_tensor = cpu_tensor.to(device)
    remote_id = _ext.get_registered_remote_id(mv_tensor)
    print(remote_id)
    
    return RemoteProxyTensor(mv_tensor, remote_id)

# Make remote_cuda a module in torch
class RemoteCudaModule:
    def __init__(self):
        self.is_available = is_available
        self.__version__ = "0.1.0"
        self.device = REMOTE_CUDA
        self.name = REMOTE_CUDA
        self.empty = empty  # Add the empty function
        self.RemoteProxyTensor = RemoteProxyTensor  # Expose the proxy class

torch._register_device_module("remote_cuda", RemoteCudaModule())

# Add module to torch namespace
torch.remote_cuda = RemoteCudaModule()


# Ensure our ops are available to the PyTorch dispatcher
if hasattr(torch.ops, 'load_library'):
    try:
        torch.ops.load_library(so_path)
    except Exception as e:
        print(f"Warning: Could not load operations: {e}")

# Initialize the device with PyTorch alternatively from python_bindings.cc
#_ext.register_device()
#_ext.register_dispatch_keys()

# Import core symbols
from .remote_tensor import RemoteProxyTensor

# Expose remote_cuda for tests and modules
remote_cuda = torch.remote_cuda

