import torch
import uuid
import weakref

# REMOTE_CUDA = torch.device("privateuseone")

# Global registry of remote tensors
_remote_tensor_registry = weakref.WeakValueDictionary()


class RemoteProxyTensor(torch.Tensor):
    """A proxy tensor that represents a tensor on a remote device."""
    
    @staticmethod
    def __new__(cls, base_tensor, remote_id=None):
        instance = torch.Tensor._make_subclass(cls, base_tensor)
        if remote_id is None:
            remote_id = str(uuid.uuid4())
        instance._remote_id = remote_id
        _remote_tensor_registry[remote_id] = instance
        return instance
    
    def __repr__(self):
        # Only materialize when actually displaying the tensor
        materialized = self.materialize()
        return f"RemoteProxyTensor(id={self._remote_id}, value={materialized})"
    
    def materialize(self):
        """Fetch the actual tensor data from the remote server."""
        # Call your C++ extension to get the actual tensor data
        # This would communicate with the remote server
        print(f"Materializing tensor with ID: {self._remote_id}")
        
        # For now, this is just a placeholder
        # In reality, you'd call into your C++ code to fetch the data
        result = torch.zeros_like(self, device="cpu")
        return result
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
            
        # Check if this operation requires materialization
        if func.__name__ in ['copy_', 'print', 'to_string']:
            # Materialize all remote tensors in args
            new_args = []
            for arg in args:
                if isinstance(arg, RemoteProxyTensor):
                    new_args.append(arg.materialize())
                else:
                    new_args.append(arg)
            result = func(*new_args, **kwargs)
            return result
        
        # Otherwise, execute the operation remotely and return a new proxy
        print(f"Executing {func.__name__} remotely")
        
        # For non-materializing operations, just create a new proxy tensor
        # that represents the result of the operation
        result_tensor = func(*args, **kwargs)
        return RemoteProxyTensor(result_tensor, remote_id=str(uuid.uuid4()))
    
    
# Modify your existing code to use the proxy tensor
def empty(*size, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    """Create a remote proxy tensor with uninitialized data."""
    if device is None or device.type != "privateuseone":
        raise ValueError("Device must be remote_cuda (privateuseone)")
    
    # Create a metadata-only tensor on CPU first
    cpu_tensor = torch.empty(*size, dtype=dtype, layout=layout, device="cpu", 
                            requires_grad=requires_grad)
    
    # Generate a remote ID
    remote_id = str(uuid.uuid4())
    
    # Register an empty tensor on the remote server
    # This would call into your C++ extension
    # _ext.register_remote_tensor(remote_id, size, dtype, layout)
    
    # Return a proxy that points to the remote tensor
    return RemoteProxyTensor(cpu_tensor.to(device), remote_id)

