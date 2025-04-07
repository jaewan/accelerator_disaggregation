import torch
from remote_cuda import remote_cuda, RemoteProxyTensor

def test_handle_empty_strided():
    print("[TEST] Calling remote_cuda.empty() on privateuseone")

    tensor = remote_cuda.empty(2, 2, device=torch.device("privateuseone"))
    
    # print("[INFO] Returned tensor:", tensor)
    print("[INFO] Type:", type(tensor))
    assert isinstance(tensor, RemoteProxyTensor)

    print("[PASS] handle_empty_strided() was called and returned RemoteProxyTensor")

if __name__ == "__main__":
    test_handle_empty_strided()

