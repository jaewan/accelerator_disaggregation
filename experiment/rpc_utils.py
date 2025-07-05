"""rpc_utils.py
Shared helper functions for Torch RPC interactions used by both client and
server processes in the semantic-gap experiments.

These helpers MUST reside in a standalone module (as opposed to a script
executed as ``__main__``) so that Torch RPC can locate them on every worker
when deserialising pickled Remote Procedure Calls.
"""

from __future__ import annotations

from typing import Callable, Any
from torch.distributed import rpc

__all__ = ["_call_method", "rpc_sync_with_rref"]


def _call_method(method: Callable[..., Any], rref: "rpc.RRef", *args, **kwargs):
    """Execute ``method`` on the object referenced by *rref*.

    This function runs **on the owner worker** of ``rref``. We retrieve the
    local object via :py:meth:`rref.local_value` and then call the supplied
    *unbound* method on it, forwarding ``*args`` and ``**kwargs``. Only the
    *result* is serialised back to the caller, not the object itself.
    """

    return method(rref.local_value(), *args, **kwargs)


def rpc_sync_with_rref(rref: "rpc.RRef", method: Callable[..., Any], *args, **kwargs):
    """Convenience wrapper around :pyfunc:`torch.distributed.rpc.rpc_sync` for
    calling an *instance* method on a remote object referenced by ``rref``.

    Parameters
    ----------
    rref
        Reference to the remote object.
    method
        Unbound method object defined on the target's class.
    *args, **kwargs
        Arguments forwarded to *method*.
    """

    return rpc.rpc_sync(
        rref.owner(), _call_method, args=(method, rref, *args), kwargs=kwargs
    ) 