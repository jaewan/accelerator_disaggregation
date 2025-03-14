"""
Configuration utilities for remote_cuda
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ClientConfig:
    """Configuration for the RPC client"""
    server_address: str = "localhost:50051"
    connection_timeout: int = 5000  # ms
    operation_timeout: int = 30000  # ms
    enable_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_backoff_ms: int = 1000
    max_send_message_size: int = 50 * 1024 * 1024  # 50 MB
    max_receive_message_size: int = 50 * 1024 * 1024  # 50 MB
    use_compression: bool = True
    enable_keepalive: bool = True
    keepalive_time_s: int = 60
    keepalive_timeout_s: int = 20

@dataclass
class MemoryConfig:
    """Configuration for memory management"""
    use_memory_pool: bool = True
    initial_pool_size: int = 1024 * 1024 * 1024  # 1 GB
    max_pool_size: int = 8 * 1024 * 1024 * 1024  # 8 GB
    enable_tensor_cache: bool = True
    max_cache_size: int = 512 * 1024 * 1024  # 512 MB
    use_pinned_memory: bool = True
    use_async_transfer: bool = True
    chunk_size_for_large_transfers: int = 16 * 1024 * 1024  # 16 MB

_client_config: Optional[ClientConfig] = None
_memory_config: Optional[MemoryConfig] = None

def get_client_config() -> ClientConfig:
    """Get the client configuration, initializing it if needed"""
    global _client_config
    if _client_config is None:
        _client_config = ClientConfig()
    return _client_config

def get_memory_config() -> MemoryConfig:
    """Get the memory configuration, initializing it if needed"""
    global _memory_config
    if _memory_config is None:
        _memory_config = MemoryConfig()
    return _memory_config

def set_client_config(config: ClientConfig) -> None:
    """Set the client configuration"""
    global _client_config
    _client_config = config

def set_memory_config(config: MemoryConfig) -> None:
    """Set the memory configuration"""
    global _memory_config
    _memory_config = config
