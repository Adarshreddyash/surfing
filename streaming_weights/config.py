# streaming_weights/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamingConfig:
    """Configuration for streaming weights"""

    # Server settings
    server_host: str = "localhost"
    server_port: int = 8765
    use_ssl: bool = False

    # Caching settings
    cache_size: int = 3
    enable_compression: bool = True

    # Performance settings
    prefetch_layers: bool = True
    prefetch_count: int = 1
    timeout_seconds: int = 30

    # Model settings
    model_name: str = ""
    chunks_dir: Optional[str] = None

    @property
    def websocket_uri(self) -> str:
        protocol = "wss" if self.use_ssl else "ws"
        return f"{protocol}://{self.server_host}:{self.server_port}"
