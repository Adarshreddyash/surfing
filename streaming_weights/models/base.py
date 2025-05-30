# streaming_weights/models/base.py
import torch
import torch.nn as nn
import asyncio
import websockets
import json
import io
import time
from typing import Optional, Dict, Any, List, Tuple
import logging


class StreamingBaseModel(nn.Module):
    """Base class for all streaming transformer models.
    
    This class provides the common functionality for streaming transformer models,
    including weight fetching, caching, and async execution. Specific model types
    should inherit from this class and implement their specific forward logic.
    """
    
    def __init__(
        self,
        model_name: str,
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__()

        self.model_name = model_name
        self.websocket_uri = websocket_uri
        
        # Layer cache (LRU-style)
        self.component_cache: Dict[str, nn.Module] = {}
        self.cache_size = cache_size
        self.access_order = []  # For LRU eviction

        # Stats tracking
        self._total_component_accesses = 0
        self._cache_hits = 0
        self._total_inferences = 0
        self._avg_inference_time = 0.0

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def _fetch_weights(
        self, component_type: str, component_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch weights from the server"""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                    self.websocket_uri,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10_485_760,  # 10MB
                    compression=None  # Disable compression for large messages
                ) as ws:
                    request = f"GET {component_type} {component_id}"
                    await ws.send(request)
                    response = await ws.recv()

                    data = json.loads(response)
                    if not data.get("success"):
                        self.logger.error(f"Server error: {data.get('error')}")
                        return None

                    # Decode hex-encoded weights
                    weights_hex = data.get("weights")
                    if not weights_hex:
                        self.logger.error("No weights data in response")
                        return None

                    weights_bytes = bytes.fromhex(weights_hex)
                    buffer = io.BytesIO(weights_bytes)
                    state_dict = torch.load(buffer, map_location="cpu")

                    self.logger.info(
                        f"Successfully fetched {component_type}_{component_id}"
                    )
                    return state_dict

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                asyncio.TimeoutError,
            ) as e:
                self.logger.warning(
                    f"Connection error (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff
                else:
                    self.logger.error(
                        f"Failed to fetch weights after {max_retries} attempts"
                    )
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error fetching weights: {e}")
                return None
    
    async def _load_component(self, component_type: str, component_id: str, component_class: nn.Module, config: Any) -> Optional[nn.Module]:
        """Load a specific model component with error handling and monitoring"""
        # Create a cache key from component type and ID
        cache_key = f"{component_type}_{component_id}"
        
        # Track component access for stats
        self._total_component_accesses += 1
        
        # Check cache first
        if cache_key in self.component_cache:
            self._update_access_order(cache_key)
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.component_cache[cache_key]

        # Record cache miss and fetch from server
        self.logger.info(f"Cache miss - fetching {cache_key}")
        fetch_start = time.time()

        state_dict = await self._fetch_weights(component_type, str(component_id))
        if state_dict is None:
            self.logger.error(f"Failed to load {cache_key}")
            return None

        fetch_time = time.time() - fetch_start
        self.logger.debug(f"Fetched {cache_key} in {fetch_time:.3f}s")

        # Create and initialize component
        component = component_class(config)
        component.load_state_dict(state_dict)

        # Add to cache with LRU eviction if needed
        self._add_to_cache(cache_key, component)

        return component
    
    def _add_to_cache(self, cache_key: str, component: nn.Module):
        """Add component to cache with LRU eviction"""
        if len(self.component_cache) >= self.cache_size and cache_key not in self.component_cache:
            # Evict least recently used component if cache is full
            if self.access_order:
                lru_component = self.access_order.pop(0)
                self.logger.debug(f"Evicting {lru_component} from cache")
                self.component_cache.pop(lru_component, None)

        # Add to cache and update access order
        self.component_cache[cache_key] = component
        self._update_access_order(cache_key)

    def _update_access_order(self, cache_key: str):
        """Update LRU access order"""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
    
    async def forward_async(self, *args, **kwargs):
        """Async forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward_async")

    def forward(self, *args, **kwargs):
        """Synchronous forward pass wrapper"""
        # Convert async to sync for compatibility with existing code
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self.logger.warning(
                "Event loop already running, cannot run forward synchronously. "
                "Use forward_async instead."
            )
            raise RuntimeError("Cannot run forward synchronously in running event loop")

        return loop.run_until_complete(self.forward_async(*args, **kwargs))

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        self.clear_cache()
        self.logger.info(f"{self.__class__.__name__} cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, "component_cache"):
                self.clear_cache()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state"""
        return {
            "cached_components": list(self.component_cache.keys()),
            "cache_size": len(self.component_cache),
            "max_cache_size": self.cache_size,
            "access_order": self.access_order.copy(),
            "memory_usage_mb": self._estimate_cache_memory_mb(),
        }

    def _estimate_cache_memory_mb(self) -> float:
        """Estimate memory usage of cached components"""
        total_params = 0
        for component in self.component_cache.values():
            for param in component.parameters():
                total_params += param.numel()

        # Assume 4 bytes per float32 parameter
        memory_bytes = total_params * 4
        return memory_bytes / (1024 * 1024)

    async def prefetch_components(self, prefetch_keys: List[Tuple[str, str, nn.Module, Any]]):
        """Prefetch components for better performance
        
        Args:
            prefetch_keys: List of (component_type, component_id, component_class, config) tuples to prefetch
        """
        prefetch_tasks = []

        for component_type, component_id, component_class, config in prefetch_keys:
            cache_key = f"{component_type}_{component_id}"
            if cache_key not in self.component_cache:
                self.logger.debug(f"Prefetching {cache_key}")
                task = asyncio.create_task(self._load_component(component_type, component_id, component_class, config))
                prefetch_tasks.append(task)

        # Run prefetch tasks in background (don't await)
        if prefetch_tasks:
            asyncio.create_task(self._run_prefetch_background(prefetch_tasks))

    async def _run_prefetch_background(self, tasks: List[asyncio.Task]):
        """Run prefetch tasks in background"""
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            self.logger.warning(f"Prefetch failed: {e}")

    def clear_cache(self):
        """Clear the component cache"""
        self.component_cache.clear()
        self.access_order.clear()
        self.logger.info("Cache cleared")

    async def warmup(self, components: List[Tuple[str, str, nn.Module, Any]]):
        """Warm up cache by preloading specific components
        
        Args:
            components: List of (component_type, component_id, component_class, config) tuples to preload
        """
        self.logger.info(f"Warming up cache with {len(components)} components")

        warmup_tasks = []
        for component_type, component_id, component_class, config in components:
            task = asyncio.create_task(self._load_component(component_type, component_id, component_class, config))
            warmup_tasks.append(task)

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks)
            self.logger.info(f"Cache warmed up with {len(warmup_tasks)} components")

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            "total_inferences": self._total_inferences,
            "avg_inference_time": self._avg_inference_time,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self._total_component_accesses == 0:
            return 0.0
        return self._cache_hits / self._total_component_accesses
