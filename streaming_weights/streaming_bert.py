# streaming_weights/streaming_bert.py
import torch
import torch.nn as nn
import asyncio
import websockets
import json
import io
import time
from transformers import BertConfig, BertModel, BertLayer
from typing import Optional, Dict, Any, List
import logging


class StreamingBertModel(nn.Module):
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__()

        self.model_name = model_name
        self.websocket_uri = websocket_uri
        self.config = BertConfig.from_pretrained(model_name)

        # Load only embeddings and pooler locally (lightweight components)
        base_model = BertModel.from_pretrained(model_name)
        self.embeddings = base_model.embeddings
        self.pooler = base_model.pooler

        # Layer cache (LRU-style)
        self.layer_cache: Dict[int, BertLayer] = {}
        self.cache_size = cache_size
        self.access_order = []  # For LRU eviction

        # Current loaded layer info
        self.current_layers = {}

        # Stats tracking
        self._total_layer_accesses = 0
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
                # Remove unsupported 'timeout' argument for websockets.connect
                async with websockets.connect(
                    self.websocket_uri, ping_interval=20, ping_timeout=10
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

    async def _load_layer(self, layer_idx: int) -> Optional[BertLayer]:
        """Load a specific transformer layer with error handling and monitoring"""
        # Track layer access for stats
        self._total_layer_accesses += 1
        
        # Check cache first
        if layer_idx in self.layer_cache:
            self._update_access_order(layer_idx)
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for layer {layer_idx}")
            return self.layer_cache[layer_idx]

        # Record cache miss and fetch from server
        self.logger.info(f"Cache miss - fetching layer {layer_idx}")
        fetch_start = time.time()

        state_dict = await self._fetch_weights("LAYER", str(layer_idx))
        if state_dict is None:
            self.logger.error(f"Failed to load layer {layer_idx}")
            return None

        fetch_time = time.time() - fetch_start
        self.logger.debug(f"Fetched layer {layer_idx} in {fetch_time:.3f}s")

        # Create and initialize layer
        layer = BertLayer(self.config)
        layer.load_state_dict(state_dict)

        # Add to cache with LRU eviction if needed
        self._add_to_cache(layer_idx, layer)

        return layer

    def _add_to_cache(self, layer_idx: int, layer: BertLayer):
        """Add layer to cache with LRU eviction"""
        if len(self.layer_cache) >= self.cache_size and layer_idx not in self.layer_cache:
            # Evict least recently used layer if cache is full
            if self.access_order:
                lru_layer = self.access_order.pop(0)
                self.logger.debug(f"Evicting layer {lru_layer} from cache")
                self.layer_cache.pop(lru_layer, None)

        # Add to cache and update access order
        self.layer_cache[layer_idx] = layer
        self._update_access_order(layer_idx)

    def _update_access_order(self, layer_idx: int):
        """Update LRU access order"""
        if layer_idx in self.access_order:
            self.access_order.remove(layer_idx)
        self.access_order.append(layer_idx)

    async def forward_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        enable_prefetch: bool = True,
    ):
        """Async forward pass with streaming layers and enhanced features"""
        start_time = time.time()
        self._total_inferences += 1

        # Validate inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask
        head_mask = [None] * self.config.num_hidden_layers

        # Process embeddings (local)
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # Process each transformer layer (streaming)
        layer_output = embedding_output
        all_encoder_layers = []

        for i in range(self.config.num_hidden_layers):
            # Prefetch next layer if enabled
            if enable_prefetch and i < self.config.num_hidden_layers - 1:
                await self.prefetch_next_layers(i, prefetch_count=1)

            # Load current layer
            layer = await self._load_layer(i)
            if layer is None:
                # Fallback: use a new uninitialized layer if loading fails
                self.logger.warning(f"Using uninitialized fallback for layer {i}")
                layer = BertLayer(self.config)

            # Process layer
            layer_output = layer(
                layer_output,
                extended_attention_mask,
                head_mask[i],
            )[0]

            all_encoder_layers.append(layer_output)

        # Apply pooler (local)
        pooled_output = self.pooler(layer_output)

        # Update inference time stats
        inference_time = time.time() - start_time
        self._avg_inference_time = (
            (self._avg_inference_time * (self._total_inferences - 1) + inference_time)
            / self._total_inferences
        )

        self.logger.debug(f"Inference completed in {inference_time:.3f}s")
        return layer_output, pooled_output

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
        self.logger.info("StreamingBertModel cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, "layer_cache"):
                self.clear_cache()
        except Exception:
            pass  # Ignore errors during cleanup

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state"""
        return {
            "cached_layers": list(self.layer_cache.keys()),
            "cache_size": len(self.layer_cache),
            "max_cache_size": self.cache_size,
            "access_order": self.access_order.copy(),
            "memory_usage_mb": self._estimate_cache_memory_mb(),
        }

    def _estimate_cache_memory_mb(self) -> float:
        """Estimate memory usage of cached layers"""
        total_params = 0
        for layer in self.layer_cache.values():
            for param in layer.parameters():
                total_params += param.numel()

        # Assume 4 bytes per float32 parameter
        memory_bytes = total_params * 4
        return memory_bytes / (1024 * 1024)

    async def prefetch_next_layers(self, current_layer: int, prefetch_count: int = 1):
        """Prefetch next layers for better performance"""
        prefetch_tasks = []

        for i in range(1, prefetch_count + 1):
            next_layer = current_layer + i
            if (
                next_layer < self.config.num_hidden_layers
                and next_layer not in self.layer_cache
            ):
                self.logger.debug(f"Prefetching layer {next_layer}")
                task = asyncio.create_task(self._load_layer(next_layer))
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
        """Clear the layer cache"""
        self.layer_cache.clear()
        self.access_order.clear()
        self.logger.info("Cache cleared")

    async def warmup(self, layer_indices: Optional[List[int]] = None):
        """Warm up cache by preloading specific layers"""
        if layer_indices is None:
            layer_indices = list(
                range(min(self.cache_size, self.config.num_hidden_layers))
            )

        self.logger.info(f"Warming up cache with layers: {layer_indices}")

        warmup_tasks = []
        for layer_idx in layer_indices:
            if layer_idx < self.config.num_hidden_layers:
                task = asyncio.create_task(self._load_layer(layer_idx))
                warmup_tasks.append(task)

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks)
            self.logger.info(f"Cache warmed up with {len(warmup_tasks)} layers")

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            "total_inferences": self._total_inferences,
            "avg_inference_time": self._avg_inference_time,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self._total_layer_accesses == 0:
            return 0.0
        return self._cache_hits / self._total_layer_accesses
