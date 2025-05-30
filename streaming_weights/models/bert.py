# streaming_weights/models/bert.py
import torch
import time
from transformers import BertConfig, BertModel, BertLayer
from typing import Optional, List

from .base import StreamingBaseModel


class StreamingBertModel(StreamingBaseModel):
    """Streaming implementation of BERT models.
    
    This class provides streaming access to BERT transformer models,
    loading layers on-demand and caching frequently used components.
    """
    
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__(model_name, websocket_uri, cache_size)

        self.config = BertConfig.from_pretrained(model_name)

        # Load only embeddings and pooler locally (lightweight components)
        base_model = BertModel.from_pretrained(model_name)
        self.embeddings = base_model.embeddings
        self.pooler = base_model.pooler

        # Current loaded layer info
        self.current_layers = {}

    async def _load_layer(self, layer_idx: int) -> Optional[BertLayer]:
        """Load a specific transformer layer with error handling and monitoring"""
        return await self._load_component("layer", str(layer_idx), BertLayer, self.config)

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
                next_layer_idx = i + 1
                await self.prefetch_components([
                    ("layer", str(next_layer_idx), BertLayer, self.config)
                ])

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

    async def prefetch_next_layers(self, current_layer: int, prefetch_count: int = 1):
        """Prefetch next layers for better performance"""
        prefetch_keys = []

        for i in range(1, prefetch_count + 1):
            next_layer = current_layer + i
            if next_layer < self.config.num_hidden_layers:
                prefetch_keys.append(("layer", str(next_layer), BertLayer, self.config))

        await self.prefetch_components(prefetch_keys)

    async def warmup(self, layer_indices: Optional[List[int]] = None):
        """Warm up cache by preloading specific layers"""
        if layer_indices is None:
            layer_indices = list(
                range(min(self.cache_size, self.config.num_hidden_layers))
            )

        self.logger.info(f"Warming up cache with layers: {layer_indices}")

        components = [
            ("layer", str(layer_idx), BertLayer, self.config)
            for layer_idx in layer_indices
            if layer_idx < self.config.num_hidden_layers
        ]

        await super().warmup(components)
