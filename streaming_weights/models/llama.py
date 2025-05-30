# streaming_weights/models/llama.py
import torch
import time
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Optional, List, Tuple

from .base import StreamingBaseModel


class StreamingLlamaModel(StreamingBaseModel):
    """Streaming implementation of LLaMA models.
    
    This class provides streaming access to LLaMA transformer models,
    loading layers on-demand and caching frequently used components.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__(model_name, websocket_uri, cache_size)

        self.config = LlamaConfig.from_pretrained(model_name)

        # Load only embeddings and other lightweight components locally
        base_model = LlamaModel.from_pretrained(model_name)
        self.embed_tokens = base_model.embed_tokens
        self.norm = base_model.norm  # Final normalization layer
        
        # Store rotary embedding parameters for position encoding
        if hasattr(base_model.layers[0].self_attn, "rotary_emb"):
            self.rotary_emb = base_model.layers[0].self_attn.rotary_emb

    async def _load_layer(self, layer_idx: int) -> Optional[LlamaDecoderLayer]:
        """Load a specific transformer layer with error handling and monitoring"""
        return await self._load_component("layer", str(layer_idx), LlamaDecoderLayer, self.config)

    async def forward_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        enable_prefetch: bool = True,
    ):
        """Async forward pass with streaming layers and enhanced features"""
        start_time = time.time()
        self._total_inferences += 1

        # Validate inputs
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            # Create position ids based on input sequence
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)

        # Process embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Initialize past key values if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.config.hidden_sizes) if hasattr(self.config, 'hidden_sizes') else [None] * self.config.num_hidden_layers

        # Prepare causal attention mask
        # LLaMA uses a causal mask that masks out all tokens after the current one
        if attention_mask is not None:
            # Extend attention mask for causal attention
            attention_mask = self._prepare_attention_mask(attention_mask, seq_length)

        # Process each transformer layer (streaming)
        presents = [] if use_cache else None

        num_layers = len(self.config.hidden_sizes) if hasattr(self.config, 'hidden_sizes') else self.config.num_hidden_layers
        
        for i in range(num_layers):
            # Prefetch next layer if enabled
            if enable_prefetch and i < num_layers - 1:
                next_layer_idx = i + 1
                await self.prefetch_components([
                    ("layer", str(next_layer_idx), LlamaDecoderLayer, self.config)
                ])

            # Load current layer
            layer = await self._load_layer(i)
            if layer is None:
                # Fallback: use a new uninitialized layer if loading fails
                self.logger.warning(f"Using uninitialized fallback for layer {i}")
                layer = LlamaDecoderLayer(self.config)

            # Process layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i] if past_key_values is not None else None,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                presents.append(layer_outputs[1])

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        # Update inference time stats
        inference_time = time.time() - start_time
        self._avg_inference_time = (
            (self._avg_inference_time * (self._total_inferences - 1) + inference_time)
            / self._total_inferences
        )

        self.logger.debug(f"Inference completed in {inference_time:.3f}s")
        
        # Return output based on use_cache
        if use_cache:
            return hidden_states, presents
        else:
            return hidden_states

    def _prepare_attention_mask(self, attention_mask, seq_length):
        """Prepare causal attention mask for LLaMA attention"""
        # LLaMA uses a specific attention mask format
        # Convert to 4D mask format [batch_size, 1, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
        
        # Add causal mask - only attend to positions in the past
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=torch.bool, device=attention_mask.device),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        
        # Combine masks: attention_mask & causal_mask
        # Convert to float and set masked positions to -inf
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        causal_mask = causal_mask.to(dtype=torch.float32) * -10000.0
        extended_attention_mask = extended_attention_mask + causal_mask
        
        return extended_attention_mask

    async def prefetch_next_layers(self, current_layer: int, prefetch_count: int = 1):
        """Prefetch next layers for better performance"""
        prefetch_keys = []
        num_layers = len(self.config.hidden_sizes) if hasattr(self.config, 'hidden_sizes') else self.config.num_hidden_layers

        for i in range(1, prefetch_count + 1):
            next_layer = current_layer + i
            if next_layer < num_layers:
                prefetch_keys.append(("layer", str(next_layer), LlamaDecoderLayer, self.config))

        await self.prefetch_components(prefetch_keys)

    async def warmup(self, layer_indices: Optional[List[int]] = None):
        """Warm up cache by preloading specific layers"""
        num_layers = len(self.config.hidden_sizes) if hasattr(self.config, 'hidden_sizes') else self.config.num_hidden_layers
        
        if layer_indices is None:
            layer_indices = list(range(min(self.cache_size, num_layers)))

        self.logger.info(f"Warming up cache with layers: {layer_indices}")

        components = [
            ("layer", str(layer_idx), LlamaDecoderLayer, self.config)
            for layer_idx in layer_indices
            if layer_idx < num_layers
        ]

        await super().warmup(components)
