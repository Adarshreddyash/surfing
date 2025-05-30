# streaming_weights/models/gpt.py
import torch
import time
from transformers import GPT2Config, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from typing import Optional, List

from .base import StreamingBaseModel


class StreamingGPTModel(StreamingBaseModel):
    """Streaming implementation of GPT-style models.
    
    This class provides streaming access to GPT-style transformer models,
    loading blocks on-demand and caching frequently used components.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__(model_name, websocket_uri, cache_size)

        self.config = GPT2Config.from_pretrained(model_name)

        # Load only embeddings locally (lightweight components)
        base_model = GPT2Model.from_pretrained(model_name)
        self.wte = base_model.wte  # Word token embeddings
        self.wpe = base_model.wpe  # Position embeddings
        self.drop = base_model.drop  # Dropout
        self.ln_f = base_model.ln_f  # Final layer norm

    async def _load_block(self, block_idx: int) -> Optional[GPT2Block]:
        """Load a specific transformer block with error handling and monitoring"""
        return await self._load_component("block", str(block_idx), GPT2Block, self.config)

    async def forward_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
        enable_prefetch: bool = True,
    ):
        """Async forward pass with streaming blocks"""
        start_time = time.time()
        self._total_inferences += 1

        # Validate inputs
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)

        # Prepare attention mask (GPT style)
        # GPT uses a causal mask that masks out all tokens after the current one
        attention_mask = attention_mask.view(batch_size, -1)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=torch.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0  # Convert to additive mask

        # Process embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        # Initialize past key values if needed
        if past_key_values is None:
            past_key_values = [None] * self.config.n_layer

        # Process each transformer block (streaming)
        presents = [] if use_cache else None

        for i in range(self.config.n_layer):
            # Prefetch next block if enabled
            if enable_prefetch and i < self.config.n_layer - 1:
                next_block_idx = i + 1
                await self.prefetch_components([
                    ("block", str(next_block_idx), GPT2Block, self.config)
                ])

            # Load current block
            block = await self._load_block(i)
            if block is None:
                # Fallback: use a new uninitialized block if loading fails
                self.logger.warning(f"Using uninitialized fallback for block {i}")
                block = GPT2Block(self.config)

            # Process block
            layer_outputs = block(
                hidden_states,
                layer_past=past_key_values[i],
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                presents.append(layer_outputs[1])

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

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

    async def prefetch_next_blocks(self, current_block: int, prefetch_count: int = 1):
        """Prefetch next blocks for better performance"""
        prefetch_keys = []

        for i in range(1, prefetch_count + 1):
            next_block = current_block + i
            if next_block < self.config.n_layer:
                prefetch_keys.append(("block", str(next_block), GPT2Block, self.config))

        await self.prefetch_components(prefetch_keys)

    async def warmup(self, block_indices: Optional[List[int]] = None):
        """Warm up cache by preloading specific blocks"""
        if block_indices is None:
            block_indices = list(
                range(min(self.cache_size, self.config.n_layer))
            )

        self.logger.info(f"Warming up cache with blocks: {block_indices}")

        components = [
            ("block", str(block_idx), GPT2Block, self.config)
            for block_idx in block_indices
            if block_idx < self.config.n_layer
        ]

        await super().warmup(components)
