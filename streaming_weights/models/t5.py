# streaming_weights/models/t5.py
import torch
import time
from transformers import T5Config, T5Model, T5Block
from typing import Optional, List, Tuple

from .base import StreamingBaseModel


class StreamingT5Model(StreamingBaseModel):
    """Streaming implementation of T5-style encoder-decoder models.
    
    This class provides streaming access to T5-style transformer models,
    loading encoder and decoder blocks on-demand and caching frequently used components.
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        websocket_uri: str = "ws://localhost:8765",
        cache_size: int = 3,
    ):
        super().__init__(model_name, websocket_uri, cache_size)

        self.config = T5Config.from_pretrained(model_name)

        # Load only embeddings locally (lightweight components)
        base_model = T5Model.from_pretrained(model_name)
        self.shared = base_model.shared  # Shared embeddings
        self.encoder_embed_tokens = base_model.encoder.embed_tokens
        self.decoder_embed_tokens = base_model.decoder.embed_tokens
        
        # Final layer norms
        self.encoder_final_layer_norm = base_model.encoder.final_layer_norm
        self.decoder_final_layer_norm = base_model.decoder.final_layer_norm
        
        # Dropouts
        self.encoder_dropout = base_model.encoder.dropout
        self.decoder_dropout = base_model.decoder.dropout

    async def _load_encoder_block(self, block_idx: int) -> Optional[T5Block]:
        """Load a specific encoder block with error handling and monitoring"""
        return await self._load_component("encoder_block", str(block_idx), T5Block, self.config)

    async def _load_decoder_block(self, block_idx: int) -> Optional[T5Block]:
        """Load a specific decoder block with error handling and monitoring"""
        return await self._load_component("decoder_block", str(block_idx), T5Block, self.config)

    async def forward_async(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        enable_prefetch: bool = True,
    ):
        """Async forward pass with streaming blocks"""
        start_time = time.time()
        self._total_inferences += 1

        # Process encoder if encoder_outputs not provided
        if encoder_outputs is None:
            encoder_outputs = await self._encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                enable_prefetch=enable_prefetch,
            )

        # Process decoder
        decoder_outputs = await self._decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            enable_prefetch=enable_prefetch,
        )

        # Update inference time stats
        inference_time = time.time() - start_time
        self._avg_inference_time = (
            (self._avg_inference_time * (self._total_inferences - 1) + inference_time)
            / self._total_inferences
        )

        self.logger.debug(f"Inference completed in {inference_time:.3f}s")
        
        return decoder_outputs

    async def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        enable_prefetch: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Encode input sequence using T5 encoder"""
        # Prepare inputs
        batch_size, seq_length = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)

        # Process embeddings
        inputs_embeds = self.encoder_embed_tokens(input_ids)
        hidden_states = self.encoder_dropout(inputs_embeds)

        # Extended attention mask for encoder
        extended_attention_mask = self._prepare_attention_mask(attention_mask)

        # Process each encoder block (streaming)
        for i in range(self.config.num_layers):
            # Prefetch next block if enabled
            if enable_prefetch and i < self.config.num_layers - 1:
                next_block_idx = i + 1
                await self.prefetch_components([
                    ("encoder_block", str(next_block_idx), T5Block, self.config)
                ])

            # Load current block
            block = await self._load_encoder_block(i)
            if block is None:
                # Fallback: use a new uninitialized block if loading fails
                self.logger.warning(f"Using uninitialized fallback for encoder block {i}")
                block = T5Block(self.config)

            # Process block
            layer_outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
            )

            hidden_states = layer_outputs[0]

        # Apply final layer norm
        hidden_states = self.encoder_final_layer_norm(hidden_states)

        return (hidden_states,)

    async def _decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        enable_prefetch: bool = True,
    ):
        """Decode using T5 decoder with encoder outputs"""
        # Prepare inputs
        batch_size, seq_length = decoder_input_ids.shape
        
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones(batch_size, seq_length, device=decoder_input_ids.device)

        # Process embeddings
        inputs_embeds = self.decoder_embed_tokens(decoder_input_ids)
        hidden_states = self.decoder_dropout(inputs_embeds)

        # Extended attention masks for decoder
        decoder_extended_attention_mask = self._prepare_attention_mask(decoder_attention_mask, causal=True)
        encoder_extended_attention_mask = self._prepare_attention_mask(encoder_attention_mask) if encoder_attention_mask is not None else None

        # Initialize past key values if needed
        if past_key_values is None:
            past_key_values = [None] * self.config.num_decoder_layers

        # Process each decoder block (streaming)
        presents = [] if use_cache else None

        for i in range(self.config.num_decoder_layers):
            # Prefetch next block if enabled
            if enable_prefetch and i < self.config.num_decoder_layers - 1:
                next_block_idx = i + 1
                await self.prefetch_components([
                    ("decoder_block", str(next_block_idx), T5Block, self.config)
                ])

            # Load current block
            block = await self._load_decoder_block(i)
            if block is None:
                # Fallback: use a new uninitialized block if loading fails
                self.logger.warning(f"Using uninitialized fallback for decoder block {i}")
                block = T5Block(self.config)

            # Process block
            layer_outputs = block(
                hidden_states,
                attention_mask=decoder_extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                layer_past=past_key_values[i] if past_key_values is not None else None,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                presents.append(layer_outputs[1])

        # Apply final layer norm
        hidden_states = self.decoder_final_layer_norm(hidden_states)

        # Return output based on use_cache
        if use_cache:
            return hidden_states, presents
        else:
            return hidden_states

    def _prepare_attention_mask(self, attention_mask, causal=False):
        """Prepare attention mask for T5 attention"""
        # T5 uses a different attention mask format
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Add causal mask if needed
        if causal:
            seq_length = attention_mask.size(-1)
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=attention_mask.device),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            causal_mask = causal_mask.to(dtype=torch.float32) * -10000.0
            extended_attention_mask = extended_attention_mask + causal_mask
            
        return extended_attention_mask

    async def prefetch_encoder_blocks(self, current_block: int, prefetch_count: int = 1):
        """Prefetch next encoder blocks for better performance"""
        prefetch_keys = []

        for i in range(1, prefetch_count + 1):
            next_block = current_block + i
            if next_block < self.config.num_layers:
                prefetch_keys.append(("encoder_block", str(next_block), T5Block, self.config))

        await self.prefetch_components(prefetch_keys)

    async def prefetch_decoder_blocks(self, current_block: int, prefetch_count: int = 1):
        """Prefetch next decoder blocks for better performance"""
        prefetch_keys = []

        for i in range(1, prefetch_count + 1):
            next_block = current_block + i
            if next_block < self.config.num_decoder_layers:
                prefetch_keys.append(("decoder_block", str(next_block), T5Block, self.config))

        await self.prefetch_components(prefetch_keys)

    async def warmup(self, encoder_indices: Optional[List[int]] = None, decoder_indices: Optional[List[int]] = None):
        """Warm up cache by preloading specific blocks"""
        if encoder_indices is None:
            encoder_indices = list(
                range(min(self.cache_size // 2, self.config.num_layers))
            )

        if decoder_indices is None:
            decoder_indices = list(
                range(min(self.cache_size // 2, self.config.num_decoder_layers))
            )

        self.logger.info(f"Warming up cache with encoder blocks: {encoder_indices} and decoder blocks: {decoder_indices}")

        components = []
        
        # Add encoder blocks
        components.extend([
            ("encoder_block", str(block_idx), T5Block, self.config)
            for block_idx in encoder_indices
            if block_idx < self.config.num_layers
        ])
        
        # Add decoder blocks
        components.extend([
            ("decoder_block", str(block_idx), T5Block, self.config)
            for block_idx in decoder_indices
            if block_idx < self.config.num_decoder_layers
        ])

        await super().warmup(components)
