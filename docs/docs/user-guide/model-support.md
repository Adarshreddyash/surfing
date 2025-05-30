# Model Support

Surfing Weights supports various transformer model architectures through its streaming model implementations. This guide covers the supported model types, how to use them, and how to extend support for new architectures.

## Supported Model Types

Currently, Surfing Weights provides native support for the following model architectures:

- **BERT Models**: Through `StreamingBertModel`
- **LLaMA Models**: Through `StreamingLlamaModel`
- **T5 Models**: Through `StreamingT5Model`
- **GPT Models**: Through `StreamingGPTModel` (optional dependency)

## Model Architecture Overview

Each model implementation follows a common architecture pattern:

1. **Lightweight Local Components**: 
   - Embeddings and final layers are loaded locally
   - These components are typically small and frequently accessed
   - Examples: token embeddings, normalization layers

2. **Streamed Components**:
   - Transformer layers are loaded on-demand
   - Cached using LRU (Least Recently Used) strategy
   - Automatically evicted when cache is full

3. **Smart Caching**:
   - Configurable cache size for layers
   - Automatic prefetching of next layers
   - Cache hit/miss statistics tracking

## Using Models

Here's an example of using a LLaMA model with streaming weights:

```python
from streaming_weights import StreamingLlamaModel, WeightServer
from transformers import LlamaTokenizer

# Initialize the model
model = StreamingLlamaModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    websocket_uri="ws://localhost:8765",
    cache_size=5  # Cache size for layers
)

# Warm up the model (preload first few layers)
await model.warmup([0, 1, 2])

# Use the model
outputs = await model.forward_async(
    input_ids=input_ids,
    attention_mask=attention_mask,
    enable_prefetch=True  # Enable automatic prefetching
)
```

## Model Configuration

All streaming models support the following configuration options:

- `model_name`: HuggingFace model identifier or local path
- `websocket_uri`: WebSocket URI for the weight server
- `cache_size`: Number of layers to keep in memory

## Advanced Features

### 1. Layer Prefetching

Models support automatic prefetching of upcoming layers:

```python
# Enable prefetching in forward pass
outputs = await model.forward_async(
    input_ids=input_ids,
    attention_mask=attention_mask,
    enable_prefetch=True
)

# Manual prefetch of specific layers
await model.prefetch_next_layers(current_layer=2, prefetch_count=2)
```

### 2. Cache Management

Monitor and control the layer cache:

```python
# Get cache statistics
cache_info = model.get_cache_info()
print(f"Cache hits: {cache_info['hits']}")
print(f"Cache misses: {cache_info['misses']}")
print(f"Hit rate: {cache_info['hit_rate']:.2%}")
```

### 3. Model Chunking

Before using a model, it needs to be chunked for streaming:

```python
from streaming_weights import ModelChunker

# Chunk model to local filesystem
chunker = ModelChunker(
    model_name="model-name",
    output_dir="./model_chunks"
)
chunk_info = await chunker.chunk_model()

# Or chunk to S3
chunker = ModelChunker(
    model_name="model-name",
    storage_backend=s3_backend
)
chunk_info = await chunker.chunk_model()
```

## Storage Backends

Models can be chunked and stored using different backends:

1. **Filesystem Backend** (default):
   - Local storage of model chunks
   - Fastest access for local deployment

2. **S3 Backend**:
   - Cloud storage of model chunks
   - Good for distributed deployments
   - Automatic compression support

## Performance Considerations

1. **Cache Size**:
   - Larger cache = better performance but more memory
   - Recommended: 3-5 layers for most use cases

2. **Prefetching**:
   - Enable for sequential processing
   - Disable for random access patterns

3. **Initial Warmup**:
   - Preload frequently used layers
   - Reduces initial latency

## Examples

Check out complete examples in the `examples/` directory:

- `llama_example.py`: Basic LLaMA model usage
- `llama_s3_example.py`: Using LLaMA with S3 storage
- `inference_example.py`: Optimized inference setup

## Error Handling

All model operations include proper error handling:

- Network errors: Automatic retries
- Missing layers: Fallback to uninitialized layers
- Cache errors: Automatic recovery

## Extending Model Support

To add support for a new model architecture:

1. Create a new class inheriting from `StreamingBaseModel`
2. Implement the required methods:
   - `__init__`: Initialize model config and local components
   - `_load_layer`: Layer loading logic
   - `forward_async`: Model-specific forward pass
3. Add chunking support in `ModelChunker`