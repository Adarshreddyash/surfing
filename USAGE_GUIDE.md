# 🌊 Surfing - Usage Guide

This guide provides detailed instructions for using the Surfing weights streaming system.

## 📦 Chunking Models

Before you can stream a model, you need to chunk it into smaller pieces. This process only needs to be done once per model.

### Command Line Interface

```bash
# Basic usage
python -m streaming_weights.chunker <model_name> --output-dir <output_directory>

# Example with compression enabled
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --compress

# With verbose logging
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --verbose
```

### Python API

```python
from streaming_weights import ModelChunker

# Initialize chunker
chunker = ModelChunker(
    model_name="prajjwal1/bert-tiny",
    output_dir="./chunks/bert-tiny",
    compress=True  # Optional: Enable compression
)

# Chunk the model
chunk_info = chunker.chunk_model()
print(f"Model chunked into {len(chunk_info['chunks'])} pieces")
print(f"Total size: {chunk_info['total_size_mb']:.2f} MB")
```

### Chunk Output Structure

After chunking, your output directory will contain:

- `embeddings.pt`: The model's embedding layer
- `layer_0.pt`, `layer_1.pt`, etc.: Individual transformer layers
- `pooler.pt`: The model's pooler layer (if applicable)
- `config.json`: The model configuration
- `chunk_info.json`: Metadata about the chunked model

## 🚀 Starting a Weight Server

The weight server provides model chunks on demand via WebSocket.

### Command Line Interface

```bash
# Basic usage
python -m streaming_weights.weight_server --chunks-dir <chunks_directory> --port <port_number>

# Example
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --port 8765

# With verbose logging
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --port 8765 --verbose
```

### Python API

```python
import asyncio
from streaming_weights import WeightServer

async def start_server():
    # Initialize server
    server = WeightServer(
        chunks_dir="./chunks/bert-tiny",
        port=8765
    )
    
    # Start server (runs indefinitely)
    await server.start_server()

# Run the server
asyncio.run(start_server())
```

## 🧠 Running Inference with Streaming Weights

```python
import asyncio
import torch
from transformers import AutoTokenizer
from streaming_weights import StreamingBertModel

async def run_inference():
    # Initialize streaming model
    model = StreamingBertModel(
        model_name="prajjwal1/bert-tiny",
        websocket_uri="ws://localhost:8765",
        cache_size=2  # Number of layers to cache
    )
    
    # Optional: Warm up the cache
    await model.warmup([0, 1])  # Pre-load layers 0 and 1
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = await model.forward_async(**inputs)
    
    # Get cache statistics
    print(f"Cache info: {model.get_cache_info()}")
    print(f"Inference stats: {model.get_inference_stats()}")
    
    return outputs

# Run inference
outputs = asyncio.run(run_inference())
```

## 🔧 Advanced Configuration

### Custom Cache Size

Adjust the cache size based on your available memory:

```python
model = StreamingBertModel(
    model_name="prajjwal1/bert-tiny",
    websocket_uri="ws://localhost:8765",
    cache_size=5  # Cache more layers for better performance
)
```

### Prefetching

Enable or disable prefetching of upcoming layers:

```python
# Disable prefetching
outputs = await model.forward_async(**inputs, enable_prefetch=False)
```

### Monitoring

Get detailed performance metrics:

```python
# After running inference
stats = model.get_inference_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
```

## 🔒 Security Best Practices

1. **Use WSS (WebSocket Secure)** in production:
   ```python
   model = StreamingBertModel(
       websocket_uri="wss://your-secure-server.com:8765"
   )
   ```

2. **Validate chunk integrity** using checksums:
   ```python
   from streaming_weights.utils import calculate_chunk_hash
   
   chunk_hash = calculate_chunk_hash("./chunks/bert-tiny/layer_0.pt")
   print(f"Chunk hash: {chunk_hash}")
   ```

3. **Implement rate limiting** on your weight server to prevent DoS attacks.
