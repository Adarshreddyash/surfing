# ⚠️ WARNING: This package is in developmental and beta phase. Do NOT use in production environments.

# streaming-weights
Streaming weights to edge device

# 🌊 Streaming Weights

A Python Server for streaming transformer model weights to enable efficient AI inference on edge devices, IoT, and mobile platforms.

## 🚀 Key Features

- **Zero Local Storage**: Stream model weights on-demand instead of downloading entire models
- **Smart Caching**: LRU cache for frequently used layers with configurable cache size
- **Edge Optimized**: Designed for resource-constrained devices (IoT, mobile, embedded)
- **HuggingFace Compatible**: Works with existing transformer models from HuggingFace Hub
- **Async Architecture**: Non-blocking inference with async/await support
<!-- - **Production Ready**: Monitoring, compression, and distributed caching support -->

## 📦 Installation

```bash
pip install streaming-weights

# With optional dependencies
pip install streaming-weights[server,dev]
```

## 🔧 Quick Start

### 1. Chunk Your Model

```python
from streaming_weights import ModelChunker

# Chunk a model for streaming
chunker = ModelChunker("prajjwal1/bert-tiny", "./chunks/bert-tiny")
chunk_info = chunker.chunk_model()
print(f"Model chunked into {len(chunk_info['chunks'])} pieces")
```

### 2. Start Weight Server

```python
from streaming_weights import WeightServer
import asyncio

async def start_server():
    server = WeightServer("./chunks/bert-tiny", port=8765)
    await server.start_server()

# Run server
asyncio.run(start_server())
```

### 3. Stream Model Inference

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
        cache_size=2
    )
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Hello world!", return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = await model.forward_async(**inputs)
    
    print(f"Output shape: {outputs[0].shape}")
    return outputs

# Run inference
outputs = asyncio.run(run_inference())
```

## 🏗️ Architecture

### Chunking Strategy
- **Embeddings**: Loaded locally (lightweight)
- **Transformer Layers**: Streamed on-demand
- **Pooler**: Loaded locally (lightweight)

### Caching System
- **LRU Eviction**: Least recently used layers are evicted first
- **Configurable Size**: Control memory usage with cache size limits
- **Hit Rate Monitoring**: Track cache performance

### Network Protocol
- **WebSocket**: Low-latency bidirectional communication
- **JSON Messages**: Simple request/response format
- **Binary Weights**: Efficient tensor serialization

## 📊 Performance Benefits

| Metric | Traditional Loading | Streaming Weights |
|--------|-------------------|------------------|
| Initial Memory | 100% model size | ~5% model size |
| Startup Time | Full download | Instant |
| Storage Required | Full model | None |
| Network Usage | One-time large | On-demand small |

## 🔧 Advanced Usage

### Custom Configuration

```python
from streaming_weights import StreamingConfig, StreamingBertModel

config = StreamingConfig(
    server_host="your-server.com",
    server_port=8765,
    cache_size=5,
    enable_compression=True,
    prefetch_layers=True
)

model = StreamingBertModel(config=config)
```

### Performance Monitoring

```python
from streaming_weights import StreamingMonitor

monitor = StreamingMonitor()
model = StreamingBertModel("bert-base", monitor=monitor)

# After inference
metrics = monitor.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

### Distributed Deployment

```python
from streaming_weights import AdvancedWeightServer

# Server with Redis caching
server = AdvancedWeightServer(
    chunks_dir="./chunks",
    redis_url="redis://localhost:6379",
    port=8765
)

await server.start_servers()
```

## 🛠️ CLI Tools

### Chunk Models

Chunking splits a model into smaller pieces that can be streamed on demand:

```bash
# Basic chunking
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny

# With compression (recommended for edge devices)
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --compress

# With verbose logging
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --verbose
```

After chunking, your output directory will contain the model's embedding layer, individual transformer layers, pooler layer, configuration, and chunk metadata.

### Start Weight Server

The weight server provides model chunks on demand via WebSocket:

```bash
# Start a weight server on the default port (8765)
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny

# Specify a custom port
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --port 9000

# With verbose logging
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --verbose
```

Once the server is running, you can connect to it using the StreamingBertModel as shown in the examples above.

## 📈 Use Cases

- **IoT Devices**: Run large models on Raspberry Pi, edge computers
- **Mobile Apps**: AI inference without large app downloads
- **Serverless**: Cold start optimization for cloud functions
- **Development**: Fast model switching during experimentation
- **Multi-tenant**: Share models across multiple applications

## 🔒 Security Considerations

- Use WSS (WebSocket Secure) in production
- Implement authentication for weight servers
- Validate chunk integrity with checksums
- Rate limiting for DoS
