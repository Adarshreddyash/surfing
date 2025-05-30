# Quick Start Guide

This guide will help you get up and running with Surfing Weights in minutes.

## Basic Setup

1. First, install Surfing Weights:
```bash
pip install streaming-weights[server]
```

## Chunking a Model

Before streaming a model, you need to chunk it into smaller pieces:

```python
from streaming_weights import ModelChunker

# Initialize the chunker with a HuggingFace model
chunker = ModelChunker("prajjwal1/bert-tiny", "./chunks/bert-tiny")

# Chunk the model
chunk_info = chunker.chunk_model()
print(f"Model chunked into {len(chunk_info['chunks'])} pieces")
```

## Starting the Weight Server

Create a server to stream your chunked model:

```python
from streaming_weights import WeightServer
import asyncio

async def start_server():
    # Initialize the server with your chunked model
    server = WeightServer(
        model_path="./chunks/bert-tiny",
        port=8765,
        cache_size="2GB"  # Optional: Set cache size
    )
    
    # Start the server
    await server.start_server()

# Run the server
if __name__ == "__main__":
    asyncio.run(start_server())
```

## Client Usage

Connect to the weight server and use the model:

```python
from streaming_weights import StreamingBertModel

# Initialize the streaming model
model = StreamingBertModel(
    server_url="http://localhost:8765",
    model_name="prajjwal1/bert-tiny"
)

# Use the model for inference
text = "Hello, world!"
outputs = model.encode(text)
```

## Configuration Options

Basic server configuration:

```python
server = WeightServer(
    model_path="./chunks/bert-tiny",
    port=8765,
    cache_size="2GB",
    compression=True,  # Enable weight compression
    monitoring=True    # Enable Prometheus metrics
)
```

## Next Steps

- Learn about [Core Concepts](concepts.md)
- Explore [Configuration Options](../user-guide/configuration.md)
- See [Example Use Cases](../examples/basic-usage.md)
- Read about [Storage Backends](../user-guide/storage-backends.md)