# 🌊 Surfing Weights

Welcome to Surfing Weights - a Python server for streaming transformer model weights to enable efficient AI inference on edge devices, IoT, and mobile platforms.

## Overview

Surfing Weights solves the challenge of deploying large AI models to resource-constrained environments by streaming model weights on-demand instead of requiring the entire model to be downloaded upfront.

### Key Features

- **🚫 Zero Local Storage**: Stream model weights as needed instead of downloading entire models
- **📦 Smart Caching**: LRU cache for frequently used layers with configurable cache size
- **📱 Edge Optimized**: Designed for resource-constrained devices (IoT, mobile, embedded)
- **🤗 HuggingFace Compatible**: Works with existing transformer models from HuggingFace Hub
- **⚡ Async Architecture**: Non-blocking inference with async/await support
- **🚀 Production Ready**: Monitoring, compression, and distributed caching support

## Quick Example

```python
from streaming_weights import WeightServer
import asyncio

async def start_server():
    server = WeightServer("./chunks/bert-tiny", port=8765)
    await server.start_server()

asyncio.run(start_server())
```

## Getting Started

- [Installation Guide](getting-started/installation.md) - Install Surfing Weights
- [Quick Start](getting-started/quick-start.md) - Start streaming weights in minutes
- [Core Concepts](getting-started/concepts.md) - Learn the fundamental concepts

## Why Surfing Weights?

Traditional approaches to deploying AI models require downloading and storing the entire model locally. This becomes impractical for:

- Edge devices with limited storage
- Mobile applications where model size impacts app size
- IoT devices with constrained resources
- Environments requiring multiple model variants

Surfing Weights enables these scenarios by:

1. Streaming only the required weights on-demand
2. Intelligently caching frequently used layers
3. Minimizing memory usage and network bandwidth
4. Supporting distributed deployment scenarios

## Next Steps

1. Follow the [Installation Guide](getting-started/installation.md) to set up Surfing Weights
2. Try the [Quick Start Tutorial](getting-started/quick-start.md)
3. Explore [Example Use Cases](examples/basic-usage.md)
4. Read the [API Documentation](api/weight-server.md)
