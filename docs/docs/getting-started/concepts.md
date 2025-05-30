# Core Concepts

## Overview

Surfing Weights is built around several key concepts that enable efficient streaming of model weights. Understanding these concepts will help you make the most of the library.

## Weight Chunking

### What is Chunking?

Chunking is the process of breaking down a large model into smaller, manageable pieces that can be:

- Stored efficiently
- Transmitted quickly
- Loaded on demand

### How Chunking Works

1. Model Analysis
   - The model's architecture is analyzed
   - Weights are grouped by layers
   
2. Chunk Creation
   - Each layer's weights are saved separately
   - Metadata about chunks is stored
   - Configuration is preserved

## Weight Streaming

### The Streaming Process

1. Initial Setup
   - Client connects to weight server
   - Model architecture is initialized
   - Only metadata is loaded initially

2. On-Demand Loading
   - Weights are requested as needed
   - Server streams requested chunks
   - Client processes received weights

3. Smart Caching
   - Frequently used weights are cached
   - LRU policy manages cache size
   - Cold weights are released

## Storage Backends

Surfing Weights supports multiple storage backends:

1. Local Filesystem
   - Direct access to local files
   - Fastest for local deployment
   
2. Amazon S3
   - Cloud-based storage
   - Scalable and reliable
   - Good for distributed setups

3. Custom Backends
   - Extensible interface
   - Support for other storage systems

## Caching System

### Cache Levels

1. Server-Side Cache
   - Reduces storage backend access
   - Shared across clients
   - Configurable size

2. Client-Side Cache
   - Reduces network requests
   - Per-client caching
   - Memory-efficient

### Cache Management

- LRU (Least Recently Used) policy
- Configurable cache sizes
- Automatic memory management

## Model Support

Surfing Weights supports various transformer architectures:

- BERT
- GPT
- T5
- LLaMA
- Custom models

Each model type has specific:
- Chunking strategies
- Loading patterns
- Optimization techniques

## Monitoring

Built-in monitoring provides:

1. Performance Metrics
   - Request latency
   - Cache hit rates
   - Memory usage

2. Health Checks
   - Server status
   - Backend connectivity
   - Resource utilization

## Next Steps

- See [Configuration Guide](../user-guide/configuration.md)
- Learn about [Storage Backends](../user-guide/storage-backends.md)
- Explore [API Reference](../api/weight-server.md)