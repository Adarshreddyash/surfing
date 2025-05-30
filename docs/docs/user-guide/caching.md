# Caching System

Surfing Weights implements a sophisticated caching system to optimize performance and memory usage. This guide explains how the caching system works and how to configure it for your needs.

## Overview

The caching system operates at two levels:

1. Server-Side Caching
   - Caches raw model weights
   - Shared across all clients
   - Memory-efficient storage

2. Client-Side Caching
   - Caches loaded model components
   - Per-client cache
   - Optimized for inference

## Server-Side Cache

### Configuration

```python
from streaming_weights import WeightServer

server = WeightServer(
    model_path="./chunks/bert-tiny",
    cache_size_mb=200  # Set cache size in megabytes
)
```

Command line configuration:
```bash
streaming-weights-server --chunks-dir ./chunks/bert-tiny \
    --cache-size 200  # Cache size in MB
```

### Features

1. LRU (Least Recently Used) Eviction
   - Automatically removes least used weights
   - Optimizes memory usage
   - Adapts to access patterns

2. Size-Based Management
   - Configurable maximum size
   - Automatic eviction when full
   - Memory usage monitoring

## Client-Side Cache

### Configuration

```python
from streaming_weights import StreamingBertModel

model = StreamingBertModel(
    model_name="prajjwal1/bert-tiny",
    cache_size=3  # Number of layers to cache
)
```

### Features

1. Component-Level Caching
   - Caches entire model layers
   - Maintains layer state
   - Optimizes inference speed

2. Smart Prefetching
   ```python
   # Enable prefetching for better performance
   outputs = await model.forward_async(
       input_ids=inputs,
       enable_prefetch=True,
       prefetch_count=2  # Prefetch next 2 layers
   )
   ```

3. Cache Warmup
   ```python
   # Preload specific layers
   await model.warmup(layer_indices=[0, 1, 2])
   ```

## Performance Monitoring

### Cache Statistics

```python
# Get cache performance metrics
stats = model.get_inference_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Average inference time: {stats['avg_inference_time']:.3f}s")

# Get current cache state
cache_info = model.get_cache_info()
print(f"Cached components: {cache_info['cached_components']}")
print(f"Cache memory usage: {cache_info['memory_usage_mb']:.2f} MB")
```

### Cache Management

```python
# Clear the cache manually
model.clear_cache()

# Update cache size at runtime
model.cache_size = 5  # Increase cache size
```

## Advanced Features

### Cache Optimization

1. Access Pattern Optimization
   ```python
   # Order layers for optimal caching
   await model.warmup([0, 1, 2])  # Cache first layers
   await model.prefetch_next_layers(2, prefetch_count=2)  # Prefetch next layers
   ```

2. Memory Management
   ```python
   # Monitor and adjust cache size
   if model.get_cache_info()['memory_usage_mb'] > 1000:
       model.cache_size = model.cache_size - 1
   ```

### Distributed Caching

When using multiple servers:

```python
from streaming_weights import AdvancedWeightServer

server = AdvancedWeightServer(
    chunks_dir="./chunks",
    redis_url="redis://localhost:6379",  # Redis for distributed caching
    cache_size_mb=1000
)
```

## Best Practices

1. Cache Size Configuration
   - Set server cache size based on available RAM
   - Adjust client cache size based on model architecture
   - Monitor cache hit rates for optimization

2. Performance Optimization
   - Use warmup for frequently accessed layers
   - Enable prefetching for sequential access
   - Clear cache when switching tasks

3. Memory Management
   - Monitor memory usage with get_cache_info()
   - Adjust cache sizes based on workload
   - Clear cache when memory pressure is high

## Troubleshooting

### Common Issues

1. High Memory Usage
   - Reduce cache size
   - Clear cache more frequently
   - Monitor with get_cache_info()

2. Poor Cache Performance
   - Check cache hit rates
   - Adjust cache size
   - Review access patterns

### Cache Monitoring

```python
# Monitor cache performance
while running_inference:
    stats = model.get_inference_stats()
    if stats['cache_hit_rate'] < 0.5:
        print("Warning: Low cache hit rate")
    await asyncio.sleep(60)  # Check every minute
```

## Next Steps

- Learn about [Error Handling](error-handling.md)
- Explore [Model Support](model-support.md)
- See [Example Use Cases](../examples/basic-usage.md)