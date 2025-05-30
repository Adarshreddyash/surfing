# Error Handling

Surfing Weights implements comprehensive error handling to ensure reliability in production environments. This guide explains how errors are handled at different levels and how to implement proper error handling in your applications.

## Common Error Types

### Connection Errors

1. WebSocket Connection Errors
```python
try:
    model = StreamingBertModel(websocket_uri="ws://localhost:8765")
    outputs = await model.forward_async(inputs)
except websockets.exceptions.ConnectionClosed:
    print("Connection to weight server lost")
except websockets.exceptions.WebSocketException:
    print("WebSocket error occurred")
except asyncio.TimeoutError:
    print("Connection timed out")
```

2. Storage Backend Errors
```python
try:
    storage = S3Backend(bucket_name="model-weights")
    await storage.load("layer_0.pt")
except FileNotFoundError:
    print("Model chunk not found")
except IOError as e:
    print(f"Storage error: {e}")
```

## Automatic Retry Logic

### Model Component Loading

The base model implements automatic retries for component loading:

```python
# Retry configuration is built into StreamingBaseModel
model = StreamingBertModel(
    model_name="bert-base",
    websocket_uri="ws://localhost:8765",
    # Default retry settings:
    # - max_retries = 3
    # - initial_retry_delay = 1.0 seconds
    # - exponential_backoff = True
)
```

Internal retry logic:
```python
# This is handled automatically by the library
max_retries = 3
retry_delay = 1.0

for attempt in range(max_retries):
    try:
        # Attempt to fetch weights
        weights = await fetch_weights()
        return weights
    except ConnectionError:
        if attempt < max_retries - 1:
            # Exponential backoff
            await asyncio.sleep(retry_delay * (2**attempt))
        else:
            raise
```

## Error Recovery

### Server-Side Recovery

1. Cache Management Errors
```python
# Server automatically handles cache overflow
server = WeightServer(
    model_path="./chunks/bert-tiny",
    cache_size_mb=100  # If exceeded, LRU items are evicted
)
```

2. Storage Backend Failover
```python
from streaming_weights import WeightServer, S3Backend, FilesystemBackend

# Primary storage (S3)
s3_storage = S3Backend(bucket_name="model-weights")

# Backup storage (local filesystem)
backup_storage = FilesystemBackend("./backup_chunks")

try:
    await s3_storage.load("layer_0.pt")
except Exception:
    # Fallback to backup storage
    await backup_storage.load("layer_0.pt")
```

### Client-Side Recovery

1. Cache Cleanup
```python
# Automatic cache cleanup on errors
async with StreamingBertModel() as model:
    try:
        outputs = await model.forward_async(inputs)
    except Exception:
        # Cache is automatically cleared in __aexit__
        pass
```

2. Component Reloading
```python
async def inference_with_recovery(model, inputs, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return await model.forward_async(inputs)
        except Exception:
            model.clear_cache()  # Clear potentially corrupted cache
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(1)
```

## Best Practices

### 1. Use Async Context Managers

```python
async with StreamingBertModel() as model:
    # Resources are properly cleaned up even if errors occur
    outputs = await model.forward_async(inputs)
```

### 2. Implement Health Checks

```python
async def check_server_health(uri="ws://localhost:8765"):
    try:
        async with websockets.connect(uri, timeout=5):
            return True
    except Exception:
        return False

# Check before starting inference
if not await check_server_health():
    raise RuntimeError("Weight server is not healthy")
```

### 3. Monitor Error Rates

```python
from streaming_weights import StreamingMonitor

monitor = StreamingMonitor()
model = StreamingBertModel(monitor=monitor)

# After running inference
stats = monitor.get_stats()
error_rate = stats["errors"] / stats["total_requests"]
if error_rate > 0.1:  # 10% error rate threshold
    alert_admin("High error rate detected")
```

### 4. Handle Specific Error Types

```python
from streaming_weights.exceptions import (
    WeightServerError,
    StorageError,
    CacheError,
    ModelError
)

try:
    model = StreamingBertModel()
    outputs = await model.forward_async(inputs)
except WeightServerError as e:
    # Handle server-specific errors
    logger.error(f"Weight server error: {e}")
except StorageError as e:
    # Handle storage backend errors
    logger.error(f"Storage error: {e}")
except CacheError as e:
    # Handle cache-related errors
    logger.error(f"Cache error: {e}")
    model.clear_cache()
except ModelError as e:
    # Handle model-specific errors
    logger.error(f"Model error: {e}")
```

## Security Considerations

1. Validate Model Chunks
```python
from streaming_weights.utils import calculate_chunk_hash

# Verify chunk integrity
chunk_hash = calculate_chunk_hash("./chunks/bert-tiny/layer_0.pt")
if chunk_hash != expected_hash:
    raise SecurityError("Chunk integrity check failed")
```

2. Handle Timeouts
```python
# Configure timeouts for security
model = StreamingBertModel(
    websocket_uri="ws://localhost:8765",
    timeout_seconds=30  # Prevent hanging on malicious servers
)
```

## Next Steps

- Learn about [Model Support](model-support.md)
- Explore [Example Use Cases](../examples/basic-usage.md)
- Review [Configuration Options](configuration.md)