# Configuration Guide

This guide covers all configuration options available in Surfing Weights, from basic settings to advanced features.

## Server Configuration

### Basic Server Settings

```python
from streaming_weights import WeightServer

server = WeightServer(
    model_path="./chunks/bert-tiny",  # Path to chunked model
    host="localhost",                  # Server hostname
    port=8765,                        # Server port
    use_ssl=False                     # Enable/disable SSL
)
```

### Cache Configuration

```python
server = WeightServer(
    model_path="./chunks/bert-tiny",
    cache_size="2GB",              # Server-side cache size
    enable_compression=True        # Enable weight compression
)
```

### Command Line Options

When using the CLI:

```bash
# Basic usage
streaming-weights-server --chunks-dir ./chunks/bert-tiny --port 8765

# With cache and logging options
streaming-weights-server --chunks-dir ./chunks/bert-tiny --port 8765 \
    --cache-size 200 --verbose
```

## Client Configuration

### Basic Client Settings

```python
from streaming_weights import StreamingBertModel

model = StreamingBertModel(
    model_name="prajjwal1/bert-tiny",
    server_host="localhost",
    server_port=8765,
    use_ssl=False
)
```

### Performance Settings

```python
model = StreamingBertModel(
    model_name="prajjwal1/bert-tiny",
    cache_size=3,           # Number of layers to cache
    prefetch_layers=True,   # Enable layer prefetching
    prefetch_count=1,       # Number of layers to prefetch
    timeout_seconds=30      # Request timeout
)
```

## Storage Backend Configuration

### Local Filesystem

```python
from streaming_weights import WeightServer, FilesystemBackend

storage = FilesystemBackend(base_path="./chunks/bert-tiny")
server = WeightServer(storage_backend=storage)
```

### Amazon S3

```python
from streaming_weights import WeightServer, S3Backend

# Option 1: Direct initialization
server = WeightServer(
    s3_bucket="model-weights",
    s3_prefix="models/bert-tiny",
    s3_region="us-east-1"
)

# Option 2: Custom backend configuration
storage = S3Backend(
    bucket_name="model-weights",
    prefix="models/bert-tiny",
    region_name="us-east-1",
    aws_access_key_id="YOUR_ACCESS_KEY",      # Optional
    aws_secret_access_key="YOUR_SECRET_KEY",  # Optional
    aws_session_token="YOUR_SESSION_TOKEN",   # Optional
    profile_name="default",                   # Optional
    endpoint_url="https://custom-endpoint"    # Optional
)
server = WeightServer(storage_backend=storage)
```

### S3 Command Line Options

```bash
# Basic S3 usage
streaming-weights-server --s3 --s3-bucket model-weights \
    --s3-prefix models/bert-tiny --port 8765

# With AWS credentials
streaming-weights-server --s3 --s3-bucket model-weights \
    --s3-access-key YOUR_ACCESS_KEY \
    --s3-secret-key YOUR_SECRET_KEY \
    --s3-region us-east-1
```

## Advanced Features

### Monitoring Configuration

Enable Prometheus metrics:

```python
server = WeightServer(
    model_path="./chunks/bert-tiny",
    enable_monitoring=True,
    metrics_port=9090
)
```

### SSL Configuration

Enable secure WebSocket connections:

```python
server = WeightServer(
    model_path="./chunks/bert-tiny",
    use_ssl=True,
    ssl_cert_path="path/to/cert.pem",
    ssl_key_path="path/to/key.pem"
)
```

### Environment Variables

Surfing Weights also supports configuration via environment variables:

- `SURFING_SERVER_HOST` - Server hostname
- `SURFING_SERVER_PORT` - Server port
- `SURFING_CACHE_SIZE` - Cache size in MB
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_SESSION_TOKEN` - AWS session token
- `AWS_PROFILE` - AWS profile name
- `AWS_REGION` - AWS region

## Configuration Reference

### Server Settings
| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| server_host | str | "localhost" | Server hostname |
| server_port | int | 8765 | Server port |
| use_ssl | bool | False | Enable SSL |
| cache_size | int/str | "100MB" | Cache size |
| enable_compression | bool | True | Enable compression |

### Client Settings
| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| cache_size | int | 3 | Layers to cache |
| prefetch_layers | bool | True | Enable prefetching |
| prefetch_count | int | 1 | Layers to prefetch |
| timeout_seconds | int | 30 | Request timeout |

## Next Steps

- Learn about [Storage Backends](storage-backends.md)
- Explore [Caching System](caching.md)
- Read about [Error Handling](error-handling.md)