# 🌊 Surfing - Usage Guide

This guide provides detailed instructions for using the Surfing weights streaming system.

## 📦 Chunking Models

Before you can stream a model, you need to chunk it into smaller pieces. This process only needs to be done once per model. You can save chunks to either the local filesystem or to AWS S3.

### Command Line Interface

#### Local Filesystem Storage

```bash
# Basic usage with local filesystem
python -m streaming_weights.chunker <model_name> --output-dir <output_directory>

# Example with compression enabled
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --compress

# With verbose logging
python -m streaming_weights.chunker prajjwal1/bert-tiny --output-dir ./chunks/bert-tiny --verbose
```

#### AWS S3 Storage

```bash
# Basic usage with S3 storage
python -m streaming_weights.chunker <model_name> --s3 --s3-bucket <bucket_name>

# Example with S3 prefix (folder in bucket)
python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights --s3-prefix bert-models/tiny

# With specific AWS region and compression
python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights --s3-region us-east-1 --compress

# Complete example with all options
python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights --s3-prefix models/bert-tiny --s3-region us-west-2 --compress --verbose
```

> **Note:** When using S3 storage, you need to have AWS credentials configured on your system. You can set them using environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`), AWS credentials file, or IAM roles if running on AWS infrastructure.

### Python API

#### Local Filesystem Storage

```python
import asyncio
from streaming_weights import ModelChunker

async def chunk_model():
    # Initialize chunker with local filesystem storage
    chunker = ModelChunker(
        model_name="prajjwal1/bert-tiny",
        output_dir="./chunks/bert-tiny",
        compress=True  # Optional: Enable compression
    )
    
    # Chunk the model (async operation)
    chunk_info = await chunker.chunk_model()
    print(f"Model chunked into {len(chunk_info['chunks'])} pieces")
    print(f"Total size: {chunk_info['total_size_mb']:.2f} MB")
    return chunk_info

# Run the chunking process
chunk_info = asyncio.run(chunk_model())
```

#### AWS S3 Storage

```python
import asyncio
from streaming_weights import ModelChunker, S3Backend

async def chunk_model():
    # Option 1: Initialize chunker with S3 parameters directly
    chunker = ModelChunker(
        model_name="prajjwal1/bert-tiny",
        s3_bucket="model-weights",
        s3_prefix="models/bert-tiny",  # Optional: folder in bucket
        s3_region="us-east-1",       # Optional: AWS region
        compress=True                 # Optional: Enable compression
    )
    
    # Option 2: Create S3 backend manually and pass it to chunker
    # s3_storage = S3Backend(
    #     bucket_name="model-weights",
    #     prefix="models/bert-tiny",
    #     region_name="us-east-1"
    # )
    # chunker = ModelChunker(
    #     model_name="prajjwal1/bert-tiny",
    #     storage_backend=s3_storage,
    #     compress=True
    # )
    
    # Chunk the model (async operation)
    chunk_info = await chunker.chunk_model()
    print(f"Model chunked into {len(chunk_info['chunks'])} pieces")
    print(f"Total size: {chunk_info['total_size_mb']:.2f} MB")
    return chunk_info

# Run the chunking process
chunk_info = asyncio.run(chunk_model())
```

### AWS Credentials for S3 Storage

When using S3 storage, you need to provide AWS credentials. There are several ways to do this:

1. **Command-line arguments** (most explicit):
   ```bash
   # For chunking
   python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights \
       --s3-access-key YOUR_ACCESS_KEY --s3-secret-key YOUR_SECRET_KEY
   
   # For serving
   python -m streaming_weights.weight_server --s3 --s3-bucket model-weights \
       --s3-access-key YOUR_ACCESS_KEY --s3-secret-key YOUR_SECRET_KEY
   ```

2. **Environment variables**:
   ```bash
   # Set environment variables
   export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
   export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
   
   # Then run commands without credential arguments
   python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights
   ```

3. **AWS credentials file** (`~/.aws/credentials`):
   ```ini
   # In ~/.aws/credentials
   [default]
   aws_access_key_id = YOUR_ACCESS_KEY
   aws_secret_access_key = YOUR_SECRET_KEY
   
   [profile_name]
   aws_access_key_id = ANOTHER_ACCESS_KEY
   aws_secret_access_key = ANOTHER_SECRET_KEY
   ```
   
   Then use the default profile:
   ```bash
   python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights
   ```
   
   Or specify a named profile:
   ```bash
   python -m streaming_weights.chunker prajjwal1/bert-tiny --s3 --s3-bucket model-weights \
       --s3-profile profile_name
   ```

4. **IAM roles** (when running on AWS EC2, ECS, or Lambda):
   If your code is running on AWS infrastructure with an IAM role attached, credentials will be automatically retrieved from the instance metadata service.

> **Security Note:** Avoid hardcoding AWS credentials in your code. Use one of the methods above instead.

### Chunk Output Structure

After chunking, your output directory will contain:

- `embeddings.pt`: The model's embedding layer
- `layer_0.pt`, `layer_1.pt`, etc.: Individual transformer layers
- `pooler.pt`: The model's pooler layer (if applicable)
- `config.json`: The model configuration
- `chunk_info.json`: Metadata about the chunked model

## 🚀 Starting a Weight Server

The weight server provides model chunks on demand via WebSocket. You can use either local filesystem storage or AWS S3 for storing model weights.

### Command Line Interface

#### Local Filesystem Storage

```bash
# Basic usage with local filesystem
python -m streaming_weights.weight_server --chunks-dir <chunks_directory> --port <port_number>

# Example
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --port 8765

# With verbose logging
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --port 8765 --verbose

# With custom cache size (in MB)
python -m streaming_weights.weight_server --chunks-dir ./chunks/bert-tiny --cache-size 200
```

#### AWS S3 Storage

```bash
# Basic usage with S3 storage
python -m streaming_weights.weight_server --s3 --s3-bucket <bucket_name> --port <port_number>

# Example with S3 prefix (folder in bucket)
python -m streaming_weights.weight_server --s3 --s3-bucket model-weights --s3-prefix bert-models/tiny --port 8765

# With specific AWS region
python -m streaming_weights.weight_server --s3 --s3-bucket model-weights --s3-region us-east-1 --port 8765

# Complete example with all options
python -m streaming_weights.weight_server --s3 --s3-bucket model-weights --s3-prefix models/bert-tiny --s3-region us-west-2 --port 8765 --cache-size 200 --verbose
```

> **Note:** When using S3 storage, you need to have AWS credentials configured on your system. You can set them using environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`), AWS credentials file, or IAM roles if running on AWS infrastructure.

### Python API

#### Local Filesystem Storage

```python
import asyncio
from streaming_weights import WeightServer

async def start_server():
    # Initialize server with local filesystem storage
    server = WeightServer(
        chunks_dir="./chunks/bert-tiny",
        port=8765,
        cache_size_mb=100  # Optional: Set cache size in MB
    )
    
    # Start server (runs indefinitely)
    await server.start_server()

# Run the server
asyncio.run(start_server())
```

#### AWS S3 Storage

```python
import asyncio
from streaming_weights import WeightServer, S3Backend

async def start_server():
    # Create S3 storage backend
    storage = S3Backend(
        bucket_name="model-weights",
        prefix="models/bert-tiny",  # Optional: folder in bucket
        region_name="us-east-1"     # Optional: AWS region
        # AWS credentials are loaded from environment variables or config files
    )
    
    # Initialize server with S3 storage
    server = WeightServer(
        storage_backend=storage,
        port=8765,
        cache_size_mb=100  # Optional: Set cache size in MB
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
