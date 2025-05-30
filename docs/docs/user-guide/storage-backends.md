# Storage Backends

Surfing Weights provides a flexible storage backend system for storing and retrieving model chunks. The storage backend interface is designed to be extensible, allowing you to implement custom backends for different storage solutions.

## Available Backends

### FileSystem Backend

The filesystem backend stores model chunks as files in a local directory. This is the simplest backend and is ideal for local development and testing.

```python
from streaming_weights import WeightServer, FilesystemBackend

# Initialize the backend
storage = FilesystemBackend(base_dir="./chunks/bert-tiny")

# Use with weight server
server = WeightServer(storage_backend=storage)
```

Key features:
- Direct file system access
- No additional dependencies
- Fastest for local development
- Automatic directory creation
- Asynchronous I/O for large files

### S3 Backend

The S3 backend stores model chunks in Amazon S3 or S3-compatible storage. This is ideal for production deployments and distributed systems.

#### Basic Usage

```python
from streaming_weights import WeightServer, S3Backend

# Initialize with minimal configuration
storage = S3Backend(
    bucket_name="model-weights",
    prefix="models/bert-tiny",  # Optional: folder within bucket
    region_name="us-east-1"    # Optional: AWS region
)

# Use with weight server
server = WeightServer(storage_backend=storage)
```

#### Authentication Options

The S3 backend supports multiple authentication methods:

1. Direct credentials:
```python
storage = S3Backend(
    bucket_name="model-weights",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    aws_session_token="YOUR_SESSION_TOKEN"  # Optional: for temporary credentials
)
```

2. AWS profile:
```python
storage = S3Backend(
    bucket_name="model-weights",
    profile_name="default"  # Use specific AWS credentials profile
)
```

3. Environment variables:
```bash
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
export AWS_SESSION_TOKEN=YOUR_SESSION_TOKEN  # Optional
export AWS_REGION=us-east-1                 # Optional
```

4. Instance profile or ECS task role (when running on AWS)

#### S3-Compatible Storage

Support for S3-compatible storage services (MinIO, DigitalOcean Spaces, etc.):

```python
storage = S3Backend(
    bucket_name="model-weights",
    endpoint_url="https://custom-endpoint",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)
```

## Common Operations

All storage backends implement the following interface:

### Loading Data

```python
# Check if a chunk exists
exists = await storage.exists("layer_0.pt")

# Load a chunk
data = await storage.load("layer_0.pt")
```

### Saving Data

```python
# Save bytes
await storage.save("layer_0.pt", chunk_data)

# Save from a file-like object
with open("layer_0.pt", "rb") as f:
    await storage.save("layer_0.pt", f)
```

### Listing Contents

```python
# List all chunks
chunks = await storage.list()

# List chunks with prefix
embeddings = await storage.list("embeddings_")
```

## Creating Custom Backends

You can create custom storage backends by implementing the `StorageBackend` abstract base class:

```python
from streaming_weights.storage import StorageBackend
from typing import Union, BinaryIO, List

class CustomBackend(StorageBackend):
    async def load(self, key: str) -> bytes:
        """Load data from storage"""
        pass

    async def save(self, key: str, data: Union[bytes, BinaryIO]) -> None:
        """Save data to storage"""
        pass

    async def exists(self, key: str) -> bool:
        """Check if data exists"""
        pass

    async def list(self, prefix: str = "") -> List[str]:
        """List all keys with given prefix"""
        pass
```

## Best Practices

1. Error Handling
   - Handle storage-specific errors gracefully
   - Provide meaningful error messages
   - Implement proper retries for transient failures

2. Performance
   - Use appropriate chunk sizes
   - Enable compression when beneficial
   - Implement caching for frequently accessed chunks

3. Security
   - Use secure credentials management
   - Implement proper access controls
   - Enable encryption at rest when needed

## Next Steps

- Learn about the [Caching System](caching.md)
- Explore [Error Handling](error-handling.md)
- See [Example Use Cases](../examples/basic-usage.md)