# streaming_weights/weight_server.py
import asyncio
import websockets
import torch
import json
import io
from pathlib import Path
import logging
from typing import Optional

from .storage import StorageBackend, FilesystemBackend

class WeightServer:
    def __init__(self, storage_backend: Optional[StorageBackend] = None, chunks_dir: Optional[str] = None, port: int = 8000, cache_size_mb: int = 100):
        """
        Initialize the weight server.
        
        Args:
            storage_backend: Storage backend to use. If None, a FilesystemBackend will be created using chunks_dir.
            chunks_dir: Directory containing model chunks. Only used if storage_backend is None.
            port: Port to run the server on.
            cache_size_mb: Maximum size of the in-memory cache in MB.
        """
        self.port = port
        self.cache = {}  # In-memory cache for frequently accessed weights
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.current_cache_size = 0
        self.logger = logging.getLogger(__name__)
        
        # Set up storage backend
        if storage_backend is not None:
            self.storage = storage_backend
        elif chunks_dir is not None:
            chunks_path = Path(chunks_dir)
            if not chunks_path.exists():
                raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")
            self.storage = FilesystemBackend(chunks_path)
        else:
            raise ValueError("Either storage_backend or chunks_dir must be provided")

    async def handle_client(self, websocket, path=""):
        try:
            async for message in websocket:
                response = await self.process_request(message)
                await websocket.send(response)
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Client disconnected")
        except Exception as e:
            self.logger.error(f"Error handling client: {e}")

    async def process_request(self, message):
        try:
            # Parse request: "GET LAYER 0" or "GET embeddings"
            parts = message.strip().split()
            if len(parts) != 3 or parts[0] != "GET":
                return json.dumps({"success": False, "error": "Invalid request format"})

            component_type = parts[1].lower()
            component_id = parts[2]

            # Map component type to filename
            # BERT components
            if component_type == "layer":
                filename = f"layer_{component_id}.pt"
            elif component_type == "embeddings":
                filename = "embeddings.pt"
            elif component_type == "pooler":
                filename = "pooler.pt"
            # GPT components
            elif component_type == "block":
                filename = f"block_{component_id}.pt"
            elif component_type == "wte":
                filename = "wte.pt"
            elif component_type == "wpe":
                filename = "wpe.pt"
            elif component_type == "ln_f":
                filename = "ln_f.pt"
            # T5/Encoder-Decoder components
            elif component_type == "encoder_block":
                filename = f"encoder_block_{component_id}.pt"
            elif component_type == "decoder_block":
                filename = f"decoder_block_{component_id}.pt"
            elif component_type == "shared_embeddings":
                filename = "shared_embeddings.pt"
            elif component_type == "encoder_embeddings":
                filename = "encoder_embeddings.pt"
            elif component_type == "decoder_embeddings":
                filename = "decoder_embeddings.pt"
            elif component_type == "encoder_final_layer_norm":
                filename = "encoder_final_layer_norm.pt"
            elif component_type == "decoder_final_layer_norm":
                filename = "decoder_final_layer_norm.pt"
            # LLaMA components
            elif component_type == "embed_tokens":
                filename = "embed_tokens.pt"
            elif component_type == "norm":
                filename = "norm.pt"
            # Generic components (for unknown model types)
            elif "_" in component_type:
                # Handle components like "h_0", "encoder_0", etc.
                filename = f"{component_type}_{component_id}.pt"
            else:
                # Handle single components like "embeddings", "lm_head", etc.
                filename = f"{component_type}.pt"

            # Check cache first
            cache_key = f"{component_type}_{component_id}"
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for {cache_key}")
                weights_bytes = self.cache[cache_key]
            else:
                # Check if file exists in storage
                if not await self.storage.exists(filename):
                    return json.dumps({"success": False, "error": f"File not found: {filename}"})

                self.logger.info(f"Loading {filename} from storage")
                # Load weights from storage
                weights_bytes = await self.storage.load(filename)
                
                # Deserialize, then reserialize to ensure consistent format
                buffer = io.BytesIO(weights_bytes)
                state_dict = torch.load(buffer, map_location="cpu")
                
                # Reserialize
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                weights_bytes = buffer.getvalue()

                # Update cache with LRU policy
                await self._update_cache(cache_key, weights_bytes)

            # Return serialized weights
            return json.dumps(
                {
                    "success": True,
                    "component": f"{component_type}_{component_id}",
                    "weights": weights_bytes.hex(),  # Hex encode for JSON transport
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return json.dumps({"success": False, "error": str(e)})

    async def start_server(self):
        # Print a nice startup banner
        print("\n" + "=" * 60)
        print("🌊 Surfing Weight Server v1.0")
        print(f"🚀 Starting server on port {self.port}")
        
        # Print storage backend info
        if isinstance(self.storage, FilesystemBackend):
            print(f"📂 Storage: Local Filesystem ({self.storage.base_dir})")
        else:
            # For S3 or other backends
            storage_type = self.storage.__class__.__name__
            print(f"☁️  Storage: {storage_type}")
            
            # If it's S3, show bucket info
            if hasattr(self.storage, 'bucket_name'):
                bucket_info = f"{self.storage.bucket_name}"
                if hasattr(self.storage, 'prefix') and self.storage.prefix:
                    bucket_info += f"/{self.storage.prefix}"
                print(f"🪣 S3 Bucket: {bucket_info}")
        
        print(f"💾 Cache Size: {self.cache_size_mb} MB")
        print("📡 Ready to serve model weights!")
        print("=" * 60 + "\n")
        
        self.logger.info(f"Starting weight server on port {self.port}")
        async def handler(websocket):
            await self.handle_client(websocket, "")
        # Increase message size limit to 10MB
        async with websockets.serve(
            handler, 
            "localhost", 
            self.port,
            max_size=10_485_760,  # 10MB
            compression=None  # Disable compression for large messages
        ):
            await asyncio.Future()  # Run forever


# CLI entry point
    async def _update_cache(self, key: str, data: bytes) -> None:
        """Update the cache with the given key and data using LRU policy."""
        # If data is too large for the cache, don't cache it
        if len(data) > self.cache_size_bytes:
            self.logger.warning(f"Data for {key} is too large for cache ({len(data)} bytes)")
            return
            
        # If adding this would exceed the cache size, remove items until it fits
        data_size = len(data)
        while self.current_cache_size + data_size > self.cache_size_bytes and self.cache:
            # Remove the least recently used item (first item in the cache)
            lru_key = next(iter(self.cache))
            lru_data = self.cache.pop(lru_key)
            self.current_cache_size -= len(lru_data)
            self.logger.debug(f"Removed {lru_key} from cache (size: {len(lru_data)} bytes)")
            
        # Add the new item to the cache
        self.cache[key] = data
        self.current_cache_size += data_size
        self.logger.debug(f"Added {key} to cache (size: {data_size} bytes)")
        
        # Move the key to the end of the cache (most recently used)
        # This is done by popping and re-adding
        self.cache[key] = self.cache.pop(key)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start a weight server for streaming model weights")
    parser.add_argument("--chunks-dir", "-d", help="Directory containing model chunks")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port to run server on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--cache-size", "-c", type=int, default=100, help="Maximum cache size in MB")
    
    # S3 options
    parser.add_argument("--s3", action="store_true", help="Use S3 storage backend")
    parser.add_argument("--s3-bucket", help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="", help="S3 key prefix (folder in bucket)")
    parser.add_argument("--s3-region", help="AWS region name (e.g., 'us-east-1')")
    parser.add_argument("--s3-access-key", help="AWS access key ID")
    parser.add_argument("--s3-secret-key", help="AWS secret access key")
    parser.add_argument("--s3-session-token", help="AWS session token (for temporary credentials)")
    parser.add_argument("--s3-profile", help="AWS profile name to use from credentials file")
    parser.add_argument("--s3-endpoint", help="Custom endpoint URL (for S3-compatible storage)")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create storage backend
    if args.s3:
        # Import here to avoid dependency if not using S3
        from .storage import S3Backend
        
        if not args.s3_bucket:
            parser.error("--s3-bucket is required when using --s3")
            
        storage = S3Backend(
            bucket_name=args.s3_bucket,
            prefix=args.s3_prefix,
            region_name=args.s3_region,
            aws_access_key_id=args.s3_access_key,
            aws_secret_access_key=args.s3_secret_key,
            aws_session_token=args.s3_session_token,
            profile_name=args.s3_profile,
            endpoint_url=args.s3_endpoint
        )
    elif args.chunks_dir:
        # Use filesystem backend
        storage = None  # WeightServer will create a FilesystemBackend
    else:
        parser.error("Either --chunks-dir or --s3 with --s3-bucket must be provided")
    
    # Start server
    server = WeightServer(
        storage_backend=storage,
        chunks_dir=args.chunks_dir,
        port=args.port,
        cache_size_mb=args.cache_size
    )
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()
