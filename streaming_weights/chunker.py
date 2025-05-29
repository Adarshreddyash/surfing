# streaming_weights/chunker.py
import torch
import json
import argparse
import io
import asyncio
from pathlib import Path
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, Union
import logging

from .storage import StorageBackend, FilesystemBackend
try:
    from .storage import S3Backend
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


class ModelChunker:
    """Utility to chunk transformer models for streaming
    
    This class can save model chunks to either a local filesystem or to AWS S3.
    """

    def __init__(self, model_name: str, storage_backend: Optional[StorageBackend] = None, 
                 output_dir: Optional[str] = None, compress: bool = True,
                 s3_bucket: Optional[str] = None, s3_prefix: str = "", s3_region: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None,
                 aws_session_token: Optional[str] = None, profile_name: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        """Initialize the model chunker.
        
        Args:
            model_name: Name of the HuggingFace model to chunk
            storage_backend: Optional storage backend to use. If None, a FilesystemBackend will be created using output_dir.
            output_dir: Directory to save chunks to. Only used if storage_backend is None.
            compress: Whether to compress model weights (convert to half precision)
            s3_bucket: S3 bucket name. Only used if storage_backend is None and output_dir is None.
            s3_prefix: S3 key prefix (folder in bucket). Only used with s3_bucket.
            s3_region: AWS region name. Only used with s3_bucket.
            aws_access_key_id: AWS access key ID. Only used with s3_bucket.
            aws_secret_access_key: AWS secret access key. Only used with s3_bucket.
            aws_session_token: AWS session token (for temporary credentials). Only used with s3_bucket.
            profile_name: AWS profile name to use from credentials file. Only used with s3_bucket.
            endpoint_url: Custom endpoint URL (for S3-compatible storage). Only used with s3_bucket.
        """
        self.model_name = model_name
        self.compress = compress
        self.logger = logging.getLogger(__name__)
        self.file_sizes = {}  # Track file sizes for S3 storage
        
        # Set up storage backend
        if storage_backend is not None:
            self.storage = storage_backend
        elif s3_bucket is not None:
            if not S3_AVAILABLE:
                raise ImportError("boto3 is required for S3 storage. Install it with 'pip install boto3'.")
            self.storage = S3Backend(
                bucket_name=s3_bucket,
                prefix=s3_prefix,
                region_name=s3_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                profile_name=profile_name,
                endpoint_url=endpoint_url
            )
            self.output_dir = None  # Not using filesystem
        elif output_dir is not None:
            self.output_dir = Path(output_dir)
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.storage = FilesystemBackend(self.output_dir)
        else:
            raise ValueError("Either storage_backend, output_dir, or s3_bucket must be provided")

    async def chunk_model(self) -> Dict[str, Any]:
        """Chunk a transformer model into separate files"""
        self.logger.info(f"Loading model: {self.model_name}")

        # Load model and config
        model = AutoModel.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)

        chunk_info = {
            "model_name": self.model_name,
            "model_type": config.model_type,
            "num_layers": getattr(config, "num_hidden_layers", 0),
            "chunks": {},
            "total_size_mb": 0,
        }

        # Save embeddings
        if hasattr(model, "embeddings"):
            filename = "embeddings.pt"
            await self._save_component(model.embeddings.state_dict(), filename)
            chunk_info["chunks"]["embeddings"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }

        # Save encoder layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            for i, layer in enumerate(model.encoder.layer):
                filename = f"layer_{i}.pt"
                await self._save_component(layer.state_dict(), filename)
                chunk_info["chunks"][f"layer_{i}"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }

        # Save pooler (if exists)
        if hasattr(model, "pooler") and model.pooler is not None:
            filename = "pooler.pt"
            await self._save_component(model.pooler.state_dict(), filename)
            chunk_info["chunks"]["pooler"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }

        # Save config
        if self.output_dir is not None:
            # Save config directly to filesystem if using local storage
            config.save_pretrained(self.output_dir)
        else:
            # Save config to S3 if using S3 storage
            config_dict = config.to_dict()
            config_json = json.dumps(config_dict, indent=2)
            await self.storage.save("config.json", config_json.encode('utf-8'))

        # Calculate total size
        chunk_info["total_size_mb"] = sum(
            chunk["size_mb"] for chunk in chunk_info["chunks"].values()
        )

        # Save chunk info
        chunk_info_json = json.dumps(chunk_info, indent=2)
        await self.storage.save("chunk_info.json", chunk_info_json.encode('utf-8'))

        self.logger.info(
            f"Model chunked successfully. Total size: {chunk_info['total_size_mb']:.2f} MB"
        )
        
        # Log where the chunks were saved
        if isinstance(self.storage, FilesystemBackend):
            self.logger.info(f"Chunks saved to: {self.output_dir}")
        elif hasattr(self.storage, 'bucket_name'):
            bucket_info = f"{self.storage.bucket_name}"
            if hasattr(self.storage, 'prefix') and self.storage.prefix:
                bucket_info += f"/{self.storage.prefix}"
            self.logger.info(f"Chunks saved to S3 bucket: {bucket_info}")
        else:
            self.logger.info("Chunks saved to storage backend")

        return chunk_info

    async def _save_component(self, state_dict: Dict[str, torch.Tensor], filename: str):
        """Save a model component with optional compression
        
        Args:
            state_dict: The state dict to save
            filename: The filename to save to (without path)
        """
        # Apply compression if enabled
        if self.compress:
            # Apply basic compression - you can enhance this
            compressed_dict = {}
            for key, tensor in state_dict.items():
                if tensor.dtype == torch.float32:
                    # Convert to half precision for compression
                    compressed_dict[key] = tensor.half()
                else:
                    compressed_dict[key] = tensor
            save_dict = compressed_dict
        else:
            save_dict = state_dict
            
        # Serialize the state dict to a bytes buffer
        buffer = io.BytesIO()
        torch.save(save_dict, buffer)
        buffer.seek(0)
        
        # Save to storage backend
        data = buffer.getvalue()
        await self.storage.save(filename, data)
        
        # Track file size for S3 storage where we can't get file size directly
        self.file_sizes[filename] = len(data) / (1024 * 1024)  # Size in MB

    def _get_file_size(self, filename: str) -> float:
        """Get file size in MB
        
        Args:
            filename: The filename to get the size of (without path)
            
        Returns:
            The file size in MB
        """
        # If we're using S3 or another remote storage, use the tracked size
        if filename in self.file_sizes:
            return self.file_sizes[filename]
            
        # For filesystem storage, get the size from the file
        if self.output_dir is not None:
            file_path = self.output_dir / filename
            return file_path.stat().st_size / (1024 * 1024)
            
        # Default case if we don't have size info
        return 0.0

    @staticmethod
    async def load_chunk_info(storage: Union[StorageBackend, str]) -> Dict[str, Any]:
        """Load chunk information from storage
        
        Args:
            storage: Either a StorageBackend instance or a path to a directory
            
        Returns:
            The chunk info dictionary
        """
        # Handle string path (backward compatibility)
        if isinstance(storage, str):
            chunks_dir = Path(storage)
            info_path = chunks_dir / "chunk_info.json"
            if not info_path.exists():
                raise FileNotFoundError(f"Chunk info not found: {info_path}")

            with open(info_path, "r") as f:
                return json.load(f)
        else:
            # Use storage backend
            if not await storage.exists("chunk_info.json"):
                raise FileNotFoundError("Chunk info not found in storage")
                
            data = await storage.load("chunk_info.json")
            return json.loads(data.decode('utf-8'))


def main():
    """CLI entry point for chunking models"""
    parser = argparse.ArgumentParser(
        description="Chunk transformer models for streaming"
    )
    parser.add_argument("model_name", help="HuggingFace model name")
    
    # Storage options group
    storage_group = parser.add_mutually_exclusive_group(required=True)
    storage_group.add_argument(
        "--output-dir", "-o", help="Output directory for chunks (local filesystem)"
    )
    storage_group.add_argument(
        "--s3", action="store_true", help="Use S3 storage backend"
    )
    
    # S3 options
    parser.add_argument("--s3-bucket", help="S3 bucket name (required when using --s3)")
    parser.add_argument("--s3-prefix", default="", help="S3 key prefix (folder in bucket)")
    parser.add_argument("--s3-region", help="AWS region name (e.g., 'us-east-1')")
    parser.add_argument("--s3-access-key", help="AWS access key ID")
    parser.add_argument("--s3-secret-key", help="AWS secret access key")
    parser.add_argument("--s3-session-token", help="AWS session token (for temporary credentials)")
    parser.add_argument("--s3-profile", help="AWS profile name to use from credentials file")
    parser.add_argument("--s3-endpoint", help="Custom endpoint URL (for S3-compatible storage)")
    
    # Other options
    parser.add_argument("--compress", action="store_true", help="Enable compression")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level)
    
    # Validate arguments
    if args.s3 and not args.s3_bucket:
        parser.error("--s3-bucket is required when using --s3")

    # Create chunker based on storage type
    if args.s3:
        chunker = ModelChunker(
            model_name=args.model_name,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_region=args.s3_region,
            compress=args.compress,
            # AWS credentials
            aws_access_key_id=args.s3_access_key,
            aws_secret_access_key=args.s3_secret_key,
            aws_session_token=args.s3_session_token,
            profile_name=args.s3_profile,
            endpoint_url=args.s3_endpoint
        )
    else:
        chunker = ModelChunker(
            model_name=args.model_name,
            output_dir=args.output_dir,
            compress=args.compress
        )

    # Chunk model (run async)
    chunk_info = asyncio.run(chunker.chunk_model())

    print("\u2705 Model chunked successfully!")
    if args.s3:
        bucket_info = args.s3_bucket
        if args.s3_prefix:
            bucket_info += f"/{args.s3_prefix}"
        print(f"\u2601️ S3 bucket: {bucket_info}")
    else:
        print(f"\U0001F4C1 Output directory: {args.output_dir}")
    print(f"\U0001F4C8 Total chunks: {len(chunk_info['chunks'])}")
    print(f"\U0001F4B0 Total size: {chunk_info['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
