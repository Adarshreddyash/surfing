# streaming_weights/chunker.py
import torch
import json
import argparse
import io
import asyncio
from pathlib import Path
from transformers import AutoModel, AutoConfig, BertModel, GPT2Model, T5Model, LlamaModel
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
        config = AutoConfig.from_pretrained(self.model_name)
        model_type = config.model_type.lower()
        
        # Create chunk info dictionary
        chunk_info = {
            "model_name": self.model_name,
            "model_type": model_type,
            "chunks": {},
            "total_size_mb": 0,
        }
        
        # Determine chunking strategy based on model type
        if model_type == "bert":
            await self._chunk_bert_model(chunk_info)
        elif model_type in ["gpt2", "gpt_neo", "gptj", "gpt_neox"]:
            await self._chunk_gpt_model(chunk_info)
        elif model_type in ["t5", "mt5", "bart"]:
            await self._chunk_encoder_decoder_model(chunk_info)
        elif model_type in ["llama"]:
            await self._chunk_llama_model(chunk_info)
        else:
            # Generic chunking for unknown model types
            await self._chunk_generic_model(chunk_info)
            
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
        
    async def _chunk_bert_model(self, chunk_info: Dict[str, Any]) -> None:
        """Chunk a BERT-style model into separate files"""
        self.logger.info("Chunking BERT-style model")
        
        # Load model
        model = BertModel.from_pretrained(self.model_name)
        config = model.config
        
        # Add num_layers to chunk_info
        chunk_info["num_layers"] = config.num_hidden_layers
        
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
            
    async def _chunk_gpt_model(self, chunk_info: Dict[str, Any]) -> None:
        """Chunk a GPT-style model into separate files"""
        self.logger.info("Chunking GPT-style model")
        
        # Load model
        model = GPT2Model.from_pretrained(self.model_name)
        config = model.config
        
        # Add num_layers to chunk_info
        chunk_info["num_layers"] = config.n_layer
        
        # Save token embeddings
        if hasattr(model, "wte"):
            filename = "wte.pt"
            await self._save_component(model.wte.state_dict(), filename)
            chunk_info["chunks"]["wte"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }
            
        # Save position embeddings
        if hasattr(model, "wpe"):
            filename = "wpe.pt"
            await self._save_component(model.wpe.state_dict(), filename)
            chunk_info["chunks"]["wpe"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }

        # Save transformer blocks
        if hasattr(model, "h"):
            for i, block in enumerate(model.h):
                filename = f"block_{i}.pt"
                await self._save_component(block.state_dict(), filename)
                chunk_info["chunks"][f"block_{i}"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }

        # Save final layer norm
        if hasattr(model, "ln_f"):
            filename = "ln_f.pt"
            await self._save_component(model.ln_f.state_dict(), filename)
            chunk_info["chunks"]["ln_f"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }
            
    async def _chunk_llama_model(self, chunk_info: Dict[str, Any]) -> None:
        """Chunk a LLaMA model into separate files"""
        self.logger.info("Chunking LLaMA model")
        
        # Load model
        model = LlamaModel.from_pretrained(self.model_name)
        config = model.config
        
        # Add num_layers to chunk_info
        num_layers = len(config.hidden_sizes) if hasattr(config, 'hidden_sizes') else config.num_hidden_layers
        chunk_info["num_layers"] = num_layers
        
        # Save token embeddings
        if hasattr(model, "embed_tokens"):
            filename = "embed_tokens.pt"
            await self._save_component(model.embed_tokens.state_dict(), filename)
            chunk_info["chunks"]["embed_tokens"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }
            
        # Save layers
        if hasattr(model, "layers"):
            for i, layer in enumerate(model.layers):
                filename = f"layer_{i}.pt"
                await self._save_component(layer.state_dict(), filename)
                chunk_info["chunks"][f"layer_{i}"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }

        # Save final normalization layer
        if hasattr(model, "norm"):
            filename = "norm.pt"
            await self._save_component(model.norm.state_dict(), filename)
            chunk_info["chunks"]["norm"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }
            
    async def _chunk_encoder_decoder_model(self, chunk_info: Dict[str, Any]) -> None:
        """Chunk an encoder-decoder model (T5, BART) into separate files"""
        self.logger.info("Chunking encoder-decoder model")
        
        # Load model
        model = T5Model.from_pretrained(self.model_name)
        config = model.config
        
        # Add num_layers to chunk_info
        chunk_info["num_layers"] = config.num_layers
        chunk_info["num_decoder_layers"] = config.num_decoder_layers
        
        # Save shared embeddings if they exist
        if hasattr(model, "shared"):
            filename = "shared_embeddings.pt"
            await self._save_component(model.shared.state_dict(), filename)
            chunk_info["chunks"]["shared_embeddings"] = {
                "file": filename,
                "size_mb": self._get_file_size(filename),
            }
            
        # Save encoder components
        if hasattr(model, "encoder"):
            # Save encoder embeddings if separate from shared
            if hasattr(model.encoder, "embed_tokens") and not model.encoder.embed_tokens == model.shared:
                filename = "encoder_embeddings.pt"
                await self._save_component(model.encoder.embed_tokens.state_dict(), filename)
                chunk_info["chunks"]["encoder_embeddings"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }
                
            # Save encoder blocks
            if hasattr(model.encoder, "block"):
                for i, block in enumerate(model.encoder.block):
                    filename = f"encoder_block_{i}.pt"
                    await self._save_component(block.state_dict(), filename)
                    chunk_info["chunks"][f"encoder_block_{i}"] = {
                        "file": filename,
                        "size_mb": self._get_file_size(filename),
                    }
                    
            # Save encoder final layer norm
            if hasattr(model.encoder, "final_layer_norm"):
                filename = "encoder_final_layer_norm.pt"
                await self._save_component(model.encoder.final_layer_norm.state_dict(), filename)
                chunk_info["chunks"]["encoder_final_layer_norm"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }
                
        # Save decoder components
        if hasattr(model, "decoder"):
            # Save decoder embeddings if separate from shared
            if hasattr(model.decoder, "embed_tokens") and not model.decoder.embed_tokens == model.shared:
                filename = "decoder_embeddings.pt"
                await self._save_component(model.decoder.embed_tokens.state_dict(), filename)
                chunk_info["chunks"]["decoder_embeddings"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }
                
            # Save decoder blocks
            if hasattr(model.decoder, "block"):
                for i, block in enumerate(model.decoder.block):
                    filename = f"decoder_block_{i}.pt"
                    await self._save_component(block.state_dict(), filename)
                    chunk_info["chunks"][f"decoder_block_{i}"] = {
                        "file": filename,
                        "size_mb": self._get_file_size(filename),
                    }
                    
            # Save decoder final layer norm
            if hasattr(model.decoder, "final_layer_norm"):
                filename = "decoder_final_layer_norm.pt"
                await self._save_component(model.decoder.final_layer_norm.state_dict(), filename)
                chunk_info["chunks"]["decoder_final_layer_norm"] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }
                
    async def _chunk_generic_model(self, chunk_info: Dict[str, Any]) -> None:
        """Chunk a generic transformer model by attempting to identify components"""
        self.logger.info("Chunking generic model - attempting to identify components")
        
        # Load model
        model = AutoModel.from_pretrained(self.model_name)
        config = model.config
        
        # Try to determine number of layers
        num_layers = 0
        for attr_name in ["num_hidden_layers", "n_layer", "num_layers", "encoder_layers"]:
            if hasattr(config, attr_name):
                num_layers = getattr(config, attr_name)
                chunk_info["num_layers"] = num_layers
                break
                
        # Save all top-level modules separately
        for name, module in model.named_children():
            # Skip certain modules that might be too large or not needed
            if name in ["device_map", "_no_split_modules"]:
                continue
                
            # Check if this is a list/sequence of layers
            if hasattr(module, "__len__") and not isinstance(module, torch.nn.Embedding):
                try:
                    # Try to iterate through the module
                    for i, layer in enumerate(module):
                        filename = f"{name}_{i}.pt"
                        await self._save_component(layer.state_dict(), filename)
                        chunk_info["chunks"][f"{name}_{i}"] = {
                            "file": filename,
                            "size_mb": self._get_file_size(filename),
                        }
                except (TypeError, ValueError):
                    # Not iterable, save as a single component
                    filename = f"{name}.pt"
                    await self._save_component(module.state_dict(), filename)
                    chunk_info["chunks"][name] = {
                        "file": filename,
                        "size_mb": self._get_file_size(filename),
                    }
            else:
                # Save as a single component
                filename = f"{name}.pt"
                await self._save_component(module.state_dict(), filename)
                chunk_info["chunks"][name] = {
                    "file": filename,
                    "size_mb": self._get_file_size(filename),
                }



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
