# streaming_weights/chunker.py
import torch
import json
import argparse
from pathlib import Path
from transformers import AutoModel, AutoConfig
from typing import Dict, Any
import logging


class ModelChunker:
    """Utility to chunk transformer models for streaming"""

    def __init__(self, model_name: str, output_dir: str, compress: bool = True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.compress = compress
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_model(self) -> Dict[str, Any]:
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
            embeddings_path = self.output_dir / "embeddings.pt"
            self._save_component(model.embeddings.state_dict(), embeddings_path)
            chunk_info["chunks"]["embeddings"] = {
                "file": "embeddings.pt",
                "size_mb": self._get_file_size_mb(embeddings_path),
            }

        # Save encoder layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            for i, layer in enumerate(model.encoder.layer):
                layer_path = self.output_dir / f"layer_{i}.pt"
                self._save_component(layer.state_dict(), layer_path)
                chunk_info["chunks"][f"layer_{i}"] = {
                    "file": f"layer_{i}.pt",
                    "size_mb": self._get_file_size_mb(layer_path),
                }

        # Save pooler (if exists)
        if hasattr(model, "pooler") and model.pooler is not None:
            pooler_path = self.output_dir / "pooler.pt"
            self._save_component(model.pooler.state_dict(), pooler_path)
            chunk_info["chunks"]["pooler"] = {
                "file": "pooler.pt",
                "size_mb": self._get_file_size_mb(pooler_path),
            }

        # Save config
        config.save_pretrained(self.output_dir)

        # Calculate total size
        chunk_info["total_size_mb"] = sum(
            chunk["size_mb"] for chunk in chunk_info["chunks"].values()
        )

        # Save chunk info
        info_path = self.output_dir / "chunk_info.json"
        with open(info_path, "w") as f:
            json.dump(chunk_info, f, indent=2)

        self.logger.info(
            f"Model chunked successfully. Total size: {chunk_info['total_size_mb']:.2f} MB"
        )
        self.logger.info(f"Chunks saved to: {self.output_dir}")

        return chunk_info

    def _save_component(self, state_dict: Dict[str, torch.Tensor], file_path: Path):
        """Save a model component with optional compression"""
        if self.compress:
            # Apply basic compression - you can enhance this
            compressed_dict = {}
            for key, tensor in state_dict.items():
                if tensor.dtype == torch.float32:
                    # Convert to half precision for compression
                    compressed_dict[key] = tensor.half()
                else:
                    compressed_dict[key] = tensor
            torch.save(compressed_dict, file_path)
        else:
            torch.save(state_dict, file_path)

    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        return file_path.stat().st_size / (1024 * 1024)

    @staticmethod
    def load_chunk_info(chunks_dir: str) -> Dict[str, Any]:
        """Load chunk information from directory"""
        info_path = Path(chunks_dir) / "chunk_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Chunk info not found: {info_path}")

        with open(info_path, "r") as f:
            return json.load(f)


def main():
    """CLI entry point for chunking models"""
    parser = argparse.ArgumentParser(
        description="Chunk transformer models for streaming"
    )
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Output directory for chunks"
    )
    parser.add_argument("--compress", action="store_true", help="Enable compression")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level)

    # Chunk model
    chunker = ModelChunker(args.model_name, args.output_dir, args.compress)
    chunk_info = chunker.chunk_model()

    print("\u2705 Model chunked successfully!")
    print(f"\U0001F4C1 Output directory: {args.output_dir}")
    print(f"\U0001F4C8 Total chunks: {len(chunk_info['chunks'])}")
    print(f"\U0001F4B0 Total size: {chunk_info['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
