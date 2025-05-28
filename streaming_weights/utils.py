# streaming_weights/utils.py
import requests
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


def download_and_chunk(
    model_name: str, output_dir: str, server_url: Optional[str] = None
) -> Dict[str, Any]:
    """Download and chunk a model, optionally uploading to server"""
    from .chunker import ModelChunker

    chunker = ModelChunker(model_name, output_dir)
    chunk_info = chunker.chunk_model()

    # Upload to server if specified
    if server_url:
        upload_chunks_to_server(output_dir, server_url)

    return chunk_info


def upload_chunks_to_server(chunks_dir: str, server_url: str):
    """Upload chunks to a remote server"""
    chunks_path = Path(chunks_dir)

    for chunk_file in chunks_path.glob("*.pt"):
        with open(chunk_file, "rb") as f:
            files = {"file": (chunk_file.name, f, "application/octet-stream")}
            response = requests.post(f"{server_url}/upload", files=files)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to upload {chunk_file.name}")


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a model for streaming"""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)

    return {
        "model_name": model_name,
        "model_type": config.model_type,
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "vocab_size": getattr(config, "vocab_size", 0),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 0),
    }


def calculate_chunk_hash(file_path: str) -> str:
    """Calculate hash of a chunk file for integrity checking"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
