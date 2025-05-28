# examples/basic_usage.py
import asyncio
import torch
from transformers import AutoTokenizer
from streaming_weights import StreamingBertModel, ModelChunker, WeightServer


async def basic_streaming_example():
    """Basic example of streaming model inference"""

    # 1. First, chunk your model (do this once)
    print("📦 Chunking model...")
    chunker = ModelChunker("prajjwal1/bert-tiny", "./chunks/bert-tiny")
    chunk_info = chunker.chunk_model()

    # 2. Start the weight server (in production, this would be separate)
    print("🚀 Starting weight server...")
    server = WeightServer("./chunks/bert-tiny", port=8765)
    server_task = asyncio.create_task(server.start_server())

    # Give server time to start
    await asyncio.sleep(1)

    # 3. Initialize streaming model
    print("🧠 Initializing streaming model...")
    model = StreamingBertModel(
        model_name="prajjwal1/bert-tiny",
        websocket_uri="ws://localhost:8765",
        cache_size=2,  # Cache only 2 layers
    )

    # 4. Prepare input
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    text = "Hello, this is a streaming weights test!"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # 5. Run inference
    print("⚡ Running inference...")
    with torch.no_grad():
        outputs = await model.forward_async(**inputs)

    print(f"✅ Inference complete!")
    print(f"Hidden states shape: {outputs[0].shape}")
    print(f"Pooled output shape: {outputs[1].shape}")
    print(f"Cache info: {model.get_cache_info()}")

    # Cancel server task
    server_task.cancel()


# examples/performance_test.py
import time
import asyncio
import torch
from transformers import AutoModel, AutoTokenizer
from streaming_weights import StreamingBertModel


async def performance_comparison():
    """Compare streaming vs traditional model loading"""

    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Performance test input text for comparison"
    inputs = tokenizer(text, return_tensors="pt")

    # Traditional model
    print("🔄 Loading traditional model...")
    start_time = time.time()
    traditional_model = AutoModel.from_pretrained(model_name)
    traditional_load_time = time.time() - start_time

    start_time = time.time()
    with torch.no_grad():
        traditional_output = traditional_model(**inputs)
    traditional_inference_time = time.time() - start_time

    # Streaming model (assuming server is running)
    print("🌊 Loading streaming model...")
    start_time = time.time()
    streaming_model = StreamingBertModel(model_name, "ws://localhost:8765")
    streaming_load_time = time.time() - start_time

    start_time = time.time()
    with torch.no_grad():
        streaming_output = await streaming_model.forward_async(**inputs)
    streaming_inference_time = time.time() - start_time

    # Results
    print("\n📊 Performance Comparison:")
    print(
        f"Traditional - Load: {traditional_load_time:.3f}s, Inference: {traditional_inference_time:.3f}s"
    )
    print(
        f"Streaming   - Load: {streaming_load_time:.3f}s, Inference: {streaming_inference_time:.3f}s"
    )
    print(f"Load speedup: {traditional_load_time / streaming_load_time:.2f}x")


# tests/test_chunker.py
import pytest
import tempfile
import shutil
from pathlib import Path
from streaming_weights import ModelChunker


def test_model_chunking():
    """Test model chunking functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        chunker = ModelChunker("prajjwal1/bert-tiny", temp_dir)
        chunk_info = chunker.chunk_model()

        # Verify chunks were created
        assert "embeddings" in chunk_info["chunks"]
        assert any("layer_" in key for key in chunk_info["chunks"])

        # Verify files exist
        output_dir = Path(temp_dir)
        assert (output_dir / "embeddings.pt").exists()
        assert (output_dir / "chunk_info.json").exists()

        # Verify chunk info can be loaded
        loaded_info = ModelChunker.load_chunk_info(temp_dir)
        assert loaded_info["model_name"] == "prajjwal1/bert-tiny"


# tests/test_streaming_model.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from streaming_weights import StreamingBertModel


@pytest.mark.asyncio
async def test_streaming_model_init():
    """Test streaming model initialization"""
    model = StreamingBertModel("prajjwal1/bert-tiny")

    assert model.model_name == "prajjwal1/bert-tiny"
    assert model.config.num_hidden_layers == 2  # TinyBERT has 2 layers
    assert len(model.layer_cache) == 0


@pytest.mark.asyncio
async def test_layer_caching():
    """Test LRU layer caching"""
    model = StreamingBertModel("prajjwal1/bert-tiny", cache_size=2)

    # Mock the weight fetching
    with patch.object(model, "_fetch_weights", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = {"attention.self.query.weight": torch.randn(128, 128)}

        # Load layers to test caching
        layer_0 = await model._load_layer(0)
        layer_1 = await model._load_layer(1)

        assert len(model.layer_cache) == 2
        assert 0 in model.layer_cache
        assert 1 in model.layer_cache


if __name__ == "__main__":
    # Run basic example
    asyncio.run(basic_streaming_example())
