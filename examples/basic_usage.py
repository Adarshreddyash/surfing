# examples/basic_usage.py
import asyncio
import torch
from transformers import AutoTokenizer
import os
import sys

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streaming_weights import StreamingBertModel, ModelChunker, WeightServer


async def basic_streaming_example():
    """Basic example of streaming model inference"""

    # 1. First, chunk your model (do this once)
    print("📦 Chunking model...")
    chunks_dir = "./chunks/bert-tiny"
    os.makedirs(chunks_dir, exist_ok=True)
    chunker = ModelChunker("prajjwal1/bert-tiny", chunks_dir)
    # Store chunk info for potential later use
    chunk_info = chunker.chunk_model()
    print(f"Model chunked into {len(chunk_info['chunks'])} pieces")

    # 2. Start the weight server (in production, this would be separate)
    print("🚀 Starting weight server...")
    server = WeightServer(chunks_dir, port=8765)
    server_task = asyncio.create_task(server.start_server())

    # Give server time to start
    await asyncio.sleep(1)

    # 3. Initialize streaming model...
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

    print("✅ Inference complete!")
    print(f"Hidden states shape: {outputs[0].shape}")
    print(f"Pooled output shape: {outputs[1].shape}")
    print(f"Cache info: {model.get_cache_info()}")

    # Cancel server task
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        print("Server task cancelled")


if __name__ == "__main__":
    # Run basic example
    asyncio.run(basic_streaming_example())
