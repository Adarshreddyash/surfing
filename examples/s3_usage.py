# examples/s3_usage.py
import asyncio
import torch
from transformers import AutoTokenizer
import os
import sys

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streaming_weights import StreamingBertModel, ModelChunker, WeightServer
from streaming_weights.storage import S3Backend
import logging


async def s3_streaming_example():
    """Example of streaming model inference using S3 storage"""

    # Load environment variables from .env file
    from dotenv import load_dotenv
    if not load_dotenv():
        print("❌ Error: No .env file found!")
        return

    # 1. Create S3 backend for the server
    print("☁️ Setting up S3 storage backend...")
    
    # Configure logging for better debugging
    logging.basicConfig(level=logging.DEBUG)
    
    # Get bucket info from environment or use defaults
    bucket_name = os.getenv("AWS_S3_BUCKET", "streaming-weights-bucket")
    bucket_prefix = os.getenv("AWS_S3_PREFIX", "bert-models/tiny")
    
    print(f"🪣 Using S3 bucket: {bucket_name}/{bucket_prefix}")
    print(f"🔑 AWS Region: {os.getenv('AWS_DEFAULT_REGION', 'us-east-1')}")
    
    s3_storage = S3Backend(
        bucket_name=bucket_name,
        prefix=bucket_prefix,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    # Verify model files exist in S3
    print("🔍 Checking for model files in S3...")
    files_to_check = ["embeddings.pt", "pooler.pt", "layer_0.pt", "layer_1.pt"]
    for file in files_to_check:
        exists = await s3_storage.exists(file)
        print(f"  {'✅' if exists else '❌'} {file}")

    # 2. Start the weight server with S3 backend
    print("🚀 Starting weight server...")
    server = WeightServer(storage_backend=s3_storage, port=8765)
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
    text = "Hello, this is a streaming weights test with S3!"
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


def main():
    """Run the example with proper error handling"""
    try:
        asyncio.run(s3_streaming_example())
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        # Check common issues
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            print("\nMissing AWS credentials! Make sure to:")
            print("1. Set AWS_ACCESS_KEY_ID environment variable")
            print("2. Set AWS_SECRET_ACCESS_KEY environment variable")
            print("3. Set AWS_DEFAULT_REGION environment variable (optional, defaults to us-east-1)")
        
        if "Access Denied" in str(e):
            print("\nAWS access denied! Make sure:")
            print("1. Your AWS credentials are correct")
            print("2. Your IAM user has s3:GetObject permission on the bucket")
            print("3. The S3 bucket name and prefix are correct")


if __name__ == "__main__":
    main()
