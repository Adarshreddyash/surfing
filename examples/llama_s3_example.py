# examples/llama_s3_example.py
import asyncio
import torch
from transformers import LlamaTokenizer
import os
import sys

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streaming_weights.models import StreamingLlamaModel
from streaming_weights.weight_server import WeightServer
from streaming_weights.chunker import ModelChunker
from streaming_weights.storage import S3Backend
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# S3 Configuration from environment variables
# Configure these in your environment or .env file:
# AWS_S3_BUCKET - S3 bucket name
# AWS_S3_PREFIX - Folder in the bucket (default: models/tinyllama)
# AWS_DEFAULT_REGION - AWS region (default: us-east-1)
# AWS_ACCESS_KEY_ID - AWS access key (optional)
# AWS_SECRET_ACCESS_KEY - AWS secret key (optional)

# Get bucket info from environment or use defaults
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "streaming-weights-bucket")
S3_PREFIX = os.getenv("AWS_S3_PREFIX", "models/tinyllama")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


async def main():
    # Model settings
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    port = 8765
    
    # Create S3 backend
    print("☁️ Setting up S3 storage backend...")
    print(f"🚮 Using S3 bucket: {S3_BUCKET}/{S3_PREFIX}")
    print(f"🔑 AWS Region: {S3_REGION}")
    
    s3_backend = S3Backend(
        bucket_name=S3_BUCKET,
        prefix=S3_PREFIX,
        region_name=S3_REGION
        # Credentials will be automatically picked up from environment variables:
        # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    )
    
    # Start the weight server with S3 backend
    print("🚀 Starting weight server with S3 backend...")
    server = WeightServer(storage_backend=s3_backend, port=port)
    server_task = asyncio.create_task(server.start_server())
    
    # Wait a moment for the server to start
    await asyncio.sleep(1)
    
    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
    # Create the streaming model
    print("📚 Creating streaming LLaMA model...")
    model = StreamingLlamaModel(
        model_name=model_name,
        websocket_uri=f"ws://localhost:{port}",
        cache_size=5,  # Cache size for layers
    )
    
    # Warm up the model by preloading the first few layers
    print("🔥 Warming up model...")
    await model.warmup([0, 1, 2])  # Preload first 3 layers
    
    # Prepare a simple prompt
    prompt = "Hi there! My name is Surfing AI. How can I help you today?"
    print(f"\n💬 Prompt: {prompt}")
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate a response
    print("\n🤖 Generating response...")
    
    # Process the input with the streaming model
    with torch.no_grad():
        # Forward pass through the model
        outputs = await model.forward_async(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # Get outputs from the model
        
        # Print some stats about the model
        cache_info = model.get_cache_info()
        print(f"\n📊 Model cache info: {cache_info}")
        
        # Generate a simple continuation using the model
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Simple auto-regressive generation
        generated_text = prompt
        print("\n🔄 Generated continuation:")
        print(generated_text, end="")
        
        for _ in range(20):  # Generate 20 new tokens
            # Get predictions for next token
            with torch.no_grad():
                outputs = await model.forward_async(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            
            # Get the predictions for the next token
            next_token_logits = outputs[:, -1, :]
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add the predicted token to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=-1)
            
            # Decode and print the token
            token_text = tokenizer.decode(next_token[0])
            generated_text += token_text
            print(token_text, end="", flush=True)
    
    print("\n\n✅ Done!")
    
    # Clean up
    await model.cleanup()
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


# Function to chunk the model and upload to S3
async def chunk_model_to_s3():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Create S3 backend
    s3_backend = S3Backend(
        bucket_name=S3_BUCKET,
        prefix=S3_PREFIX,
        region_name=S3_REGION
        # Credentials will be automatically picked up from environment variables
    )
    
    print(f"📦 Chunking model {model_name} and uploading to S3...")
    chunker = ModelChunker(model_name=model_name, storage_backend=s3_backend)
    chunk_info = await chunker.chunk_model()
    print(f"✅ Model chunked and uploaded to S3 successfully! Total size: {chunk_info['total_size_mb']:.2f} MB")
    print(f"☁️ S3 Location: s3://{S3_BUCKET}/{S3_PREFIX}/")


def check_s3_chunks_exist():
    """Check if the model chunks exist in S3 bucket"""
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        s3 = boto3.client(
            's3',
            region_name=S3_REGION
            # Credentials will be automatically picked up from environment variables
        )
        
        # Check if config.json exists in the bucket/prefix
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/config.json")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                # For other errors, assume chunks don't exist
                print(f"Error checking S3: {e}")
                return False
    except Exception as e:
        print(f"Error connecting to S3: {e}")
        return False


if __name__ == "__main__":
    # Check if model chunks exist in S3, if not, chunk the model first
    print("Checking if model chunks exist in S3...")
    if not check_s3_chunks_exist():
        print("Model chunks not found in S3. Chunking the model and uploading to S3 first...")
        asyncio.run(chunk_model_to_s3())
    else:
        print("Model chunks found in S3. Proceeding with streaming...")
    
    # Run the main function
    asyncio.run(main())
