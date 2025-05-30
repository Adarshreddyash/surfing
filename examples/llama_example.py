# examples/llama_example.py
import asyncio
import torch
from transformers import LlamaTokenizer

# Import the streaming LLaMA model
from streaming_weights import StreamingLlamaModel, WeightServer, ModelChunker


async def main():
    # Model settings
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Using TinyLlama as requested
    chunks_dir = "./model_chunks/tinyllama"
    port = 8765
    
    # Start the weight server in the background
    print("🚀 Starting weight server...")
    server = WeightServer(chunks_dir=chunks_dir, port=port)
    server_task = asyncio.create_task(server.start_server())
    
    # Wait a moment for the server to start
    await asyncio.sleep(1)
    
    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
    # Create the streaming model
    print("📚 Creating streaming model...")
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
        # For a proper generation, you would typically use a language modeling head
        # Here we'll just demonstrate how to use the streaming model for inference
        
        # Prepare for text generation
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Simple auto-regressive generation (just a few tokens for demonstration)
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
            
            # Get the predictions for the next token (last token in sequence)
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


# Function to chunk the model if needed
async def chunk_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./model_chunks/tinyllama"
    
    print(f"📦 Chunking model {model_name}...")
    chunker = ModelChunker(model_name=model_name, output_dir=output_dir)
    chunk_info = await chunker.chunk_model()
    print(f"✅ Model chunked successfully! Total size: {chunk_info['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    import os
    
    # Check if model chunks exist, if not, chunk the model first
    if not os.path.exists("./model_chunks/tinyllama"):
        print("Model chunks not found. Chunking the model first...")
        asyncio.run(chunk_model())
    
    # Run the main function
    asyncio.run(main())
