# streaming_weights/weight_server.py
import asyncio
import websockets
import torch
import json
import io
from pathlib import Path
import logging

 
class WeightServer:
    def __init__(self, chunks_dir, port=8000):
        self.chunks_dir = Path(chunks_dir)
        self.port = port
        self.cache = {}  # In-memory cache for frequently accessed weights
        self.logger = logging.getLogger(__name__)
        
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {self.chunks_dir}")

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

            # Load weights based on request
            if component_type == "layer":
                filename = f"layer_{component_id}.pt"
            elif component_type == "embeddings":
                filename = "embeddings.pt"
            elif component_type == "pooler":
                filename = "pooler.pt"
            else:
                return json.dumps({"success": False, "error": f"Unknown component: {component_type}"})

            # Check cache first
            cache_key = f"{component_type}_{component_id}"
            if cache_key in self.cache:
                self.logger.debug(f"Cache hit for {cache_key}")
                weights_bytes = self.cache[cache_key]
            else:
                # Load from disk
                file_path = self.chunks_dir / filename
                if not file_path.exists():
                    return json.dumps({"success": False, "error": f"File not found: {filename}"})

                self.logger.info(f"Loading {filename} from disk")
                # Load state dict and serialize
                state_dict = torch.load(file_path, map_location="cpu")
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                weights_bytes = buffer.getvalue()

                # Cache for future requests
                self.cache[cache_key] = weights_bytes

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
        self.logger.info(f"Starting weight server on port {self.port}")
        async def handler(websocket):
            await self.handle_client(websocket, "")
        async with websockets.serve(handler, "localhost", self.port):
            await asyncio.Future()  # Run forever


# CLI entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start a weight server for streaming model weights")
    parser.add_argument("--chunks-dir", "-d", required=True, help="Directory containing model chunks")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port to run server on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Start server
    server = WeightServer(args.chunks_dir, args.port)
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()
