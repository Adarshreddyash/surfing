# streaming_weights/models/__init__.py

# Always import base models
from .bert import StreamingBertModel
from .llama import StreamingLlamaModel

# Try to import optional models
try:
    from .gpt import StreamingGPTModel
except ImportError:
    StreamingGPTModel = None

try:
    from .t5 import StreamingT5Model
except ImportError:
    StreamingT5Model = None

# List all available models
__all__ = [
    'StreamingBertModel',
    'StreamingGPTModel',
    'StreamingT5Model',
    'StreamingLlamaModel',
]
