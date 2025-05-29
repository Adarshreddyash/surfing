# streaming_weights/storage/base.py
from abc import ABC, abstractmethod
from typing import Union, BinaryIO


class StorageBackend(ABC):
    """Abstract base class for storage backends.
    
    This defines the interface that all storage backends must implement.
    Storage backends are responsible for loading and saving model weights.
    """
    
    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Load data from storage.
        
        Args:
            key: The identifier for the data to load
            
        Returns:
            The loaded data as bytes
            
        Raises:
            FileNotFoundError: If the data doesn't exist
        """
        pass
    
    @abstractmethod
    async def save(self, key: str, data: Union[bytes, BinaryIO]) -> None:
        """Save data to storage.
        
        Args:
            key: The identifier for the data
            data: The data to save, either as bytes or a file-like object
            
        Raises:
            IOError: If the data couldn't be saved
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if data exists in storage.
        
        Args:
            key: The identifier for the data
            
        Returns:
            True if the data exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def list(self, prefix: str = "") -> list[str]:
        """List all keys in storage with the given prefix.
        
        Args:
            prefix: Optional prefix to filter keys
            
        Returns:
            List of keys
        """
        pass
