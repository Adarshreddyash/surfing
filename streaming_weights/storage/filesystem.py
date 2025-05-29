# streaming_weights/storage/filesystem.py
import asyncio
from pathlib import Path
from typing import Union, BinaryIO, List
import logging

from .base import StorageBackend


class FilesystemBackend(StorageBackend):
    """Storage backend that uses the local filesystem.
    
    This backend stores data as files in a directory on the local filesystem.
    It's compatible with the original implementation of the weight server.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize the filesystem backend.
        
        Args:
            base_dir: The base directory where files are stored
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")
    
    async def load(self, key: str) -> bytes:
        """Load data from a file.
        
        Args:
            key: The file name to load
            
        Returns:
            The file contents as bytes
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.base_dir / key
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use asyncio to avoid blocking the event loop when reading large files
        return await asyncio.to_thread(self._read_file, file_path)
    
    def _read_file(self, file_path: Path) -> bytes:
        """Read a file synchronously."""
        self.logger.debug(f"Loading file: {file_path}")
        return file_path.read_bytes()
    
    async def save(self, key: str, data: Union[bytes, BinaryIO]) -> None:
        """Save data to a file.
        
        Args:
            key: The file name to save to
            data: The data to save, either as bytes or a file-like object
            
        Raises:
            IOError: If the file couldn't be saved
        """
        file_path = self.base_dir / key
        
        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use asyncio to avoid blocking the event loop when writing large files
        if isinstance(data, bytes):
            await asyncio.to_thread(self._write_bytes, file_path, data)
        else:
            await asyncio.to_thread(self._write_file, file_path, data)
    
    def _write_bytes(self, file_path: Path, data: bytes) -> None:
        """Write bytes to a file synchronously."""
        self.logger.debug(f"Saving file: {file_path}")
        file_path.write_bytes(data)
    
    def _write_file(self, file_path: Path, data: BinaryIO) -> None:
        """Write a file-like object to a file synchronously."""
        self.logger.debug(f"Saving file: {file_path}")
        with open(file_path, 'wb') as f:
            f.write(data.read())
    
    async def exists(self, key: str) -> bool:
        """Check if a file exists.
        
        Args:
            key: The file name to check
            
        Returns:
            True if the file exists, False otherwise
        """
        file_path = self.base_dir / key
        return file_path.exists()
    
    async def list(self, prefix: str = "") -> List[str]:
        """List all files with the given prefix.
        
        Args:
            prefix: Optional prefix to filter files
            
        Returns:
            List of file names
        """
        # Use a generator expression to avoid loading all files into memory at once
        files = []
        for path in self.base_dir.glob(f"{prefix}*"):
            if path.is_file():
                # Return the relative path from the base directory
                files.append(str(path.relative_to(self.base_dir)))
        return files
