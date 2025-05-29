# streaming_weights/storage/s3.py
import asyncio
import logging
from typing import Union, BinaryIO, List, Optional
import importlib.util

from .base import StorageBackend

# Check if boto3 is available
S3_AVAILABLE = importlib.util.find_spec('boto3') is not None

# Import boto3 for AWS S3 access if available
if S3_AVAILABLE:
    import boto3
    from botocore.exceptions import ClientError


class S3Backend(StorageBackend):
    """Storage backend that uses AWS S3.
    
    This backend stores data as objects in an S3 bucket.
    """
    
    def __init__(self, bucket_name: str, prefix: str = "", 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, 
                 aws_session_token: Optional[str] = None,
                 profile_name: Optional[str] = None,
                 region_name: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        """Initialize the S3 backend.
        
        AWS credentials can be provided in several ways:
        1. Directly via aws_access_key_id and aws_secret_access_key parameters
        2. Via AWS profile using profile_name parameter
        3. From environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        4. From AWS credentials file (~/.aws/credentials)
        5. From EC2 instance profile or ECS task role (when running on AWS)
        
        Args:
            bucket_name: The name of the S3 bucket
            prefix: Optional prefix to prepend to all keys (folder in bucket)
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            aws_session_token: Optional AWS session token (for temporary credentials)
            profile_name: Optional AWS profile name to use from credentials file
            region_name: Optional AWS region name (e.g., 'us-east-1')
            endpoint_url: Optional endpoint URL (for S3-compatible storage)
            
        Raises:
            ImportError: If boto3 is not installed
            ValueError: If the bucket doesn't exist or is not accessible
        """
        if not S3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install it with 'pip install boto3'."
            )
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.logger = logging.getLogger(__name__)
        
        # Create S3 client with provided credentials
        # If no credentials are provided, boto3 will look for them in the environment
        # or credentials file, or instance profile
        # Create session with profile if provided, otherwise use default credentials
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            self.s3 = session.client('s3', endpoint_url=endpoint_url)
        else:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
                endpoint_url=endpoint_url
            )
        
        # Verify bucket exists and is accessible
        try:
            self.s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ValueError(f"Bucket {bucket_name} does not exist")
            elif error_code == '403':
                raise ValueError(f"No permission to access bucket {bucket_name}")
            else:
                raise
    
    async def load(self, key: str) -> bytes:
        """Load data from S3.
        
        Args:
            key: The object key to load
            
        Returns:
            The object contents as bytes
            
        Raises:
            FileNotFoundError: If the object doesn't exist
        """
        s3_key = self._get_full_key(key)
        
        try:
            # Use asyncio to avoid blocking the event loop
            return await asyncio.to_thread(self._download_object, s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"Object not found: {s3_key}")
            else:
                self.logger.error(f"Error downloading object {s3_key}: {e}")
                raise
    
    def _download_object(self, s3_key: str) -> bytes:
        """Download an object from S3 synchronously."""
        self.logger.debug(f"Downloading object: {s3_key}")
        response = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)
        return response['Body'].read()
    
    async def save(self, key: str, data: Union[bytes, BinaryIO]) -> None:
        """Save data to S3.
        
        Args:
            key: The object key to save to
            data: The data to save, either as bytes or a file-like object
            
        Raises:
            IOError: If the object couldn't be saved
        """
        s3_key = self._get_full_key(key)
        
        try:
            # Use asyncio to avoid blocking the event loop
            if isinstance(data, bytes):
                await asyncio.to_thread(self._upload_bytes, s3_key, data)
            else:
                await asyncio.to_thread(self._upload_fileobj, s3_key, data)
        except ClientError as e:
            self.logger.error(f"Error uploading object {s3_key}: {e}")
            raise IOError(f"Failed to upload object to S3: {e}")
    
    def _upload_bytes(self, s3_key: str, data: bytes) -> None:
        """Upload bytes to S3 synchronously."""
        self.logger.debug(f"Uploading object: {s3_key}")
        self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=data)
    
    def _upload_fileobj(self, s3_key: str, data: BinaryIO) -> None:
        """Upload a file-like object to S3 synchronously."""
        self.logger.debug(f"Uploading object: {s3_key}")
        self.s3.upload_fileobj(data, self.bucket_name, s3_key)
    
    async def exists(self, key: str) -> bool:
        """Check if an object exists in S3.
        
        Args:
            key: The object key to check
            
        Returns:
            True if the object exists, False otherwise
        """
        s3_key = self._get_full_key(key)
        
        try:
            # Use asyncio to avoid blocking the event loop
            return await asyncio.to_thread(self._check_object_exists, s3_key)
        except Exception as e:
            self.logger.error(f"Error checking if object {s3_key} exists: {e}")
            return False
    
    def _check_object_exists(self, s3_key: str) -> bool:
        """Check if an object exists in S3 synchronously."""
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    async def list(self, prefix: str = "") -> List[str]:
        """List all objects in S3 with the given prefix.
        
        Args:
            prefix: Optional prefix to filter objects
            
        Returns:
            List of object keys
        """
        s3_prefix = self._get_full_key(prefix)
        
        try:
            # Use asyncio to avoid blocking the event loop
            return await asyncio.to_thread(self._list_objects, s3_prefix)
        except ClientError as e:
            self.logger.error(f"Error listing objects with prefix {s3_prefix}: {e}")
            return []
    
    def _list_objects(self, s3_prefix: str) -> List[str]:
        """List objects in S3 synchronously."""
        self.logger.debug(f"Listing objects with prefix: {s3_prefix}")
        
        objects = []
        paginator = self.s3.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Remove the prefix from the key to get the relative key
                    key = obj['Key']
                    if self.prefix and key.startswith(self.prefix):
                        key = key[len(self.prefix):]
                    objects.append(key)
        
        return objects
    
    def _get_full_key(self, key: str) -> str:
        """Get the full S3 key with the prefix."""
        return f"{self.prefix}{key}"
