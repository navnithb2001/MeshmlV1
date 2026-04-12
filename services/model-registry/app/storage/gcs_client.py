"""
Google Cloud Storage client for model file management
"""

import hashlib
import logging
from datetime import timedelta
from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound

from ..config import settings

logger = logging.getLogger(__name__)


class GCSClient:
    """Google Cloud Storage client wrapper"""

    def __init__(self):
        self.client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None

    async def initialize(self):
        """Initialize GCS client and bucket"""
        try:
            if settings.GCS_CREDENTIALS_PATH:
                self.client = storage.Client.from_service_account_json(
                    settings.GCS_CREDENTIALS_PATH, project=settings.GCS_PROJECT_ID
                )
            else:
                # Use default credentials (for GKE, Cloud Run, etc.)
                self.client = storage.Client(project=settings.GCS_PROJECT_ID)

            # Get or create bucket
            try:
                self.bucket = self.client.get_bucket(settings.GCS_BUCKET_NAME)
                logger.info(f"Connected to existing bucket: {settings.GCS_BUCKET_NAME}")
            except NotFound:
                self.bucket = self.client.create_bucket(settings.GCS_BUCKET_NAME, location="US")
                logger.info(f"Created new bucket: {settings.GCS_BUCKET_NAME}")

        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise

    def get_model_path(self, model_id: int, filename: str = "model.py") -> str:
        """Get GCS path for model file"""
        return f"{settings.MODEL_STORAGE_PREFIX}/{model_id}/{filename}"

    def get_gcs_uri(self, blob_path: str) -> str:
        """Get GCS URI for blob path"""
        return f"gs://{settings.GCS_BUCKET_NAME}/{blob_path}"

    async def generate_upload_url(
        self, model_id: int, filename: str = "model.py", expires_in_seconds: int = 3600
    ) -> str:
        """
        Generate signed URL for direct model upload

        Args:
            model_id: Model ID
            filename: Filename to upload
            expires_in_seconds: URL expiration time

        Returns:
            Signed upload URL
        """
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expires_in_seconds),
            method="PUT",
            content_type="text/x-python",
        )

        logger.info(f"Generated upload URL for model {model_id}: {blob_path}")
        return url

    async def upload_file(
        self, model_id: int, file_content: bytes, filename: str = "model.py"
    ) -> tuple[str, int, str]:
        """
        Upload model file to GCS

        Args:
            model_id: Model ID
            file_content: File content as bytes
            filename: Filename

        Returns:
            Tuple of (gcs_path, file_size, file_hash)
        """
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        file_size = len(file_content)

        # Upload with metadata
        blob.metadata = {
            "model_id": str(model_id),
            "file_hash": file_hash,
            "content_type": "text/x-python",
        }

        import asyncio

        await asyncio.to_thread(
            blob.upload_from_string, file_content, content_type="text/x-python"
        )

        logger.info(f"Uploaded model {model_id} to {blob_path} ({file_size} bytes)")

        return self.get_gcs_uri(blob_path), file_size, file_hash

    async def download_file(self, model_id: int, filename: str = "model.py") -> bytes:
        """
        Download model file from GCS

        Args:
            model_id: Model ID
            filename: Filename

        Returns:
            File content as bytes
        """
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        try:
            import asyncio

            content = await asyncio.to_thread(blob.download_as_bytes)
            logger.info(f"Downloaded model {model_id} from {blob_path}")
            return content
        except NotFound:
            logger.error(f"Model file not found: {blob_path}")
            raise FileNotFoundError(f"Model file not found: {blob_path}")

    async def generate_download_url(
        self, model_id: int, filename: str = "model.py", expires_in_seconds: int = 3600
    ) -> str:
        """
        Generate signed URL for direct model download

        Args:
            model_id: Model ID
            filename: Filename
            expires_in_seconds: URL expiration time

        Returns:
            Signed download URL
        """
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        url = blob.generate_signed_url(
            version="v4", expiration=timedelta(seconds=expires_in_seconds), method="GET"
        )

        logger.info(f"Generated download URL for model {model_id}: {blob_path}")
        return url

    async def delete_file(self, model_id: int, filename: str = "model.py") -> bool:
        """
        Delete model file from GCS

        Args:
            model_id: Model ID
            filename: Filename

        Returns:
            True if deleted successfully
        """
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        try:
            import asyncio

            await asyncio.to_thread(blob.delete)
            logger.info(f"Deleted model {model_id} from {blob_path}")
            return True
        except NotFound:
            logger.warning(f"Model file not found for deletion: {blob_path}")
            return False

    async def file_exists(self, model_id: int, filename: str = "model.py") -> bool:
        """Check if model file exists in GCS"""
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)
        import asyncio

        return await asyncio.to_thread(blob.exists)

    async def get_file_metadata(self, model_id: int, filename: str = "model.py") -> dict:
        """Get file metadata from GCS"""
        blob_path = self.get_model_path(model_id, filename)
        blob = self.bucket.blob(blob_path)

        try:
            import asyncio

            await asyncio.to_thread(blob.reload)
            return {
                "size": blob.size,
                "content_type": blob.content_type,
                "md5_hash": blob.md5_hash,
                "created": blob.time_created,
                "updated": blob.updated,
                "metadata": blob.metadata or {},
            }
        except NotFound:
            raise FileNotFoundError(f"Model file not found: {blob_path}")
