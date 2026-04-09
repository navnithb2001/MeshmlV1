"""Google Cloud Storage utilities for file management."""

import logging
import os
from datetime import timedelta
from typing import BinaryIO, Optional

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

logger = logging.getLogger(__name__)


class StorageClient:
    """GCS client wrapper for model and dataset storage."""

    def __init__(self, bucket_name: str):
        """
        Initialize storage client.

        Args:
            bucket_name: GCS bucket name
        """
        self.bucket_name = bucket_name
        self._client: Optional[storage.Client] = None
        self._bucket: Optional[storage.Bucket] = None

    @property
    def client(self) -> storage.Client:
        """Lazy-load GCS client."""
        if self._client is None:
            # Use application default credentials or service account key
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                self._client = storage.Client.from_service_account_json(credentials_path)
            else:
                # Will use default credentials (works in GCP environments)
                self._client = storage.Client()
        return self._client

    @property
    def bucket(self) -> storage.Bucket:
        """Get GCS bucket."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def generate_presigned_upload_url(
        self, blob_path: str, content_type: str = "text/x-python", expires_in: int = 3600
    ) -> str:
        """
        Generate presigned URL for uploading files.

        Args:
            blob_path: Path within bucket (e.g., "models/123/model.py")
            content_type: MIME type of file
            expires_in: URL expiration in seconds (default: 1 hour)

        Returns:
            Presigned URL for PUT request
        """
        try:
            blob = self.bucket.blob(blob_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="PUT",
                content_type=content_type,
            )
            logger.info(f"Generated presigned upload URL for {blob_path}")
            return url
        except GoogleCloudError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def generate_presigned_download_url(self, blob_path: str, expires_in: int = 3600) -> str:
        """
        Generate presigned URL for downloading files.

        Args:
            blob_path: Path within bucket
            expires_in: URL expiration in seconds

        Returns:
            Presigned URL for GET request
        """
        try:
            blob = self.bucket.blob(blob_path)
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method="GET",
            )
            logger.info(f"Generated presigned download URL for {blob_path}")
            return url
        except GoogleCloudError as e:
            logger.error(f"Failed to generate download URL: {e}")
            raise

    def upload_file(
        self, source_file: BinaryIO, blob_path: str, content_type: str = "text/x-python"
    ) -> str:
        """
        Upload file directly to GCS.

        Args:
            source_file: File-like object to upload
            blob_path: Destination path in bucket
            content_type: MIME type

        Returns:
            GCS path (gs://bucket/path)
        """
        try:
            blob = self.bucket.blob(blob_path)
            blob.upload_from_file(source_file, content_type=content_type)
            gcs_path = f"gs://{self.bucket_name}/{blob_path}"
            logger.info(f"Uploaded file to {gcs_path}")
            return gcs_path
        except GoogleCloudError as e:
            logger.error(f"Failed to upload file: {e}")
            raise

    def download_file(self, blob_path: str, destination_file: BinaryIO) -> None:
        """
        Download file from GCS.

        Args:
            blob_path: Source path in bucket
            destination_file: File-like object to write to
        """
        try:
            blob = self.bucket.blob(blob_path)
            blob.download_to_file(destination_file)
            logger.info(f"Downloaded file from {blob_path}")
        except GoogleCloudError as e:
            logger.error(f"Failed to download file: {e}")
            raise

    def delete_file(self, blob_path: str) -> None:
        """
        Delete file from GCS.

        Args:
            blob_path: Path to delete
        """
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            logger.info(f"Deleted file {blob_path}")
        except GoogleCloudError as e:
            logger.error(f"Failed to delete file: {e}")
            raise

    def file_exists(self, blob_path: str) -> bool:
        """
        Check if file exists in GCS.

        Args:
            blob_path: Path to check

        Returns:
            True if file exists
        """
        try:
            blob = self.bucket.blob(blob_path)
            return blob.exists()
        except GoogleCloudError as e:
            logger.error(f"Failed to check file existence: {e}")
            return False

    def get_file_size(self, blob_path: str) -> Optional[int]:
        """
        Get file size in bytes.

        Args:
            blob_path: Path to file

        Returns:
            File size in bytes, or None if file doesn't exist
        """
        try:
            blob = self.bucket.blob(blob_path)
            blob.reload()
            return blob.size
        except GoogleCloudError as e:
            logger.error(f"Failed to get file size: {e}")
            return None


# Global storage client instances
def get_model_storage() -> StorageClient:
    """Get storage client for models bucket."""
    from app.config import settings

    return StorageClient(settings.GCS_BUCKET_MODELS)


def get_dataset_storage() -> StorageClient:
    """Get storage client for datasets bucket."""
    from app.config import settings

    return StorageClient(settings.GCS_BUCKET_DATASETS)


def get_artifact_storage() -> StorageClient:
    """Get storage client for artifacts bucket."""
    from app.config import settings

    return StorageClient(settings.GCS_BUCKET_ARTIFACTS)
