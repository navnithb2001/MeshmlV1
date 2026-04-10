"""
Data loading utilities

Handles downloading data shards and creating dataloaders.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataDownloader:
    """Download and manage training data shards

    Features:
    - Download data from URLs
    - Download from GCS
    - Progress tracking
    - Cache management
    """

    def __init__(self, data_dir: Path, use_cache: bool = True):
        """Initialize data downloader

        Args:
            data_dir: Directory for data storage
            use_cache: Whether to use cached data
        """
        self.data_dir = data_dir
        self.use_cache = use_cache
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_from_url(self, url: str, filename: str, force_download: bool = False) -> Path:
        """Download data from URL

        Args:
            url: URL to download from
            filename: Local filename
            force_download: Force re-download

        Returns:
            Path to downloaded file
        """
        file_path = self.data_dir / filename

        # Check cache
        if file_path.exists() and self.use_cache and not force_download:
            logger.info(f"Using cached data: {file_path}")
            return file_path

        logger.info(f"Downloading data from {url}")

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(file_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Data downloaded to {file_path}")
            return file_path

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download data from {url}: {e}")

    def download_from_gcs(
        self, bucket_name: str, blob_name: str, filename: str, force_download: bool = False
    ) -> Path:
        """Download data from Google Cloud Storage

        Args:
            bucket_name: GCS bucket name
            blob_name: Blob name (path)
            filename: Local filename
            force_download: Force re-download

        Returns:
            Path to downloaded file
        """
        file_path = self.data_dir / filename

        # Check cache
        if file_path.exists() and self.use_cache and not force_download:
            logger.info(f"Using cached data: {file_path}")
            return file_path

        logger.info(f"Downloading data from gs://{bucket_name}/{blob_name}")

        try:
            from google.cloud import storage

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download
            blob.download_to_filename(str(file_path))

            logger.info(f"Data downloaded to {file_path}")
            return file_path

        except ImportError:
            raise ImportError(
                "google-cloud-storage not installed. "
                "Install with: pip install google-cloud-storage"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download data from GCS gs://{bucket_name}/{blob_name}: {e}"
            )


def download_data_shard(shard_url: str, data_dir: Path, shard_id: Optional[str] = None) -> Path:
    """Download data shard

    Args:
        shard_url: URL to data shard
        data_dir: Directory for data storage
        shard_id: Optional shard ID for filename

    Returns:
        Path to downloaded shard
    """
    downloader = DataDownloader(data_dir)

    # Determine filename
    if shard_id:
        filename = f"shard_{shard_id}.data"
    else:
        filename = Path(shard_url).name

    # Download based on URL type
    if shard_url.startswith("gs://"):
        # GCS
        parts = shard_url[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS URL: {shard_url}")

        bucket_name, blob_name = parts
        return downloader.download_from_gcs(
            bucket_name=bucket_name, blob_name=blob_name, filename=filename
        )
    else:
        # HTTP(S)
        return downloader.download_from_url(url=shard_url, filename=filename)
