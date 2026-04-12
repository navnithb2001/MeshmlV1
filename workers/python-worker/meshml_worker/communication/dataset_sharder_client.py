"""
Dataset Sharder Client

Handles communication with the Dataset Sharder service for:
- Getting worker's batch assignments
- Downloading assigned data shards
- Reporting batch download/processing status
"""

import asyncio
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class DatasetSharderClient:
    """Client for Dataset Sharder service communication"""

    def __init__(
        self, sharder_url: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 2.0
    ):
        """
        Initialize Dataset Sharder client

        Args:
            sharder_url: Base URL of Dataset Sharder service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.sharder_url = sharder_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"Initialized Dataset Sharder client for {sharder_url}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def connect(self) -> None:
        """Create HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("Dataset Sharder client session created")

    async def close(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Dataset Sharder client session closed")

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for aiohttp request

        Returns:
            Response JSON data

        Raises:
            RuntimeError: If request fails after all retries
        """
        if not self.session:
            await self.connect()

        url = f"{self.sharder_url}{endpoint}"

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Request to {endpoint} failed (attempt {attempt}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    raise RuntimeError(
                        f"Request to {endpoint} failed after {self.max_retries} attempts: {e}"
                    )

    async def get_worker_assignment(
        self, worker_id: str, job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get batch assignments for this worker

        Args:
            worker_id: Worker identifier
            job_id: Optional job ID filter

        Returns:
            Assignment data with batch IDs, shard info, progress
        """
        logger.info(f"Getting batch assignment for worker {worker_id}")

        endpoint = f"/distribution/workers/{worker_id}/assignment"
        if job_id:
            endpoint += f"?job_id={job_id}"

        assignment = await self._make_request("GET", endpoint)

        logger.info(
            f"Worker {worker_id} assigned {len(assignment.get('assigned_batches', []))} batches "
            f"from shard {assignment.get('shard_id')}"
        )

        return assignment

    async def get_worker_batches(self, worker_id: str) -> List[str]:
        """
        Get list of batch IDs assigned to worker

        Args:
            worker_id: Worker identifier

        Returns:
            List of batch IDs
        """
        logger.info(f"Getting batch list for worker {worker_id}")

        endpoint = f"/distribution/workers/{worker_id}/batches"
        batches = await self._make_request("GET", endpoint)

        logger.info(f"Worker {worker_id} has {len(batches)} assigned batches")

        return batches

    async def download_batch(
        self, worker_id: str, batch_id: str, local_path: Path, extract: bool = True
    ) -> Path:
        """
        Download a data batch to local storage

        Args:
            worker_id: Worker identifier
            batch_id: Batch identifier to download
            local_path: Local directory to save batch
            extract: Whether to extract tar.gz archive

        Returns:
            Path to downloaded/extracted batch

        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Downloading batch {batch_id} for worker {worker_id}")

        # Update status to downloading
        await self.update_batch_status(worker_id, batch_id, "downloading")

        try:
            if not self.session:
                await self.connect()

            # Ensure local path exists
            local_path.mkdir(parents=True, exist_ok=True)

            # Prefer signed URL if available (GCS)
            download_url = None
            try:
                signed = await self._make_request(
                    "GET", f"/distribution/batches/{batch_id}/download-url"
                )
                download_url = signed.get("download_url")
            except Exception:
                download_url = None

            if download_url:
                url = download_url
            else:
                endpoint = f"/distribution/workers/{worker_id}/batches/{batch_id}/download"
                url = f"{self.sharder_url}{endpoint}"

            # Download file
            tar_file = local_path / f"{batch_id}.tar.gz"
            async with self.session.get(url) as response:
                response.raise_for_status()
                with open(tar_file, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

            logger.info(f"Downloaded batch {batch_id} to {tar_file}")

            # Extract if requested
            if extract:
                extract_path = local_path / batch_id
                extract_path.mkdir(parents=True, exist_ok=True)

                with tarfile.open(tar_file, "r:gz") as tar:
                    tar.extractall(path=extract_path)

                logger.info(f"Extracted batch {batch_id} to {extract_path}")
                tar_file.unlink()
                await self.update_batch_status(worker_id, batch_id, "completed")
                return extract_path

            await self.update_batch_status(worker_id, batch_id, "completed")
            return tar_file

        except Exception as e:
            logger.error(f"Failed to download batch {batch_id}: {e}")

            # Update status to failed
            await self.update_batch_status(worker_id, batch_id, "failed", failure_reason=str(e))

            raise RuntimeError(f"Batch download failed: {e}")

    async def update_batch_status(
        self, worker_id: str, batch_id: str, status: str, failure_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update batch download/processing status

        Args:
            worker_id: Worker identifier
            batch_id: Batch identifier
            status: Status (downloading, completed, failed)
            failure_reason: Reason if status is failed

        Returns:
            Response data
        """
        logger.info(f"Updating batch {batch_id} status to {status}")

        endpoint = f"/distribution/workers/{worker_id}/batches/{batch_id}/status"

        data = {"status": status}
        if failure_reason:
            data["failure_reason"] = failure_reason

        return await self._make_request("POST", endpoint, json=data)

    async def get_batch_assignment_info(self, batch_id: str) -> Dict[str, Any]:
        """
        Get assignment information for a specific batch

        Args:
            batch_id: Batch identifier

        Returns:
            Assignment details (worker_id, status, timestamps, etc.)
        """
        logger.info(f"Getting assignment info for batch {batch_id}")

        endpoint = f"/distribution/batches/{batch_id}/assignment"
        return await self._make_request("GET", endpoint)

    async def download_all_assigned_batches(
        self, worker_id: str, local_base_path: Path, job_id: Optional[str] = None
    ) -> List[Path]:
        """
        Download all batches assigned to this worker

        Args:
            worker_id: Worker identifier
            local_base_path: Base directory for downloads
            job_id: Optional job ID filter

        Returns:
            List of paths to downloaded batches
        """
        logger.info(f"Downloading all batches for worker {worker_id}")

        # Get assignment
        assignment = await self.get_worker_assignment(worker_id, job_id)
        batch_ids = assignment.get("assigned_batches", [])

        if not batch_ids:
            logger.warning(f"No batches assigned to worker {worker_id}")
            return []

        # Download each batch
        downloaded_paths = []
        for batch_id in batch_ids:
            try:
                batch_path = await self.download_batch(
                    worker_id, batch_id, local_base_path / f"batch_{batch_id}", extract=True
                )
                downloaded_paths.append(batch_path)

            except Exception as e:
                logger.error(f"Failed to download batch {batch_id}: {e}")
                # Continue with other batches
                continue

        logger.info(
            f"Downloaded {len(downloaded_paths)}/{len(batch_ids)} batches "
            f"for worker {worker_id}"
        )

        return downloaded_paths

    async def get_distribution_stats(self) -> Dict[str, Any]:
        """
        Get overall distribution statistics

        Returns:
            Statistics about batch distribution, workers, progress
        """
        logger.info("Getting distribution statistics")

        endpoint = "/distribution/stats"
        return await self._make_request("GET", endpoint)

    async def health_check(self) -> bool:
        """
        Check if Dataset Sharder service is healthy

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            endpoint = "/distribution/health"
            response = await self._make_request("GET", endpoint)
            return response.get("status") == "healthy"

        except Exception as e:
            logger.error(f"Dataset Sharder health check failed: {e}")
            return False
