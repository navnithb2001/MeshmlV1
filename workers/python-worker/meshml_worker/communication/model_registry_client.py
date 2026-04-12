"""
Model Registry Client

Handles communication with the Model Registry service for:
- Fetching model metadata
- Downloading model files
- Querying model versions
- Model lifecycle management
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """Client for Model Registry service communication"""

    def __init__(
        self, registry_url: str, timeout: int = 60, max_retries: int = 3, retry_delay: float = 2.0
    ):
        """
        Initialize Model Registry client

        Args:
            registry_url: Base URL of Model Registry service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.registry_url = registry_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"Initialized Model Registry client for {registry_url}")

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
            logger.info("Model Registry client session created")

    async def close(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Model Registry client session closed")

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

        url = f"{self.registry_url}{endpoint}"

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

    async def get_model(self, model_id: int) -> Dict[str, Any]:
        """
        Get model metadata by ID

        Args:
            model_id: Model identifier

        Returns:
            Model metadata including name, version, paths, etc.
        """
        logger.info(f"Getting model metadata for model {model_id}")

        endpoint = f"/api/v1/models/{model_id}"
        model_data = await self._make_request("GET", endpoint)

        logger.info(
            f"Retrieved model: {model_data.get('name')} "
            f"v{model_data.get('version')} "
            f"(state: {model_data.get('state')})"
        )

        return model_data

    async def search_models(
        self,
        name: Optional[str] = None,
        architecture_type: Optional[str] = None,
        dataset_type: Optional[str] = None,
        group_id: Optional[int] = None,
        state: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Search for models with filters

        Args:
            name: Filter by model name (partial match)
            architecture_type: Filter by architecture (CNN, RNN, etc.)
            dataset_type: Filter by dataset type (ImageFolder, etc.)
            group_id: Filter by group ID
            state: Filter by lifecycle state
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Search results with models list and pagination info
        """
        logger.info("Searching models with filters")

        # Build query parameters
        params = {"limit": limit, "offset": offset}
        if name:
            params["name"] = name
        if architecture_type:
            params["architecture_type"] = architecture_type
        if dataset_type:
            params["dataset_type"] = dataset_type
        if group_id:
            params["group_id"] = group_id
        if state:
            params["state"] = state

        endpoint = "/api/v1/search/models"
        results = await self._make_request("GET", endpoint, params=params)

        logger.info(
            f"Found {results.get('total', 0)} models " f"(showing {len(results.get('models', []))})"
        )

        return results

    async def get_model_versions(self, parent_model_id: int) -> List[Dict[str, Any]]:
        """
        Get all versions of a model

        Args:
            parent_model_id: Parent model identifier

        Returns:
            List of model versions
        """
        logger.info(f"Getting versions for model {parent_model_id}")

        endpoint = f"/api/v1/versions/{parent_model_id}"
        versions = await self._make_request("GET", endpoint)

        logger.info(f"Found {len(versions)} versions")

        return versions

    async def get_latest_model_version(self, parent_model_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model

        Args:
            parent_model_id: Parent model identifier

        Returns:
            Latest model version or None
        """
        logger.info(f"Getting latest version for model {parent_model_id}")

        endpoint = f"/api/v1/versions/{parent_model_id}/latest"
        latest = await self._make_request("GET", endpoint)

        if latest:
            logger.info(f"Latest version: {latest.get('name')} " f"v{latest.get('version')}")
        else:
            logger.info("No versions found")

        return latest

    async def download_model(self, model_id: int, local_path: Path) -> Path:
        """
        Download model file from registry

        Args:
            model_id: Model identifier
            local_path: Local path to save model file

        Returns:
            Path to downloaded model file

        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Downloading model {model_id}")

        try:
            if not self.session:
                await self.connect()

            # Ensure local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Get download URL
            endpoint = f"/api/v1/models/{model_id}/download"
            url = f"{self.registry_url}{endpoint}"

            # Download file
            async with self.session.get(url) as response:
                response.raise_for_status()

                # Get filename from headers if available
                content_disposition = response.headers.get("Content-Disposition", "")
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')
                    local_path = local_path.parent / filename

                # Stream download to file
                with open(local_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

            logger.info(f"Model downloaded to {local_path}")

            return local_path

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise RuntimeError(f"Model download failed: {e}")

    async def get_model_by_name(
        self, name: str, group_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find model by name

        Args:
            name: Model name
            group_id: Optional group filter

        Returns:
            Model data or None if not found
        """
        logger.info(f"Finding model by name: {name}")

        # Search for model
        results = await self.search_models(name=name, group_id=group_id, limit=1)

        models = results.get("models", [])
        if models:
            return models[0]

        logger.info(f"Model not found: {name}")
        return None

    async def get_group_models(self, group_id: int) -> List[Dict[str, Any]]:
        """
        Get all models for a group

        Args:
            group_id: Group identifier

        Returns:
            List of models in the group
        """
        logger.info(f"Getting models for group {group_id}")

        endpoint = f"/api/v1/search/groups/{group_id}/models"
        models = await self._make_request("GET", endpoint)

        logger.info(f"Found {len(models)} models in group {group_id}")

        return models

    async def get_popular_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular models (most used)

        Args:
            limit: Maximum models to return

        Returns:
            List of popular models
        """
        logger.info(f"Getting {limit} popular models")

        endpoint = "/api/v1/search/popular"
        models = await self._make_request("GET", endpoint, params={"limit": limit})

        return models

    async def get_recent_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently created models

        Args:
            limit: Maximum models to return

        Returns:
            List of recent models
        """
        logger.info(f"Getting {limit} recent models")

        endpoint = "/api/v1/search/recent"
        models = await self._make_request("GET", endpoint, params={"limit": limit})

        return models

    async def get_architecture_types(self) -> List[str]:
        """
        Get list of available architecture types

        Returns:
            List of architecture type strings
        """
        logger.info("Getting architecture types")

        endpoint = "/api/v1/search/architecture-types"
        response = await self._make_request("GET", endpoint)

        return response.get("architecture_types", [])

    async def get_dataset_types(self) -> List[str]:
        """
        Get list of available dataset types

        Returns:
            List of dataset type strings
        """
        logger.info("Getting dataset types")

        endpoint = "/api/v1/search/dataset-types"
        response = await self._make_request("GET", endpoint)

        return response.get("dataset_types", [])

    async def update_model_state(
        self, model_id: int, new_state: str, comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update model lifecycle state

        Args:
            model_id: Model identifier
            new_state: New state (uploading, validating, active, deprecated, etc.)
            comment: Optional state change comment

        Returns:
            Updated model data
        """
        logger.info(f"Updating model {model_id} state to {new_state}")

        endpoint = f"/api/v1/lifecycle/{model_id}/state"

        data = {"new_state": new_state}
        if comment:
            data["comment"] = comment

        result = await self._make_request("POST", endpoint, json=data)

        logger.info(f"Model state updated to {new_state}")

        return result

    async def health_check(self) -> bool:
        """
        Check if Model Registry service is healthy

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            endpoint = "/health"
            response = await self._make_request("GET", endpoint)
            return response.get("status") == "healthy"

        except Exception as e:
            logger.error(f"Model Registry health check failed: {e}")
            return False
