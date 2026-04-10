"""
HTTP REST API Client for Parameter Server

Implements HTTP-based communication with the Parameter Server,
replacing gRPC for immediate functionality with JSON-based tensor transfer.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class HTTPClient:
    """HTTP REST API client for Parameter Server communication

    Provides methods to:
    - Register worker
    - Submit gradients
    - Get model parameters
    - Sync with parameter server

    Uses JSON for tensor serialization (less efficient than gRPC/protobuf
    but works with existing Parameter Server HTTP APIs).
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize HTTP client

        Args:
            base_url: Base URL of Parameter Server (e.g., http://34.61.230.151:8003)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connected = False

        # Create session with retry logic
        self.session = requests.Session()

        # Configure retries for connection errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(f"HTTP client initialized: base_url={base_url}")

    def connect(self) -> bool:
        """Connect to Parameter Server (verify availability)

        Returns:
            bool: True if server is reachable
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            self.connected = True
            logger.info(f"Connected to Parameter Server: {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Parameter Server: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Parameter Server"""
        self.session.close()
        self.connected = False
        logger.info("Disconnected from Parameter Server")

    def register_worker(
        self, worker_id: str, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Register worker with Parameter Server

        Args:
            worker_id: Worker identifier
            model_id: Model identifier
            metadata: Optional worker metadata

        Returns:
            dict: Registration response with worker info

        Raises:
            RuntimeError: If not connected or registration fails
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        payload = {"worker_id": worker_id, "model_id": model_id, "metadata": metadata or {}}

        try:
            response = self.session.post(
                f"{self.base_url}/sync/workers/register", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Worker registered: {worker_id}")
            return result
        except Exception as e:
            logger.error(f"Worker registration failed: {e}")
            raise RuntimeError(f"Worker registration failed: {e}")

    def _tensor_to_dict(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert PyTorch tensor to JSON-serializable dict

        Args:
            tensor: PyTorch tensor

        Returns:
            dict: Tensor data with shape, data array, and dtype
        """
        # Convert to CPU and numpy
        np_array = tensor.detach().cpu().numpy()

        return {
            "shape": list(np_array.shape),
            "data": np_array.flatten().tolist(),
            "dtype": str(np_array.dtype),
        }

    def _dict_to_tensor(self, data: Dict[str, Any]) -> torch.Tensor:
        """Convert dict to PyTorch tensor

        Args:
            data: Tensor data dict with shape, data, dtype

        Returns:
            torch.Tensor: Reconstructed tensor
        """
        shape = tuple(data["shape"])
        flat_data = data["data"]
        dtype_str = data.get("dtype", "float32")

        # Map dtype string to numpy dtype
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
        }
        dtype = dtype_map.get(dtype_str, np.float32)

        # Reconstruct array
        np_array = np.array(flat_data, dtype=dtype).reshape(shape)
        return torch.from_numpy(np_array)

    def push_gradients(
        self,
        worker_id: str,
        model_id: str,
        version_id: int,
        gradients: Dict[str, torch.Tensor],
        num_samples: int,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Submit gradients to Parameter Server

        Args:
            worker_id: Worker identifier
            model_id: Model identifier
            version_id: Parameter version used for gradient computation
            gradients: Dict of parameter name -> gradient tensor
            num_samples: Number of samples used for gradient computation
            loss: Training loss (optional)
            metrics: Additional metrics (optional)

        Returns:
            dict: Server response

        Raises:
            RuntimeError: If not connected or submission fails
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        # Convert gradients to JSON format
        gradient_dicts = {}
        for name, grad_tensor in gradients.items():
            gradient_dicts[name] = self._tensor_to_dict(grad_tensor)

        payload = {
            "worker_id": worker_id,
            "model_id": model_id,
            "version_id": version_id,
            "gradients": gradient_dicts,
            "num_samples": num_samples,
            "loss": loss,
            "metrics": metrics or {},
        }

        try:
            response = self.session.post(
                f"{self.base_url}/gradients/submit", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Gradients submitted: {len(gradients)} parameters, {num_samples} samples")
            return result
        except Exception as e:
            logger.error(f"Gradient submission failed: {e}")
            raise RuntimeError(f"Gradient submission failed: {e}")

    def get_weights(
        self, model_id: str, version_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get model parameters from Parameter Server

        Args:
            model_id: Model identifier
            version_id: Specific version to retrieve (optional, defaults to latest)

        Returns:
            dict: Parameter name -> tensor

        Raises:
            RuntimeError: If not connected or retrieval fails
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        # Build URL with optional version query param
        url = f"{self.base_url}/parameters/{model_id}"
        params = {}
        if version_id is not None:
            params["version_id"] = version_id

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

            # Note: The API docs say "metadata only, not actual tensors"
            # We need to check the actual response format
            # If the response contains tensor data, convert it
            if "parameters" in result:
                parameters = {}
                for name, tensor_data in result["parameters"].items():
                    if (
                        isinstance(tensor_data, dict)
                        and "shape" in tensor_data
                        and "data" in tensor_data
                    ):
                        parameters[name] = self._dict_to_tensor(tensor_data)
                    else:
                        logger.warning(f"Parameter {name} has unexpected format")

                logger.debug(f"Retrieved {len(parameters)} parameters")
                return parameters
            else:
                logger.warning(f"Unexpected response format: {result}")
                return {}

        except Exception as e:
            logger.error(f"Parameter retrieval failed: {e}")
            raise RuntimeError(f"Parameter retrieval failed: {e}")

    def get_model_version(self, model_id: str) -> int:
        """Get current model parameter version

        Args:
            model_id: Model identifier

        Returns:
            int: Current version ID

        Raises:
            RuntimeError: If not connected or retrieval fails
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        try:
            response = self.session.get(
                f"{self.base_url}/parameters/{model_id}/version", timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract version from response
            version = result.get("version_id", 0)
            logger.debug(f"Current model version: {version}")
            return version

        except Exception as e:
            logger.error(f"Version retrieval failed: {e}")
            raise RuntimeError(f"Version retrieval failed: {e}")

    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister worker from Parameter Server

        Args:
            worker_id: Worker identifier

        Returns:
            bool: True if successful
        """
        if not self.connected:
            return False

        try:
            response = self.session.delete(
                f"{self.base_url}/sync/workers/{worker_id}", timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Worker unregistered: {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Worker unregistration failed: {e}")
            return False
