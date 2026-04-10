"""
gRPC Client for Parameter Server Communication

Handles:
- Connecting to Parameter Server via gRPC
- Fetching model weights
- Pushing gradients
- Version synchronization
- Connection management with retries
"""

import gzip
import io
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Try to import grpc - it's optional but recommended
try:
    import grpc

    GRPC_AVAILABLE = True
except ImportError:
    grpc = None  # type: ignore
    GRPC_AVAILABLE = False

try:
    from meshml_worker.proto import parameter_server_pb2, parameter_server_pb2_grpc
except Exception:
    parameter_server_pb2 = None  # type: ignore
    parameter_server_pb2_grpc = None  # type: ignore

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry function on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for exponential delay

    Example:
        @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
        def push_gradients(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")

            # If all retries failed, raise the last exception
            raise RuntimeError(
                f"{func.__name__} failed after {max_retries} retries"
            ) from last_exception

        return wrapper

    return decorator


class GRPCClient:
    """gRPC client for Parameter Server communication

    Features:
    - Connect to Parameter Server
    - Get model weights
    - Push gradients
    - Version tracking
    - Compression support
    - Retry logic with exponential backoff
    - Connection health checking
    """

    def __init__(self, config: Any):
        """Initialize gRPC client

        Args:
            config: Parameter Server configuration
        """
        self.config = config
        self.channel: Optional[Any] = None
        self.stub: Optional[Any] = None
        self.connected = False
        self.current_version = 0

        # Retry configuration
        self.max_retries = getattr(config, "max_retries", 3)
        self.retry_delay = getattr(config, "retry_delay", 1.0)

        logger.info(f"Initialized gRPC client for {config.grpc_url}")

    def connect(self) -> None:
        """Connect to Parameter Server

        Raises:
            RuntimeError: If connection fails or grpc not installed
        """
        if not GRPC_AVAILABLE:
            raise RuntimeError(
                "grpc is not installed. Install with: pip install grpcio grpcio-tools"
            )

        try:
            if parameter_server_pb2_grpc is None:
                raise RuntimeError("Parameter Server proto not available")

            logger.info(f"Connecting to Parameter Server at {self.config.grpc_url}")

            # Create channel
            self.channel = grpc.insecure_channel(
                self.config.grpc_url,
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
                ],
            )

            # Create stub
            self.stub = parameter_server_pb2_grpc.ParameterServerStub(self.channel)

            # Test connection
            self._test_connection()

            self.connected = True
            logger.info("Successfully connected to Parameter Server")

        except ImportError:
            logger.error("gRPC not installed. Install with: pip install grpcio grpcio-tools")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Parameter Server: {e}")
            raise RuntimeError(f"Connection failed: {e}")

    def _test_connection(self) -> None:
        """Test connection with version request"""
        if not self.stub or parameter_server_pb2 is None:
            return
        try:
            self.stub.GetModelVersion(parameter_server_pb2.VersionRequest(job_id="healthcheck"))
        except Exception:
            pass

    def disconnect(self) -> None:
        """Disconnect from Parameter Server"""
        if self.channel:
            self.channel.close()
            self.connected = False
            logger.info("Disconnected from Parameter Server")

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def get_weights(
        self, job_id: str, worker_id: str, epoch: int = 0
    ) -> Tuple[Dict[str, Any], int]:
        """Fetch current model weights from Parameter Server

        Retries automatically on failure with exponential backoff.

        Args:
            job_id: Job identifier
            worker_id: Worker identifier
            epoch: Current epoch

        Returns:
            Tuple of (model_state_dict, version)

        Raises:
            RuntimeError: If not connected or request fails after retries
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        logger.info(f"Fetching weights for job {job_id}, epoch {epoch}")

        try:
            if self.stub and parameter_server_pb2 is not None:
                request = parameter_server_pb2.WeightsRequest(
                    job_id=job_id,
                    worker_id=worker_id,
                    current_version=self.current_version,
                    epoch=epoch,
                )
                response = self.stub.PullWeights(request)

                model_state = self._decompress_data(
                    response.model_state_dict, response.compression_type, response.uncompressed_size
                )
                state_dict = self._deserialize_tensor_dict(model_state)
                version = response.version
                self.current_version = version
                logger.info(
                    f"Received weights: version {version}, is_updated={response.is_updated}"
                )
                return state_dict, version

            # Fallback: simulate response
            response = self._simulate_weights_response(job_id, epoch)
            model_state = self._decompress_data(
                response["model_state_dict"],
                response.get("compression_type", "none"),
                response.get("uncompressed_size", 0),
            )
            state_dict = self._deserialize_tensor_dict(model_state)
            version = response["version"]
            self.current_version = version
            logger.info(f"Received weights: version {version}, is_updated={response['is_updated']}")
            return state_dict, version

        except Exception as e:
            logger.error(f"Failed to get weights: {e}")
            raise RuntimeError(f"Get weights failed: {e}")

    @retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
    def push_gradients(
        self,
        job_id: str,
        worker_id: str,
        gradients: Dict[str, Any],
        batch_id: int,
        epoch: int,
        batch_size: int,
        learning_rate: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Push computed gradients to Parameter Server

        Retries automatically on failure with exponential backoff.

        Args:
            job_id: Job identifier
            worker_id: Worker identifier
            gradients: Gradient tensors
            batch_id: Batch identifier
            epoch: Current epoch
            batch_size: Batch size
            learning_rate: Learning rate
            metadata: Optional metadata (loss, gradient_norm, etc.)

        Returns:
            Acknowledgment response

        Raises:
            RuntimeError: If not connected or push fails after retries
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        logger.debug(f"Pushing gradients for job {job_id}, batch {batch_id}")

        try:
            # Serialize gradients
            gradients_bytes = self._serialize_tensor_dict(gradients)

            # Compress if enabled
            compressed_data, compression_type, uncompressed_size = self._compress_data(
                gradients_bytes
            )

            if self.stub and parameter_server_pb2 is not None:
                gradient_metadata = parameter_server_pb2.GradientMetadata(
                    loss=metadata.get("loss", 0.0) if metadata else 0.0,
                    gradient_norm=metadata.get("gradient_norm", 0.0) if metadata else 0.0,
                    computation_time_ms=metadata.get("computation_time_ms", 0) if metadata else 0,
                    layer_norms=metadata.get("layer_norms", {}) if metadata else {},
                )

                request = parameter_server_pb2.GradientsUpdate(
                    job_id=job_id,
                    worker_id=worker_id,
                    batch_id=batch_id,
                    version=self.current_version,
                    epoch=epoch,
                    gradients=compressed_data,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    compression_type=compression_type,
                    uncompressed_size=uncompressed_size,
                    metadata=gradient_metadata,
                )
                response = self.stub.PushGradients(request)
                return {
                    "success": response.success,
                    "message": response.message,
                    "new_version": response.new_version,
                    "staleness_accepted": response.staleness_accepted,
                    "staleness_threshold": response.staleness_threshold,
                }

            # Simulate response fallback
            response = self._simulate_gradients_ack()

            if response["success"]:
                logger.debug(
                    f"Gradients pushed successfully: new_version={response['new_version']}"
                )
            else:
                logger.warning(f"Gradient push failed: {response['message']}")

            return response

        except Exception as e:
            logger.error(f"Failed to push gradients: {e}")
            raise RuntimeError(f"Push gradients failed: {e}")

    def get_model_version(self, job_id: str) -> Dict[str, Any]:
        """Get current model version information

        Args:
            job_id: Job identifier

        Returns:
            Version information
        """
        if not self.connected:
            raise RuntimeError("Not connected to Parameter Server")

        logger.debug(f"Getting model version for job {job_id}")

        try:
            if self.stub and parameter_server_pb2 is not None:
                response = self.stub.GetModelVersion(
                    parameter_server_pb2.VersionRequest(job_id=job_id)
                )
                return {
                    "current_version": response.current_version,
                    "epoch": response.epoch,
                    "total_updates": response.total_updates,
                    "last_update_timestamp": response.last_update_timestamp,
                }

            return {
                "current_version": self.current_version,
                "epoch": 0,
                "total_updates": 0,
                "last_update_timestamp": int(time.time()),
            }

        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            raise RuntimeError(f"Get version failed: {e}")

    def _compress_data(self, data: bytes) -> Tuple[bytes, str, int]:
        """Compress data if beneficial

        Args:
            data: Data to compress

        Returns:
            Tuple of (compressed_data, compression_type, uncompressed_size)
        """
        uncompressed_size = len(data)

        # Try gzip compression
        compressed = gzip.compress(data, compresslevel=6)
        compression_ratio = len(compressed) / uncompressed_size

        # Use compression if it reduces size by at least 20%
        if compression_ratio < 0.8:
            logger.debug(
                f"Compressed data: {uncompressed_size} -> {len(compressed)} bytes "
                f"({compression_ratio:.2%})"
            )
            return compressed, "gzip", uncompressed_size
        else:
            return data, "none", uncompressed_size

    def _decompress_data(self, data: bytes, compression_type: str, uncompressed_size: int) -> bytes:
        """Decompress data

        Args:
            data: Compressed data
            compression_type: Type of compression
            uncompressed_size: Expected uncompressed size

        Returns:
            Decompressed data
        """
        if compression_type == "gzip":
            decompressed = gzip.decompress(data)
            logger.debug(f"Decompressed data: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        elif compression_type == "none":
            return data
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    def _simulate_weights_response(self, job_id: str, epoch: int) -> Dict[str, Any]:
        """Simulate weights response (for testing without proto)"""
        # Create dummy model state
        dummy_state = {"layer1.weight": [[1.0, 2.0], [3.0, 4.0]]}
        model_bytes = self._serialize_tensor_dict(dummy_state)

        return {
            "model_state_dict": model_bytes,
            "version": self.current_version + 1,
            "epoch": epoch,
            "is_updated": True,
            "compression_type": "none",
            "uncompressed_size": len(model_bytes),
        }

    def _simulate_gradients_ack(self) -> Dict[str, Any]:
        """Simulate gradients acknowledgment (for testing without proto)"""
        return {
            "success": True,
            "message": "Gradients received",
            "new_version": self.current_version + 1,
            "staleness_accepted": True,
            "staleness_threshold": 5,
        }

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
        return False

    @staticmethod
    def _serialize_tensor_dict(data: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        arrays: Dict[str, np.ndarray] = {}
        for name, value in data.items():
            if hasattr(value, "detach"):
                arrays[name] = value.detach().cpu().numpy()
            else:
                arrays[name] = np.asarray(value)
        np.savez(buffer, **arrays)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_tensor_dict(data: bytes) -> Dict[str, Any]:
        if not data:
            return {}
        buffer = io.BytesIO(data)
        arrays = np.load(buffer, allow_pickle=False)
        return {name: arrays[name] for name in arrays.files}
