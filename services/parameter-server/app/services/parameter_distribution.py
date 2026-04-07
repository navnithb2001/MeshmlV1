"""
Parameter Distribution Service for Parameter Server

Implements efficient parameter distribution to workers including:
- Push mode: Server broadcasts parameters to workers
- Pull mode: Workers request parameters from server
- Delta compression: Send only changed parameters
- Version-based synchronization
- Efficient serialization and transfer

Key Features:
- Multiple distribution strategies (push/pull/hybrid)
- Delta compression to reduce bandwidth
- Batch distribution to multiple workers
- Version tracking and validation
- Format conversion (PyTorch/NumPy/protobuf)
- Compression support (gzip, zstd)
"""

import gzip
import hashlib
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from app.services.parameter_storage import ParameterStorageService

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class DistributionMode(str, Enum):
    """Parameter distribution mode"""

    PUSH = "push"  # Server pushes to workers
    PULL = "pull"  # Workers pull from server
    HYBRID = "hybrid"  # Combination of push and pull


class CompressionType(str, Enum):
    """Compression type for parameter transfer"""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"  # Better compression ratio but requires zstandard package


class ParameterFormat(str, Enum):
    """Parameter serialization format"""

    PYTORCH = "pytorch"  # PyTorch tensors (native)
    NUMPY = "numpy"  # NumPy arrays
    PICKLE = "pickle"  # Python pickle
    PROTOBUF = "protobuf"  # Protocol buffers (future)


# ==================== Data Classes ====================


@dataclass
class ParameterPackage:
    """Package of parameters for distribution"""

    model_id: str
    version_id: int
    parameters: Dict[str, torch.Tensor]
    parameter_names: List[str]
    is_delta: bool = False
    base_version: Optional[int] = None
    checksum: str = ""
    size_bytes: int = 0
    compressed: bool = False
    compression_type: Optional[CompressionType] = None
    format_type: ParameterFormat = ParameterFormat.PYTORCH
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DistributionRequest:
    """Request for parameter distribution"""

    model_id: str
    worker_id: str
    current_version: Optional[int] = None
    requested_version: Optional[int] = None  # None = latest
    delta_only: bool = False
    compression: CompressionType = CompressionType.NONE
    format_type: ParameterFormat = ParameterFormat.PYTORCH
    parameter_names: Optional[List[str]] = None  # None = all parameters


@dataclass
class DistributionRecord:
    """Record of parameter distribution"""

    record_id: str
    model_id: str
    version_id: int
    worker_ids: List[str]
    is_delta: bool
    size_bytes: int
    compression_type: CompressionType
    distributed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionConfig:
    """Configuration for parameter distribution"""

    mode: DistributionMode = DistributionMode.PULL
    default_compression: CompressionType = CompressionType.NONE
    default_format: ParameterFormat = ParameterFormat.PYTORCH
    enable_delta_compression: bool = True
    delta_threshold: float = 0.1  # Send delta if changes < 10% of parameters
    auto_push_on_update: bool = False  # Auto push when parameters updated
    max_package_size_mb: float = 100.0  # Maximum package size
    enable_checksum: bool = True


# ==================== Parameter Distribution Service ====================


class ParameterDistributionService:
    """
    Service for distributing parameters to workers efficiently.

    Features:
    - Pull mode: Workers request parameters
    - Push mode: Server broadcasts to workers (requires worker endpoints)
    - Delta compression: Send only changed parameters
    - Version tracking: Ensure workers get correct versions
    - Multiple formats: PyTorch, NumPy, pickle
    - Compression: gzip, zstd
    """

    def __init__(
        self,
        parameter_storage: ParameterStorageService,
        default_config: Optional[DistributionConfig] = None,
    ):
        """
        Initialize parameter distribution service.

        Args:
            parameter_storage: Parameter storage service
            default_config: Default distribution configuration
        """
        self.parameter_storage = parameter_storage
        self.default_config = default_config or DistributionConfig()

        # Distribution history
        self.distribution_history: List[DistributionRecord] = []

        # Worker subscription tracking (for push mode)
        # Key: model_id -> Set[worker_id]
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # Cache for recently distributed packages
        # Key: (model_id, version_id, is_delta, compression) -> ParameterPackage
        self._package_cache: Dict[Tuple, ParameterPackage] = {}

        logger.info(
            f"ParameterDistributionService initialized (mode: {self.default_config.mode.value})"
        )

    def prepare_parameters(
        self, request: DistributionRequest, config: Optional[DistributionConfig] = None
    ) -> ParameterPackage:
        """
        Prepare parameters for distribution to a worker.

        Args:
            request: Distribution request
            config: Distribution configuration

        Returns:
            ParameterPackage ready for transfer
        """
        config = config or self.default_config

        # Determine target version
        target_version = request.requested_version
        if target_version is None:
            target_version = self.parameter_storage.get_current_version(request.model_id)

        # Check if delta compression is beneficial
        use_delta = False
        base_version = None

        if config.enable_delta_compression and request.delta_only and request.current_version:
            # Calculate delta benefit
            delta_info = self.parameter_storage.calculate_delta(
                request.model_id, request.current_version, target_version
            )

            if delta_info:
                compression_ratio = delta_info.compression_ratio
                if compression_ratio < config.delta_threshold:
                    use_delta = True
                    base_version = request.current_version
                    logger.info(
                        f"Using delta compression: {compression_ratio:.1%} of parameters changed"
                    )

        # Get parameters
        if use_delta:
            # Get only changed parameters
            parameters = self._get_delta_parameters(
                request.model_id, base_version, target_version, request.parameter_names
            )
        else:
            # Get full parameters
            parameters = self.parameter_storage.get_parameters(
                model_id=request.model_id,
                version_id=target_version,
                format="pytorch",  # Always get as PyTorch first
            )

            # Filter by parameter names if specified
            if request.parameter_names:
                parameters = {k: v for k, v in parameters.items() if k in request.parameter_names}

        # Convert format if needed
        if request.format_type != ParameterFormat.PYTORCH:
            parameters = self._convert_format(parameters, request.format_type)

        # Create package
        package = ParameterPackage(
            model_id=request.model_id,
            version_id=target_version,
            parameters=parameters,
            parameter_names=list(parameters.keys()),
            is_delta=use_delta,
            base_version=base_version,
            format_type=request.format_type,
        )

        # Calculate checksum
        if config.enable_checksum:
            package.checksum = self._calculate_checksum(parameters)

        # Calculate size
        package.size_bytes = self._calculate_size(parameters)

        # Apply compression
        if request.compression != CompressionType.NONE:
            package = self._compress_package(package, request.compression)

        logger.info(
            f"Prepared {'delta' if use_delta else 'full'} package for {request.worker_id}: "
            f"{len(parameters)} parameters, {package.size_bytes / 1024 / 1024:.2f} MB"
        )

        return package

    def distribute_to_worker(
        self,
        worker_id: str,
        request: DistributionRequest,
        config: Optional[DistributionConfig] = None,
    ) -> ParameterPackage:
        """
        Distribute parameters to a specific worker.

        Args:
            worker_id: Worker identifier
            request: Distribution request
            config: Distribution configuration

        Returns:
            ParameterPackage distributed
        """
        package = self.prepare_parameters(request, config)

        # Record distribution
        record = DistributionRecord(
            record_id=f"{request.model_id}-{package.version_id}-{worker_id}",
            model_id=request.model_id,
            version_id=package.version_id,
            worker_ids=[worker_id],
            is_delta=package.is_delta,
            size_bytes=package.size_bytes,
            compression_type=request.compression,
            metadata={"worker_id": worker_id, "format": request.format_type.value},
        )
        self.distribution_history.append(record)

        return package

    def broadcast_to_workers(
        self,
        model_id: str,
        worker_ids: List[str],
        version_id: Optional[int] = None,
        config: Optional[DistributionConfig] = None,
    ) -> Dict[str, ParameterPackage]:
        """
        Broadcast parameters to multiple workers.

        Args:
            model_id: Model identifier
            worker_ids: List of worker identifiers
            version_id: Version to distribute (None = latest)
            config: Distribution configuration

        Returns:
            Dict mapping worker_id to ParameterPackage
        """
        config = config or self.default_config

        if version_id is None:
            version_id = self.parameter_storage.get_current_version(model_id)

        packages = {}

        for worker_id in worker_ids:
            request = DistributionRequest(
                model_id=model_id,
                worker_id=worker_id,
                requested_version=version_id,
                compression=config.default_compression,
                format_type=config.default_format,
            )

            package = self.distribute_to_worker(worker_id, request, config)
            packages[worker_id] = package

        # Record broadcast
        if packages:
            sample_package = next(iter(packages.values()))
            record = DistributionRecord(
                record_id=f"{model_id}-{version_id}-broadcast",
                model_id=model_id,
                version_id=version_id,
                worker_ids=worker_ids,
                is_delta=False,
                size_bytes=sample_package.size_bytes,
                compression_type=config.default_compression,
                metadata={"broadcast": True},
            )
            self.distribution_history.append(record)

        logger.info(
            f"Broadcast parameters to {len(worker_ids)} workers "
            f"(model: {model_id}, version: {version_id})"
        )

        return packages

    def subscribe_worker(self, model_id: str, worker_id: str) -> bool:
        """
        Subscribe a worker to receive parameter updates (for push mode).

        Args:
            model_id: Model identifier
            worker_id: Worker identifier

        Returns:
            True if newly subscribed, False if already subscribed
        """
        if worker_id in self.subscriptions[model_id]:
            return False

        self.subscriptions[model_id].add(worker_id)
        logger.info(f"Worker {worker_id} subscribed to {model_id}")
        return True

    def unsubscribe_worker(self, model_id: str, worker_id: str) -> bool:
        """
        Unsubscribe a worker from parameter updates.

        Args:
            model_id: Model identifier
            worker_id: Worker identifier

        Returns:
            True if unsubscribed, False if not subscribed
        """
        if worker_id not in self.subscriptions[model_id]:
            return False

        self.subscriptions[model_id].remove(worker_id)
        logger.info(f"Worker {worker_id} unsubscribed from {model_id}")
        return True

    def get_subscribed_workers(self, model_id: str) -> List[str]:
        """Get list of workers subscribed to a model"""
        return list(self.subscriptions[model_id])

    def _get_delta_parameters(
        self,
        model_id: str,
        from_version: int,
        to_version: int,
        parameter_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get only changed parameters between versions"""
        # Get delta info
        delta_info = self.parameter_storage.calculate_delta(model_id, from_version, to_version)

        if not delta_info:
            # Fall back to full parameters
            return self.parameter_storage.get_parameters(
                model_id=model_id, version_id=to_version, format="pytorch"
            )

        # Get new version parameters
        all_params = self.parameter_storage.get_parameters(
            model_id=model_id, version_id=to_version, format="pytorch"
        )

        # Filter to only changed parameters
        changed_params = {k: v for k, v in all_params.items() if k in delta_info.changed_keys}

        # Further filter by parameter_names if specified
        if parameter_names:
            changed_params = {k: v for k, v in changed_params.items() if k in parameter_names}

        return changed_params

    def _convert_format(
        self, parameters: Dict[str, torch.Tensor], target_format: ParameterFormat
    ) -> Dict[str, Any]:
        """Convert parameters to target format"""
        if target_format == ParameterFormat.PYTORCH:
            return parameters

        elif target_format == ParameterFormat.NUMPY:
            return {k: v.cpu().numpy() for k, v in parameters.items()}

        elif target_format == ParameterFormat.PICKLE:
            # Keep as tensors, will be pickled during serialization
            return parameters

        elif target_format == ParameterFormat.PROTOBUF:
            # Future implementation
            raise NotImplementedError("Protobuf format not yet implemented")

        else:
            raise ValueError(f"Unknown format: {target_format}")

    def _calculate_checksum(self, parameters: Dict[str, Any]) -> str:
        """Calculate checksum for parameters"""
        hasher = hashlib.sha256()

        for name in sorted(parameters.keys()):
            param = parameters[name]

            # Convert to bytes
            if isinstance(param, torch.Tensor):
                param_bytes = param.cpu().numpy().tobytes()
            elif isinstance(param, np.ndarray):
                param_bytes = param.tobytes()
            else:
                param_bytes = pickle.dumps(param)

            hasher.update(name.encode())
            hasher.update(param_bytes)

        return hasher.hexdigest()

    def _calculate_size(self, parameters: Dict[str, Any]) -> int:
        """Calculate size in bytes of parameters"""
        total_size = 0

        for param in parameters.values():
            if isinstance(param, torch.Tensor):
                total_size += param.element_size() * param.nelement()
            elif isinstance(param, np.ndarray):
                total_size += param.nbytes
            else:
                # Estimate using pickle
                total_size += len(pickle.dumps(param))

        return total_size

    def _compress_package(
        self, package: ParameterPackage, compression_type: CompressionType
    ) -> ParameterPackage:
        """Compress package parameters"""
        if compression_type == CompressionType.NONE:
            return package

        # Serialize parameters
        serialized = pickle.dumps(package.parameters)

        # Compress
        if compression_type == CompressionType.GZIP:
            compressed = gzip.compress(serialized)
        elif compression_type == CompressionType.ZSTD:
            try:
                import zstandard as zstd

                compressor = zstd.ZstdCompressor()
                compressed = compressor.compress(serialized)
            except ImportError:
                logger.warning("zstandard not available, falling back to gzip")
                compressed = gzip.compress(serialized)
                compression_type = CompressionType.GZIP
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")

        # Update package
        package.compressed = True
        package.compression_type = compression_type

        # Store compressed data in metadata (in practice, would replace parameters)
        original_size = package.size_bytes
        compressed_size = len(compressed)
        package.metadata["compressed_data"] = compressed
        package.metadata["original_size"] = original_size
        package.size_bytes = compressed_size

        logger.debug(
            f"Compressed package: {original_size / 1024 / 1024:.2f} MB → "
            f"{compressed_size / 1024 / 1024:.2f} MB "
            f"({compressed_size / original_size:.1%})"
        )

        return package

    def decompress_package(self, package: ParameterPackage) -> ParameterPackage:
        """Decompress package parameters"""
        if not package.compressed:
            return package

        compressed_data = package.metadata.get("compressed_data")
        if not compressed_data:
            logger.warning("No compressed data found in package")
            return package

        # Decompress
        if package.compression_type == CompressionType.GZIP:
            decompressed = gzip.decompress(compressed_data)
        elif package.compression_type == CompressionType.ZSTD:
            try:
                import zstandard as zstd

                decompressor = zstd.ZstdDecompressor()
                decompressed = decompressor.decompress(compressed_data)
            except ImportError:
                raise RuntimeError("zstandard not available for decompression")
        else:
            raise ValueError(f"Unknown compression type: {package.compression_type}")

        # Deserialize
        parameters = pickle.loads(decompressed)

        # Update package
        package.parameters = parameters
        package.compressed = False
        package.size_bytes = package.metadata.get("original_size", package.size_bytes)

        return package

    def get_distribution_history(
        self,
        model_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[DistributionRecord]:
        """
        Get distribution history.

        Args:
            model_id: Filter by model (optional)
            worker_id: Filter by worker (optional)
            limit: Maximum records (optional)

        Returns:
            List of DistributionRecord (sorted by distributed_at desc)
        """
        history = self.distribution_history

        # Filter by model
        if model_id:
            history = [r for r in history if r.model_id == model_id]

        # Filter by worker
        if worker_id:
            history = [r for r in history if worker_id in r.worker_ids]

        # Sort descending
        history = sorted(history, key=lambda r: r.distributed_at, reverse=True)

        # Limit
        if limit:
            history = history[:limit]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get distribution statistics"""
        total_distributions = len(self.distribution_history)

        # Count delta vs full
        delta_count = len([r for r in self.distribution_history if r.is_delta])
        full_count = total_distributions - delta_count

        # Total data transferred
        total_bytes = sum(r.size_bytes for r in self.distribution_history)

        # Unique workers
        all_workers = set()
        for record in self.distribution_history:
            all_workers.update(record.worker_ids)

        # Subscriptions
        total_subscriptions = sum(len(workers) for workers in self.subscriptions.values())

        return {
            "total_distributions": total_distributions,
            "delta_distributions": delta_count,
            "full_distributions": full_count,
            "total_bytes_transferred": total_bytes,
            "total_mb_transferred": total_bytes / 1024 / 1024,
            "unique_workers": len(all_workers),
            "total_subscriptions": total_subscriptions,
            "models_with_subscriptions": len(self.subscriptions),
        }
