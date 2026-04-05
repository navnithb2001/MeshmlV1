"""
Parameter Storage Service for Parameter Server

Manages in-memory parameter tensors, version control for model checkpoints,
and Redis-backed persistence for fault tolerance.

Key Features:
- In-memory parameter storage (PyTorch tensors)
- Version control with automatic versioning
- Checkpoint management (save/load/list)
- Redis-backed persistence for durability
- Parameter retrieval with version selection
- Delta compression for efficient storage
"""

import copy
import hashlib
import io
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import redis
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class ParameterFormat(str, Enum):
    """Parameter storage format"""

    PYTORCH = "pytorch"  # PyTorch tensors
    NUMPY = "numpy"  # NumPy arrays


class CheckpointType(str, Enum):
    """Checkpoint type"""

    MANUAL = "manual"  # Manually triggered
    AUTO = "auto"  # Automatically saved (periodic)
    BEST = "best"  # Best performing model
    FINAL = "final"  # Final model after training


# ==================== Data Classes ====================


@dataclass
class ParameterVersion:
    """Version information for parameters"""

    version_id: int
    model_id: str
    created_at: datetime
    checksum: str
    num_parameters: int
    total_size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Model checkpoint information"""

    checkpoint_id: str
    model_id: str
    version_id: int
    checkpoint_type: CheckpointType
    created_at: datetime
    checksum: str
    num_parameters: int
    size_bytes: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    redis_key: Optional[str] = None


@dataclass
class ParameterDelta:
    """Delta between two parameter versions"""

    from_version: int
    to_version: int
    changed_keys: List[str]
    delta_size_bytes: int
    compression_ratio: float


# ==================== Parameter Storage Service ====================


class ParameterStorageService:
    """
    Service for managing model parameters with versioning and persistence.

    Features:
    - In-memory storage of current parameters
    - Version control with automatic incrementing
    - Redis-backed persistence
    - Checkpoint management
    - Delta compression
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: Optional[int] = None,
        enable_redis: bool = True,
        checkpoint_retention: int = 10,  # Keep last N checkpoints
    ):
        """
        Initialize parameter storage service.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            enable_redis: Whether to enable Redis persistence
            checkpoint_retention: Number of checkpoints to retain
        """
        self.enable_redis = enable_redis
        self.checkpoint_retention = checkpoint_retention

        if redis_host is None or redis_port is None or redis_db is None:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                parsed = urlparse(redis_url)
                redis_host = redis_host or parsed.hostname or "localhost"
                redis_port = redis_port or parsed.port or 6379
                redis_db = (
                    redis_db
                    if redis_db is not None
                    else int((parsed.path or "/0").lstrip("/") or 0)
                )
            else:
                redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
                redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
                redis_db = redis_db if redis_db is not None else int(os.getenv("REDIS_DB", "0"))

        # Redis client
        if enable_redis:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,  # Binary mode for pickle
            )
            logger.info(f"Redis connected: {redis_host}:{redis_port}/{redis_db}")
        else:
            self.redis_client = None
            logger.info("Redis disabled, using in-memory only")

        # In-memory parameter storage
        # Key: model_id -> state_dict
        self.parameters: Dict[str, Dict[str, torch.Tensor]] = {}

        # Version tracking
        # Key: model_id -> current version_id
        self.current_versions: Dict[str, int] = {}

        # Version history
        # Key: model_id -> List[ParameterVersion]
        self.version_history: Dict[str, List[ParameterVersion]] = {}

        # Checkpoint storage
        # Key: model_id -> List[Checkpoint]
        self.checkpoints: Dict[str, List[Checkpoint]] = {}

        logger.info("ParameterStorageService initialized")

    def store_parameters(
        self,
        model_id: str,
        parameters: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
        create_checkpoint: bool = False,
        checkpoint_type: CheckpointType = CheckpointType.AUTO,
    ) -> ParameterVersion:
        """
        Store model parameters with automatic versioning.

        Args:
            model_id: Model identifier
            parameters: State dict of parameters (key -> tensor)
            metadata: Optional metadata for this version
            create_checkpoint: Whether to create a checkpoint
            checkpoint_type: Type of checkpoint to create

        Returns:
            ParameterVersion information
        """
        # Initialize if new model
        if model_id not in self.parameters:
            self.parameters[model_id] = {}
            self.current_versions[model_id] = 0
            self.version_history[model_id] = []
            self.checkpoints[model_id] = []

        # Increment version
        version_id = self.current_versions[model_id] + 1
        self.current_versions[model_id] = version_id

        # Store parameters (deep copy to avoid mutation)
        self.parameters[model_id] = {
            k: v.clone().detach() if isinstance(v, torch.Tensor) else v
            for k, v in parameters.items()
        }

        # Calculate metrics
        num_parameters = sum(p.numel() for p in parameters.values())
        total_size_bytes = sum(p.element_size() * p.numel() for p in parameters.values())
        checksum = self._calculate_checksum(parameters)

        # Create version record
        version = ParameterVersion(
            version_id=version_id,
            model_id=model_id,
            created_at=datetime.utcnow(),
            checksum=checksum,
            num_parameters=num_parameters,
            total_size_bytes=total_size_bytes,
            metadata=metadata or {},
        )

        self.version_history[model_id].append(version)

        logger.info(
            f"Stored parameters for {model_id} v{version_id} "
            f"({num_parameters:,} params, {total_size_bytes:,} bytes)"
        )

        # Persist to Redis if enabled
        if self.enable_redis and self.redis_client:
            self._persist_to_redis(model_id, version_id, parameters)

        # Create checkpoint if requested
        if create_checkpoint:
            self.create_checkpoint(
                model_id=model_id, checkpoint_type=checkpoint_type, metadata=metadata
            )

        return version

    def get_parameters(
        self,
        model_id: str,
        version_id: Optional[int] = None,
        format: ParameterFormat = ParameterFormat.PYTORCH,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve model parameters.

        Args:
            model_id: Model identifier
            version_id: Version to retrieve (None = latest)
            format: Return format (pytorch or numpy)

        Returns:
            Parameters dict or None if not found
        """
        if model_id not in self.parameters:
            return None

        # Get from memory (latest version)
        if version_id is None or version_id == self.current_versions[model_id]:
            parameters = self.parameters[model_id]
        else:
            # Load from Redis if available
            if self.enable_redis and self.redis_client:
                parameters = self._load_from_redis(model_id, version_id)
                if parameters is None:
                    logger.warning(f"Version {version_id} not found in Redis for {model_id}")
                    return None
            else:
                logger.warning("Redis not enabled, only latest version available")
                return None

        # Convert format if needed
        if format == ParameterFormat.NUMPY:
            return {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in parameters.items()
            }

        return parameters

    def get_parameter_names(self, model_id: str) -> Optional[List[str]]:
        """
        Get list of parameter names for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of parameter names or None
        """
        if model_id not in self.parameters:
            return None

        return list(self.parameters[model_id].keys())

    def get_parameter(
        self, model_id: str, parameter_name: str, version_id: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Get a specific parameter by name.

        Args:
            model_id: Model identifier
            parameter_name: Parameter name (e.g., "layer1.weight")
            version_id: Version to retrieve (None = latest)

        Returns:
            Parameter tensor or None
        """
        parameters = self.get_parameters(model_id, version_id)

        if parameters is None:
            return None

        return parameters.get(parameter_name)

    def update_parameter(
        self, model_id: str, parameter_name: str, value: torch.Tensor, create_version: bool = True
    ) -> bool:
        """
        Update a specific parameter.

        Args:
            model_id: Model identifier
            parameter_name: Parameter name
            value: New parameter value
            create_version: Whether to create a new version

        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.parameters:
            return False

        if parameter_name not in self.parameters[model_id]:
            return False

        # Update parameter
        self.parameters[model_id][parameter_name] = value.clone().detach()

        # Create new version if requested
        if create_version:
            self.store_parameters(model_id, self.parameters[model_id])

        logger.info(f"Updated parameter {parameter_name} for {model_id}")

        return True

    def create_checkpoint(
        self,
        model_id: str,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        checkpoint_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint of current parameters.

        Args:
            model_id: Model identifier
            checkpoint_type: Type of checkpoint
            checkpoint_id: Custom checkpoint ID (auto-generated if None)
            metrics: Model metrics (loss, accuracy, etc.)
            metadata: Additional metadata

        Returns:
            Checkpoint information
        """
        if model_id not in self.parameters:
            raise ValueError(f"Model not found: {model_id}")

        # Generate checkpoint ID
        if checkpoint_id is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            version_id = self.current_versions[model_id]
            checkpoint_id = f"{model_id}_v{version_id}_{timestamp}"

        # Get current parameters
        parameters = self.parameters[model_id]
        version_id = self.current_versions[model_id]

        # Calculate metrics
        num_parameters = sum(p.numel() for p in parameters.values())
        size_bytes = sum(p.element_size() * p.numel() for p in parameters.values())
        checksum = self._calculate_checksum(parameters)

        # Create checkpoint record
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            version_id=version_id,
            checkpoint_type=checkpoint_type,
            created_at=datetime.utcnow(),
            checksum=checksum,
            num_parameters=num_parameters,
            size_bytes=size_bytes,
            metrics=metrics or {},
            metadata=metadata or {},
        )

        # Save to Redis with special checkpoint key
        if self.enable_redis and self.redis_client:
            redis_key = f"checkpoint:{model_id}:{checkpoint_id}"
            self._save_checkpoint_to_redis(redis_key, parameters, checkpoint)
            checkpoint.redis_key = redis_key

        # Add to checkpoint list
        if model_id not in self.checkpoints:
            self.checkpoints[model_id] = []

        self.checkpoints[model_id].append(checkpoint)

        # Enforce retention policy
        self._enforce_checkpoint_retention(model_id)

        logger.info(
            f"Created {checkpoint_type.value} checkpoint {checkpoint_id} "
            f"for {model_id} v{version_id}"
        )

        return checkpoint

    def load_checkpoint(
        self, model_id: str, checkpoint_id: str, restore_to_current: bool = True
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load parameters from a checkpoint.

        Args:
            model_id: Model identifier
            checkpoint_id: Checkpoint identifier
            restore_to_current: Whether to restore as current version

        Returns:
            Parameters dict or None if not found
        """
        # Find checkpoint
        checkpoint = None
        if model_id in self.checkpoints:
            for ckpt in self.checkpoints[model_id]:
                if ckpt.checkpoint_id == checkpoint_id:
                    checkpoint = ckpt
                    break

        if checkpoint is None:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

        # Load from Redis
        if self.enable_redis and self.redis_client and checkpoint.redis_key:
            parameters = self._load_checkpoint_from_redis(checkpoint.redis_key)

            if parameters and restore_to_current:
                # Restore as current version
                self.store_parameters(
                    model_id,
                    parameters,
                    metadata={
                        "restored_from": checkpoint_id,
                        "restored_at": datetime.utcnow().isoformat(),
                    },
                )

            return parameters

        logger.warning("Redis not enabled or checkpoint key not found")
        return None

    def list_checkpoints(
        self, model_id: str, checkpoint_type: Optional[CheckpointType] = None
    ) -> List[Checkpoint]:
        """
        List checkpoints for a model.

        Args:
            model_id: Model identifier
            checkpoint_type: Filter by checkpoint type (None = all)

        Returns:
            List of checkpoints (sorted by created_at desc)
        """
        if model_id not in self.checkpoints:
            return []

        checkpoints = self.checkpoints[model_id]

        # Filter by type if specified
        if checkpoint_type:
            checkpoints = [ckpt for ckpt in checkpoints if ckpt.checkpoint_type == checkpoint_type]

        # Sort by created_at descending
        return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)

    def delete_checkpoint(self, model_id: str, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            model_id: Model identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        if model_id not in self.checkpoints:
            return False

        # Find and remove checkpoint
        checkpoint = None
        for i, ckpt in enumerate(self.checkpoints[model_id]):
            if ckpt.checkpoint_id == checkpoint_id:
                checkpoint = self.checkpoints[model_id].pop(i)
                break

        if checkpoint is None:
            return False

        # Delete from Redis
        if self.enable_redis and self.redis_client and checkpoint.redis_key:
            self.redis_client.delete(checkpoint.redis_key)

        logger.info(f"Deleted checkpoint {checkpoint_id}")

        return True

    def get_version_history(
        self, model_id: str, limit: Optional[int] = None
    ) -> List[ParameterVersion]:
        """
        Get version history for a model.

        Args:
            model_id: Model identifier
            limit: Maximum number of versions to return (None = all)

        Returns:
            List of versions (sorted by version_id desc)
        """
        if model_id not in self.version_history:
            return []

        versions = sorted(self.version_history[model_id], key=lambda v: v.version_id, reverse=True)

        if limit:
            versions = versions[:limit]

        return versions

    def get_current_version(self, model_id: str) -> Optional[int]:
        """
        Get current version ID for a model.

        Args:
            model_id: Model identifier

        Returns:
            Current version ID or None
        """
        return self.current_versions.get(model_id)

    def calculate_delta(
        self, model_id: str, from_version: int, to_version: int
    ) -> Optional[ParameterDelta]:
        """
        Calculate delta between two parameter versions.

        Args:
            model_id: Model identifier
            from_version: Starting version
            to_version: Ending version

        Returns:
            ParameterDelta or None if versions not found
        """
        # Load both versions
        params_from = self.get_parameters(model_id, from_version)
        params_to = self.get_parameters(model_id, to_version)

        if params_from is None or params_to is None:
            return None

        # Find changed parameters
        changed_keys = []
        delta_size = 0

        for key in params_to.keys():
            if key not in params_from:
                changed_keys.append(key)
                delta_size += params_to[key].element_size() * params_to[key].numel()
            elif not torch.equal(params_from[key], params_to[key]):
                changed_keys.append(key)
                delta_size += params_to[key].element_size() * params_to[key].numel()

        # Calculate compression ratio
        total_size = sum(p.element_size() * p.numel() for p in params_to.values())
        compression_ratio = delta_size / total_size if total_size > 0 else 0.0

        return ParameterDelta(
            from_version=from_version,
            to_version=to_version,
            changed_keys=changed_keys,
            delta_size_bytes=delta_size,
            compression_ratio=compression_ratio,
        )

    def _calculate_checksum(self, parameters: Dict[str, torch.Tensor]) -> str:
        """Calculate SHA256 checksum of parameters"""
        buffer = io.BytesIO()
        torch.save(parameters, buffer)
        buffer.seek(0)
        return hashlib.sha256(buffer.read()).hexdigest()

    def _persist_to_redis(
        self, model_id: str, version_id: int, parameters: Dict[str, torch.Tensor]
    ) -> None:
        """Persist parameters to Redis"""
        key = f"params:{model_id}:v{version_id}"
        version_key = f"params:{model_id}:current_version"

        # Serialize parameters
        buffer = io.BytesIO()
        torch.save(parameters, buffer)
        buffer.seek(0)

        # Store in Redis
        self.redis_client.set(key, buffer.read())
        self.redis_client.set(version_key, str(version_id))

        logger.debug(f"Persisted to Redis: {key}")

    def get_latest_parameters(
        self, model_id: str
    ) -> tuple[Optional[Dict[str, torch.Tensor]], Optional[int]]:
        """Get latest parameters preferring Redis as the source of truth."""
        if self.enable_redis and self.redis_client:
            version_key = f"params:{model_id}:current_version"
            raw_version = self.redis_client.get(version_key)
            if raw_version:
                try:
                    version_id = int(raw_version)
                    params = self._load_from_redis(model_id, version_id)
                    if params is not None:
                        self.current_versions[model_id] = version_id
                        return params, version_id
                except Exception:
                    pass

        if model_id in self.parameters:
            version_id = self.current_versions.get(model_id, 0)
            return self.parameters[model_id], version_id

        return None, None

    def _load_from_redis(self, model_id: str, version_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load parameters from Redis"""
        key = f"params:{model_id}:v{version_id}"

        data = self.redis_client.get(key)

        if data is None:
            return None

        # Deserialize
        buffer = io.BytesIO(data)
        parameters = torch.load(buffer)

        logger.debug(f"Loaded from Redis: {key}")

        return parameters

    def _save_checkpoint_to_redis(
        self, redis_key: str, parameters: Dict[str, torch.Tensor], checkpoint: Checkpoint
    ) -> None:
        """Save checkpoint to Redis"""
        # Create checkpoint package
        package = {
            "parameters": parameters,
            "checkpoint_info": {
                "checkpoint_id": checkpoint.checkpoint_id,
                "model_id": checkpoint.model_id,
                "version_id": checkpoint.version_id,
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "created_at": checkpoint.created_at.isoformat(),
                "metrics": checkpoint.metrics,
                "metadata": checkpoint.metadata,
            },
        }

        # Serialize
        buffer = io.BytesIO()
        torch.save(package, buffer)
        buffer.seek(0)

        # Store in Redis
        self.redis_client.set(redis_key, buffer.read())

        logger.debug(f"Saved checkpoint to Redis: {redis_key}")

    def _load_checkpoint_from_redis(self, redis_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load checkpoint from Redis"""
        data = self.redis_client.get(redis_key)

        if data is None:
            return None

        # Deserialize
        buffer = io.BytesIO(data)
        package = torch.load(buffer)

        return package["parameters"]

    def _enforce_checkpoint_retention(self, model_id: str) -> None:
        """Enforce checkpoint retention policy"""
        if model_id not in self.checkpoints:
            return

        checkpoints = self.checkpoints[model_id]

        # Keep BEST and FINAL checkpoints always
        protected_types = {CheckpointType.BEST, CheckpointType.FINAL}
        protected = [c for c in checkpoints if c.checkpoint_type in protected_types]
        others = [c for c in checkpoints if c.checkpoint_type not in protected_types]

        # Sort others by created_at descending
        others.sort(key=lambda c: c.created_at, reverse=True)

        # Keep only recent ones
        to_delete = others[self.checkpoint_retention :]

        for checkpoint in to_delete:
            self.delete_checkpoint(model_id, checkpoint.checkpoint_id)
            logger.info(f"Deleted old checkpoint: {checkpoint.checkpoint_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_models = len(self.parameters)
        total_versions = sum(len(versions) for versions in self.version_history.values())
        total_checkpoints = sum(len(ckpts) for ckpts in self.checkpoints.values())

        total_params = 0
        total_size = 0

        for model_id, params in self.parameters.items():
            total_params += sum(p.numel() for p in params.values())
            total_size += sum(p.element_size() * p.numel() for p in params.values())

        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "total_checkpoints": total_checkpoints,
            "total_parameters": total_params,
            "total_size_bytes": total_size,
            "redis_enabled": self.enable_redis,
        }
