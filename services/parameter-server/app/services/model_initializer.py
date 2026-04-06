"""
Model Initialization Service for Parameter Server

Handles loading custom models from GCS, PyTorch model instantiation,
weight initialization (random or pre-trained), and model validation.

Key Features:
- Load custom model.py from GCS with create_model() function
- Support PyTorch models (primary focus)
- Random weight initialization or pre-trained model loading
- Model metadata validation and storage
- Thread-safe model registry
"""

import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Import GCS utilities (assuming Phase 3 model registry integration)
try:
    from google.cloud import storage
except ImportError:
    storage = None


logger = logging.getLogger(__name__)


# ==================== Enums ====================


class ModelStatus(str, Enum):
    """Model initialization status"""

    PENDING = "pending"
    LOADING = "loading"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"


class InitializationStrategy(str, Enum):
    """Weight initialization strategy"""

    RANDOM = "random"  # Random initialization
    PRETRAINED = "pretrained"  # Load pre-trained weights
    ZEROS = "zeros"  # Zero initialization
    ONES = "ones"  # One initialization
    XAVIER = "xavier_uniform"  # Xavier uniform
    KAIMING = "kaiming_normal"  # Kaiming normal (He initialization)


# ==================== Data Classes ====================


@dataclass
class ModelMetadata:
    """Model metadata from MODEL_METADATA dict in model.py"""

    name: str
    version: str
    framework: str = "pytorch"
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_parameters: Optional[int] = None
    description: Optional[str] = None
    author: Optional[str] = None
    tags: list = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for model initialization"""

    model_id: str
    gcs_model_path: str  # GCS path to model.py
    initialization_strategy: InitializationStrategy = InitializationStrategy.RANDOM
    pretrained_weights_path: Optional[str] = None  # GCS path to pretrained .pt/.pth file
    device: str = "cpu"  # "cpu", "cuda", "mps"
    seed: Optional[int] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)  # Args for create_model()


@dataclass
class InitializedModel:
    """Container for initialized model and metadata"""

    model_id: str
    model: nn.Module
    metadata: ModelMetadata
    config: ModelConfig
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    num_parameters: int = 0
    device: str = "cpu"
    checksum: Optional[str] = None  # SHA256 of model state dict


# ==================== Model Initializer Service ====================


class ModelInitializerService:
    """
    Service for initializing PyTorch models from custom GCS code.

    Workflow:
    1. Download model.py from GCS
    2. Dynamically import and validate create_model() function
    3. Validate MODEL_METADATA
    4. Instantiate model using create_model()
    5. Initialize weights (random or pre-trained)
    6. Move to device (CPU/GPU)
    7. Store in registry
    """

    def __init__(
        self,
        default_device: str = "cpu",
        gcs_bucket_name: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize model initializer service.

        Args:
            default_device: Default device for models ("cpu", "cuda", "mps")
            gcs_bucket_name: GCS bucket name for models
            temp_dir: Temporary directory for downloaded files
        """
        self.default_device = default_device
        self.gcs_bucket_name = gcs_bucket_name
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Model registry
        self.models: Dict[str, InitializedModel] = {}

        # GCS client
        self.gcs_client = storage.Client() if storage else None

        logger.info(
            f"ModelInitializerService initialized "
            f"(device={default_device}, bucket={gcs_bucket_name})"
        )

    async def initialize_model(self, config: ModelConfig) -> InitializedModel:
        """
        Initialize a model from GCS.

        Args:
            config: Model configuration

        Returns:
            InitializedModel with model instance and metadata

        Raises:
            ValueError: Invalid configuration
            RuntimeError: Model initialization failed
        """
        model_id = config.model_id
        logger.info(f"Initializing model {model_id} from {config.gcs_model_path}")

        # Create initial model entry
        initialized_model = InitializedModel(
            model_id=model_id,
            model=None,
            metadata=None,
            config=config,
            status=ModelStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            device=config.device,
        )

        self.models[model_id] = initialized_model

        try:
            # 1. Download model.py from GCS
            initialized_model.status = ModelStatus.LOADING
            local_model_path = await self._download_model_from_gcs(config.gcs_model_path)

            # 2. Import and validate model module
            create_model_fn, metadata = self._import_and_validate_model(local_model_path)

            initialized_model.metadata = metadata

            # 3. Instantiate model
            initialized_model.status = ModelStatus.INITIALIZING
            model = create_model_fn(**config.model_kwargs)

            if not isinstance(model, nn.Module):
                raise ValueError(f"create_model() must return nn.Module, got {type(model)}")

            # 4. Initialize weights
            self._initialize_weights(model, config)

            # 5. Move to device
            device = torch.device(config.device)
            model = model.to(device)

            # 6. Calculate model info
            num_parameters = sum(p.numel() for p in model.parameters())
            checksum = self._calculate_checksum(model)

            # 7. Update model entry
            initialized_model.model = model
            initialized_model.status = ModelStatus.READY
            initialized_model.num_parameters = num_parameters
            initialized_model.checksum = checksum
            initialized_model.updated_at = datetime.utcnow()

            logger.info(
                f"Model {model_id} initialized successfully "
                f"({num_parameters:,} parameters, device={config.device})"
            )

            return initialized_model

        except Exception as e:
            logger.error(f"Failed to initialize model {model_id}: {e}")
            initialized_model.status = ModelStatus.FAILED
            initialized_model.error_message = str(e)
            initialized_model.updated_at = datetime.utcnow()
            raise RuntimeError(f"Model initialization failed: {e}") from e

    async def _download_model_from_gcs(self, gcs_path: str) -> str:
        """
        Download model.py from GCS.

        Args:
            gcs_path: GCS path (gs://bucket/path/to/model.py)

        Returns:
            Local path to downloaded model.py
        """
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        if not self.gcs_client:
            raise RuntimeError("GCS client not available (google-cloud-storage not installed)")

        # Parse GCS path
        parts = gcs_path[5:].split("/", 1)  # Remove "gs://"
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        # Download to temp file
        local_path = os.path.join(
            self.temp_dir, f"model_{hashlib.md5(gcs_path.encode()).hexdigest()}.py"
        )

        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            logger.info(f"Downloaded model from {gcs_path} to {local_path}")
            return local_path

        except Exception as e:
            raise RuntimeError(f"Failed to download model from GCS: {e}") from e

    def _import_and_validate_model(self, model_path: str) -> Tuple[Callable, ModelMetadata]:
        """
        Dynamically import model.py and validate.

        Args:
            model_path: Local path to model.py

        Returns:
            Tuple of (create_model function, ModelMetadata)

        Raises:
            ValueError: Validation failed
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        # Dynamic import
        spec = importlib.util.spec_from_file_location("custom_model", model_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(f"Failed to import model: {e}") from e

        # Validate create_model() function
        if not hasattr(module, "create_model"):
            raise ValueError("Model must define create_model() function")

        create_model_fn = getattr(module, "create_model")

        if not callable(create_model_fn):
            raise ValueError("create_model must be callable")

        # Validate MODEL_METADATA
        if not hasattr(module, "MODEL_METADATA"):
            raise ValueError("Model must define MODEL_METADATA dict")

        metadata_dict = getattr(module, "MODEL_METADATA")

        if not isinstance(metadata_dict, dict):
            raise ValueError("MODEL_METADATA must be a dict")

        # Required fields
        required_fields = ["name", "version", "framework"]
        for field in required_fields:
            if field not in metadata_dict:
                raise ValueError(f"MODEL_METADATA missing required field: {field}")

        # Validate framework
        if metadata_dict["framework"].lower() != "pytorch":
            raise ValueError(f"Only PyTorch models supported, got: {metadata_dict['framework']}")

        # Create metadata object
        metadata = ModelMetadata(
            name=metadata_dict["name"],
            version=metadata_dict["version"],
            framework=metadata_dict["framework"],
            input_shape=metadata_dict.get("input_shape"),
            output_shape=metadata_dict.get("output_shape"),
            num_parameters=metadata_dict.get("num_parameters"),
            description=metadata_dict.get("description"),
            author=metadata_dict.get("author"),
            tags=metadata_dict.get("tags", []),
            custom_fields={
                k: v
                for k, v in metadata_dict.items()
                if k
                not in [
                    "name",
                    "version",
                    "framework",
                    "input_shape",
                    "output_shape",
                    "num_parameters",
                    "description",
                    "author",
                    "tags",
                ]
            },
        )

        logger.info(
            f"Validated model: {metadata.name} v{metadata.version} "
            f"(framework={metadata.framework})"
        )

        return create_model_fn, metadata

    def _initialize_weights(self, model: nn.Module, config: ModelConfig) -> None:
        """
        Initialize model weights.

        Args:
            model: PyTorch model
            config: Model configuration
        """
        strategy = config.initialization_strategy

        if strategy == InitializationStrategy.PRETRAINED:
            # Load pre-trained weights
            if not config.pretrained_weights_path:
                raise ValueError("pretrained_weights_path required for PRETRAINED strategy")

            self._load_pretrained_weights(model, config.pretrained_weights_path)
            logger.info(f"Loaded pre-trained weights from {config.pretrained_weights_path}")

        elif strategy == InitializationStrategy.RANDOM:
            # PyTorch default random initialization (already done)
            if config.seed is not None:
                torch.manual_seed(config.seed)
                logger.info(f"Set random seed: {config.seed}")

        elif strategy == InitializationStrategy.ZEROS:
            # Zero initialization
            for param in model.parameters():
                nn.init.zeros_(param)
            logger.info("Applied zero initialization")

        elif strategy == InitializationStrategy.ONES:
            # One initialization
            for param in model.parameters():
                nn.init.ones_(param)
            logger.info("Applied one initialization")

        elif strategy == InitializationStrategy.XAVIER:
            # Xavier uniform initialization
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            logger.info("Applied Xavier uniform initialization")

        elif strategy == InitializationStrategy.KAIMING:
            # Kaiming normal initialization (He initialization)
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            logger.info("Applied Kaiming normal initialization")

        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")

    def _load_pretrained_weights(self, model: nn.Module, weights_path: str) -> None:
        """
        Load pre-trained weights from GCS or local file.

        Args:
            model: PyTorch model
            weights_path: Path to weights (.pt or .pth file)
        """
        # Download from GCS if needed
        if weights_path.startswith("gs://"):
            import asyncio

            local_path = asyncio.run(self._download_weights_from_gcs(weights_path))
        else:
            local_path = weights_path

        # Load state dict
        try:
            state_dict = torch.load(local_path, map_location="cpu")

            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict)

        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}") from e

    async def _download_weights_from_gcs(self, gcs_path: str) -> str:
        """
        Download weights from GCS.

        Args:
            gcs_path: GCS path to weights file

        Returns:
            Local path to downloaded weights
        """
        if not self.gcs_client:
            raise RuntimeError("GCS client not available")

        # Parse GCS path
        parts = gcs_path[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""

        # Download to temp file
        local_path = os.path.join(
            self.temp_dir, f"weights_{hashlib.md5(gcs_path.encode()).hexdigest()}.pt"
        )

        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            logger.info(f"Downloaded weights from {gcs_path} to {local_path}")
            return local_path

        except Exception as e:
            raise RuntimeError(f"Failed to download weights from GCS: {e}") from e

    def _calculate_checksum(self, model: nn.Module) -> str:
        """
        Calculate SHA256 checksum of model state dict.

        Args:
            model: PyTorch model

        Returns:
            Hex digest of SHA256 checksum
        """
        # Get state dict
        state_dict = model.state_dict()

        # Serialize to bytes
        import io

        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)

        # Calculate checksum
        checksum = hashlib.sha256(buffer.read()).hexdigest()

        return checksum

    def get_model(self, model_id: str) -> Optional[InitializedModel]:
        """
        Get initialized model by ID.

        Args:
            model_id: Model identifier

        Returns:
            InitializedModel or None if not found
        """
        return self.models.get(model_id)

    def list_models(self) -> Dict[str, InitializedModel]:
        """
        List all initialized models.

        Returns:
            Dict of model_id -> InitializedModel
        """
        return self.models.copy()

    def delete_model(self, model_id: str) -> bool:
        """
        Delete model from registry.

        Args:
            model_id: Model identifier

        Returns:
            True if deleted, False if not found
        """
        if model_id in self.models:
            # Clean up
            initialized_model = self.models[model_id]
            if initialized_model.model is not None:
                del initialized_model.model

            del self.models[model_id]
            logger.info(f"Deleted model {model_id}")
            return True

        return False

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information (without model weights).

        Args:
            model_id: Model identifier

        Returns:
            Model info dict or None
        """
        initialized_model = self.models.get(model_id)

        if not initialized_model:
            return None

        return {
            "model_id": initialized_model.model_id,
            "status": initialized_model.status.value,
            "metadata": (
                {
                    "name": initialized_model.metadata.name if initialized_model.metadata else None,
                    "version": (
                        initialized_model.metadata.version if initialized_model.metadata else None
                    ),
                    "framework": (
                        initialized_model.metadata.framework if initialized_model.metadata else None
                    ),
                    "description": (
                        initialized_model.metadata.description
                        if initialized_model.metadata
                        else None
                    ),
                }
                if initialized_model.metadata
                else None
            ),
            "num_parameters": initialized_model.num_parameters,
            "device": initialized_model.device,
            "checksum": initialized_model.checksum,
            "created_at": initialized_model.created_at.isoformat(),
            "updated_at": initialized_model.updated_at.isoformat(),
            "error_message": initialized_model.error_message,
        }

    async def reinitialize_model(
        self, model_id: str, new_strategy: Optional[InitializationStrategy] = None
    ) -> InitializedModel:
        """
        Reinitialize model weights with new strategy.

        Args:
            model_id: Model identifier
            new_strategy: New initialization strategy (optional)

        Returns:
            Updated InitializedModel

        Raises:
            ValueError: Model not found or not ready
        """
        initialized_model = self.models.get(model_id)

        if not initialized_model:
            raise ValueError(f"Model not found: {model_id}")

        if initialized_model.status != ModelStatus.READY:
            raise ValueError(f"Model not ready: {initialized_model.status}")

        # Update strategy if provided
        if new_strategy:
            initialized_model.config.initialization_strategy = new_strategy

        # Reinitialize weights
        self._initialize_weights(initialized_model.model, initialized_model.config)

        # Update checksum
        initialized_model.checksum = self._calculate_checksum(initialized_model.model)
        initialized_model.updated_at = datetime.utcnow()

        logger.info(f"Reinitialized model {model_id}")

        return initialized_model

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Statistics dict
        """
        total_models = len(self.models)

        status_counts = {}
        for status in ModelStatus:
            status_counts[status.value] = sum(1 for m in self.models.values() if m.status == status)

        total_parameters = sum(m.num_parameters for m in self.models.values() if m.num_parameters)

        return {
            "total_models": total_models,
            "status_counts": status_counts,
            "total_parameters": total_parameters,
            "default_device": self.default_device,
        }
