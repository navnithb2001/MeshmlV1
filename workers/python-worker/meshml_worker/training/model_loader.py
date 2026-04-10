"""
Custom Model Loader

Downloads and dynamically imports custom model definitions from GCS.
Validates MODEL_METADATA and provides create_model() and create_dataloader() functions.
"""

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests

# Try to import Google Cloud Storage (optional dependency)
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    storage = None  # type: ignore
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Model metadata validation"""

    REQUIRED_FIELDS = [
        "name", "version", "framework", "input_shape", "output_shape",
        # Contract fields — define the math, not the user
        "task_type", "loss", "metrics",
    ]

    OPTIONAL_FIELDS = [
        "description", "author", "tags", "hyperparameters", "requirements",
        "num_outputs", "target_dtype",
    ]

    SUPPORTED_TASK_TYPES = {"classification", "regression", "binary"}
    SUPPORTED_LOSSES = {"cross_entropy", "mse", "mae", "bce_with_logits", "bce"}

    def __init__(self, metadata: Dict[str, Any]):
        """Initialize and validate metadata

        Args:
            metadata: Metadata dictionary
        """
        self.metadata = metadata
        self._validate()

    def _validate(self) -> None:
        """Validate metadata structure"""
        # Check required fields
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in self.metadata]
        if missing_fields:
            raise ValueError(f"MODEL_METADATA missing required fields: {missing_fields}")

        # Validate framework
        if self.metadata["framework"] not in ["pytorch", "tensorflow", "jax"]:
            raise ValueError(
                f"Unsupported framework: {self.metadata['framework']}. "
                f"Supported: pytorch, tensorflow, jax"
            )

        # Validate contract fields
        task_type = self.metadata["task_type"]
        if task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"MODEL_METADATA.task_type '{task_type}' is not supported. "
                f"Supported: {sorted(self.SUPPORTED_TASK_TYPES)}"
            )

        loss = self.metadata["loss"]
        if loss not in self.SUPPORTED_LOSSES:
            raise ValueError(
                f"MODEL_METADATA.loss '{loss}' is not supported. "
                f"Supported: {sorted(self.SUPPORTED_LOSSES)}"
            )

        metrics = self.metadata["metrics"]
        if not isinstance(metrics, list) or len(metrics) == 0:
            raise ValueError("MODEL_METADATA.metrics must be a non-empty list of strings.")

        logger.info(
            f"Model metadata validated: {self.metadata['name']} v{self.metadata['version']} "
            f"(task_type={task_type}, loss={loss})"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get metadata value

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value
        """
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get metadata value using dict syntax"""
        return self.metadata[key]


class ModelLoader:
    """Load custom model definitions dynamically

    Features:
    - Download model.py from GCS or HTTP URL
    - Dynamic import with validation
    - Extract create_model() and create_dataloader()
    - Validate MODEL_METADATA
    - Cache downloaded models
    """

    def __init__(self, models_dir: Path, use_cache: bool = True):
        """Initialize model loader

        Args:
            models_dir: Directory to store downloaded models
            use_cache: Whether to use cached models
        """
        self.models_dir = models_dir
        self.use_cache = use_cache
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_modules: Dict[str, Any] = {}

    def download_model_from_url(
        self, url: str, model_id: str, force_download: bool = False
    ) -> Path:
        """Download model definition from URL

        Args:
            url: URL to model.py file
            model_id: Model ID for storage
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model file
        """
        model_file = self.models_dir / f"{model_id}_model.py"

        # Check cache
        if model_file.exists() and self.use_cache and not force_download:
            logger.info(f"Using cached model file: {model_file}")
            return model_file

        logger.info(f"Downloading model from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Save to file
            with open(model_file, "wb") as f:
                f.write(response.content)

            logger.info(f"Model downloaded to {model_file}")
            return model_file

        except Exception as e:
            raise RuntimeError(f"Failed to download model from {url}: {e}")

    def download_model_from_gcs(
        self, bucket_name: str, blob_name: str, model_id: str, force_download: bool = False
    ) -> Path:
        """Download model definition from Google Cloud Storage

        Args:
            bucket_name: GCS bucket name
            blob_name: Blob name (path to model.py)
            model_id: Model ID for storage
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded model file
        """
        model_file = self.models_dir / f"{model_id}_model.py"

        # Check cache
        if model_file.exists() and self.use_cache and not force_download:
            logger.info(f"Using cached model file: {model_file}")
            return model_file

        logger.info(f"Downloading model from gs://{bucket_name}/{blob_name}")

        try:
            if not GCS_AVAILABLE:
                raise ImportError(
                    "google-cloud-storage not installed. "
                    "Install with: pip install google-cloud-storage"
                )

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download to file
            blob.download_to_filename(str(model_file))

            logger.info(f"Model downloaded to {model_file}")
            return model_file

        except ImportError:
            raise ImportError(
                "google-cloud-storage not installed. "
                "Install with: pip install google-cloud-storage"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from GCS gs://{bucket_name}/{blob_name}: {e}"
            )

    def load_model_module(self, model_file: Path, model_id: str) -> Any:
        """Load model module from file

        Args:
            model_file: Path to model.py file
            model_id: Model ID (used as module name)

        Returns:
            Loaded module
        """
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Check if already loaded
        module_name = f"meshml_custom_model_{model_id}"
        if module_name in self._loaded_modules:
            logger.info(f"Using cached module: {module_name}")
            return self._loaded_modules[module_name]

        # Load module dynamically
        logger.info(f"Loading module from {model_file}")

        try:
            spec = importlib.util.spec_from_file_location(module_name, model_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {model_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Cache module
            self._loaded_modules[module_name] = module

            logger.info(f"Module loaded successfully: {module_name}")
            return module

        except Exception as e:
            raise RuntimeError(f"Failed to load module from {model_file}: {e}")

    def validate_model_module(self, module: Any) -> None:
        """Validate model module has required functions

        Args:
            module: Loaded module

        Raises:
            ValueError: If module is invalid
        """
        # Check for MODEL_METADATA
        if not hasattr(module, "MODEL_METADATA"):
            raise ValueError("Module missing MODEL_METADATA constant")

        # Validate metadata
        ModelMetadata(module.MODEL_METADATA)

        # Check for create_model function
        if not hasattr(module, "create_model"):
            raise ValueError("Module missing create_model() function")

        if not callable(module.create_model):
            raise ValueError("create_model is not callable")

        # Check function signature
        sig = inspect.signature(module.create_model)
        logger.info(f"create_model signature: {sig}")

        # Check for create_dataloader function (optional but recommended)
        if hasattr(module, "create_dataloader"):
            if not callable(module.create_dataloader):
                raise ValueError("create_dataloader is not callable")

            sig = inspect.signature(module.create_dataloader)
            logger.info(f"create_dataloader signature: {sig}")
        else:
            logger.warning("Module does not have create_dataloader() function")

        logger.info("Model module validation passed")

    def get_model_functions(
        self, module: Any
    ) -> Tuple[Callable, Optional[Callable], ModelMetadata]:
        """Extract model creation functions and metadata

        Args:
            module: Loaded module

        Returns:
            Tuple of (create_model, create_dataloader, metadata)
        """
        create_model = module.create_model
        create_dataloader = getattr(module, "create_dataloader", None)
        metadata = ModelMetadata(module.MODEL_METADATA)

        return create_model, create_dataloader, metadata

    def load_model(
        self, model_source: str, model_id: str, force_download: bool = False
    ) -> Tuple[Callable, Optional[Callable], ModelMetadata]:
        """Load model from various sources

        Args:
            model_source: Source specification:
                - HTTP(S) URL: "https://example.com/model.py"
                - GCS URL: "gs://bucket/path/to/model.py"
                - Local file: "/path/to/model.py"
            model_id: Model ID
            force_download: Force re-download

        Returns:
            Tuple of (create_model, create_dataloader, metadata)
        """
        logger.info(f"Loading model from: {model_source}")

        # Determine source type and download/load
        if model_source.startswith("gs://"):
            # GCS source
            parts = model_source[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid GCS URL: {model_source}")

            bucket_name, blob_name = parts
            model_file = self.download_model_from_gcs(
                bucket_name=bucket_name,
                blob_name=blob_name,
                model_id=model_id,
                force_download=force_download,
            )
        elif model_source.startswith("http://") or model_source.startswith("https://"):
            # HTTP(S) source
            model_file = self.download_model_from_url(
                url=model_source, model_id=model_id, force_download=force_download
            )
        else:
            # Local file
            model_file = Path(model_source)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load module
        module = self.load_model_module(model_file, model_id)

        # Validate module
        self.validate_model_module(module)

        # Extract functions and metadata
        create_model, create_dataloader, metadata = self.get_model_functions(module)

        logger.info(f"Model loaded successfully: {metadata['name']} v{metadata['version']}")

        return create_model, create_dataloader, metadata

    def clear_cache(self) -> None:
        """Clear cached modules and files"""
        # Clear loaded modules
        self._loaded_modules.clear()

        # Optionally clear downloaded files
        logger.info("Cache cleared")


def create_model_loader(models_dir: Path) -> ModelLoader:
    """Create model loader instance

    Args:
        models_dir: Directory for model storage

    Returns:
        ModelLoader instance
    """
    return ModelLoader(models_dir=models_dir)
