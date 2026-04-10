"""
Configuration management for MeshML Worker
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParameterServerConfig(BaseModel):
    """Parameter Server configuration"""

    url: str = Field(default="http://localhost:8003", description="Parameter Server HTTP URL")
    grpc_url: str = Field(default="localhost:50052", description="Parameter Server gRPC URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")


class DatasetSharderConfig(BaseModel):
    """Dataset Sharder configuration"""

    url: str = Field(default="http://localhost:8001", description="Dataset Sharder URL")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")


class TaskOrchestratorConfig(BaseModel):
    """Task Orchestrator configuration"""

    grpc_url: str = Field(default="localhost:50051", description="Task Orchestrator gRPC URL")
    user_id: str = Field(default="", description="User ID who owns this worker")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")


class ModelRegistryConfig(BaseModel):
    """Model Registry configuration"""

    url: str = Field(default="http://localhost:8004", description="Model Registry URL")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")


class MetricsServiceConfig(BaseModel):
    """Metrics Service configuration"""

    grpc_url: str = Field(default="localhost:50055", description="Metrics Service gRPC URL")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")


class WorkerIdentityConfig(BaseModel):
    """Worker identity configuration"""

    id: Optional[str] = Field(default=None, description="Worker ID (auto-generated if not set)")
    name: str = Field(default="MeshML Worker", description="Worker name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Worker tags")


class TrainingConfig(BaseModel):
    """Training configuration"""

    device: str = Field(default="auto", description="Device: auto, cuda, cpu, mps")
    batch_size: int = Field(default=32, description="Training batch size")
    num_workers: int = Field(default=4, description="DataLoader worker processes")
    mixed_precision: bool = Field(default=True, description="Enable mixed precision (FP16)")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_workers must be non-negative")
        return v


class StorageConfig(BaseModel):
    """Storage configuration"""

    base_dir: Path = Field(default=Path(".meshml"), description="Base directory")
    checkpoints_dir: Optional[Path] = Field(default=None, description="Checkpoints directory")
    models_dir: Optional[Path] = Field(default=None, description="Models directory")
    data_dir: Optional[Path] = Field(default=None, description="Data directory")

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Set default subdirectories
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.base_dir / "checkpoints"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"

    def create_directories(self) -> None:
        """Create storage directories if they don't exist"""
        for dir_path in [self.base_dir, self.checkpoints_dir, self.models_dir, self.data_dir]:
            if dir_path:
                dir_path.mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format"
    )
    file: Optional[Path] = Field(default=None, description="Log file path")


class WorkerConfig(BaseModel):
    """Complete worker configuration"""

    model_config = ConfigDict(protected_namespaces=())

    api_base_url: str = Field(default="http://localhost:8000", description="API Gateway URL")
    worker: WorkerIdentityConfig = Field(default_factory=WorkerIdentityConfig)
    parameter_server: ParameterServerConfig = Field(default_factory=ParameterServerConfig)
    dataset_sharder: DatasetSharderConfig = Field(default_factory=DatasetSharderConfig)
    task_orchestrator: TaskOrchestratorConfig = Field(default_factory=TaskOrchestratorConfig)
    model_registry: ModelRegistryConfig = Field(default_factory=ModelRegistryConfig)
    metrics_service: MetricsServiceConfig = Field(default_factory=MetricsServiceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, config_path: Path) -> "WorkerConfig":
        """Load configuration from YAML file

        Args:
            config_path: Path to configuration file

        Returns:
            WorkerConfig instance
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file

        Args:
            config_path: Path to save configuration
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        config_dict = self.model_dump()

        def convert_paths(obj: Any) -> Any:
            """Convert Path objects to strings"""
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_config_path(self) -> Path:
        """Get default configuration file path

        Returns:
            Path to config file
        """
        return self.storage.base_dir / "config.yaml"

    def setup(self) -> None:
        """Setup worker (create directories, etc.)"""
        self.storage.create_directories()

        # Generate worker ID if not set
        if self.worker.id is None:
            import uuid

            self.worker.id = f"worker-{uuid.uuid4().hex[:8]}"


def get_default_config() -> WorkerConfig:
    """Get default worker configuration

    Returns:
        Default WorkerConfig instance
    """
    return WorkerConfig()


def load_config(config_path: Optional[Path] = None) -> WorkerConfig:
    """Load worker configuration

    Args:
        config_path: Optional path to config file. If None, uses default location.

    Returns:
        WorkerConfig instance
    """
    if config_path is None:
        # Try home directory first, then current directory
        home_config = Path.home() / ".meshml" / "config.yaml"
        local_config = Path(".meshml") / "config.yaml"

        if home_config.exists():
            config_path = home_config
        elif local_config.exists():
            config_path = local_config
        else:
            return get_default_config()

    return WorkerConfig.from_file(config_path)
