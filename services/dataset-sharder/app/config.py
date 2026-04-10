"""
Configuration for Dataset Sharder Service
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Service info
    SERVICE_NAME: str = "dataset-sharder"
    SERVICE_PORT: int = 8001

    # Storage settings
    LOCAL_STORAGE_PATH: str = os.getenv("LOCAL_STORAGE_PATH", "/app/datasets")
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "meshml-datasets")
    # Backward-compatible aliases used by older storage helpers.
    GCS_BUCKET_DATASETS: str = os.getenv(
        "GCS_BUCKET_DATASETS", os.getenv("GCS_BUCKET_NAME", "meshml-datasets")
    )
    GCS_BUCKET_MODELS: str = os.getenv(
        "GCS_BUCKET_MODELS", os.getenv("GCS_BUCKET_NAME", "meshml-models")
    )
    GCS_BUCKET_ARTIFACTS: str = os.getenv(
        "GCS_BUCKET_ARTIFACTS", os.getenv("GCS_BUCKET_NAME", "meshml-artifacts")
    )
    GCS_PROJECT_ID: str = os.getenv("GCS_PROJECT_ID", "")
    USE_GCS: bool = os.getenv("USE_GCS", "false").lower() == "true"

    # Sharding settings
    DEFAULT_BATCH_SIZE: int = 32
    MAX_SHARDS: int = 1000

    # Distribution settings
    CHUNK_SIZE: int = 8192  # 8KB chunks for streaming
    MAX_CONCURRENT_DOWNLOADS: int = 10

    # Database (for batch metadata)
    # Build DATABASE_URL from components if not explicitly provided
    DATABASE_URL: str = os.getenv("DATABASE_URL") or (
        f"postgresql+asyncpg://"
        f"{os.getenv('POSTGRES_USER', 'meshml_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'CHANGE_ME_IN_PRODUCTION')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'meshml')}"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
