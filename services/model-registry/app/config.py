import os
from typing import Any, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # Service
    SERVICE_NAME: str = "model-registry"
    DEBUG: bool = True

    # Database
    # Build DATABASE_URL from components if not provided
    DATABASE_URL: str = os.getenv("DATABASE_URL") or (
        f"postgresql+asyncpg://"
        f"{os.getenv('POSTGRES_USER', 'meshml_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'CHANGE_ME_IN_PRODUCTION')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'meshml')}"
    )

    # GCS Storage
    GCS_BUCKET_NAME: str = "meshml-models"
    GCS_PROJECT_ID: Optional[str] = None
    GCS_CREDENTIALS_PATH: Optional[str] = None

    # Model Storage Structure
    MODEL_STORAGE_PREFIX: str = "models"  # models/{model_id}/model.py

    # Model Validation
    MAX_MODEL_SIZE_MB: int = 100  # Max model file size
    ALLOWED_FILE_EXTENSIONS: list = [".py"]

    # Model States
    MODEL_STATE_UPLOADING: str = "uploading"
    MODEL_STATE_VALIDATING: str = "validating"
    MODEL_STATE_READY: str = "ready"
    MODEL_STATE_FAILED: str = "failed"
    MODEL_STATE_DEPRECATED: str = "deprecated"

    # Search & Discovery
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # Versioning
    ENABLE_MODEL_VERSIONING: bool = True
    MAX_VERSION_HISTORY: int = 10  # Keep last 10 versions

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> bool:
        """Accept common env values like 'release', 'true', 'false', etc."""
        if isinstance(value, bool):
            return value
        if value is None:
            return False

        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on", "debug"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", "release", "prod", "production"}:
            return False

        # Safe default for unexpected values.
        return False


settings = Settings()
