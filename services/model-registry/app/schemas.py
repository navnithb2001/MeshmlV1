"""
Pydantic schemas for Model Registry API
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


class ModelState(str, Enum):
    """Model lifecycle states"""

    UPLOADING = "uploading"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class ModelMetadata(BaseModel):
    """Model metadata structure"""

    architecture_type: Optional[str] = None
    dataset_type: Optional[str] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    num_parameters: Optional[int] = None
    framework_version: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ModelCreate(BaseModel):
    """Schema for creating a new model"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    group_id: UUID = Field(...)  # Changed to UUID
    architecture_type: Optional[str] = Field(None, max_length=100)
    dataset_type: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None
    version: str = Field(default="1.0.0", max_length=50)
    parent_model_id: Optional[int] = None

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ModelUpdate(BaseModel):
    """Schema for updating model metadata"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    architecture_type: Optional[str] = Field(None, max_length=100)
    dataset_type: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    """Schema for model response"""

    id: int
    name: str
    description: Optional[str]
    group_id: UUID  # Changed to UUID
    created_by_user_id: UUID  # Changed to UUID
    gcs_path: Optional[str]
    file_size_bytes: Optional[int]
    file_hash: Optional[str]
    state: ModelState
    validation_message: Optional[str]
    architecture_type: Optional[str]
    dataset_type: Optional[str]
    framework: str
    model_metadata: Optional[Dict[str, Any]] = None  # Made optional with default
    version: str
    parent_model_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    deprecated_at: Optional[datetime]
    usage_count: int
    download_count: int

    class Config:
        from_attributes = True


class ModelListResponse(BaseModel):
    """Schema for paginated model list"""

    models: List[ModelResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class ModelSearchQuery(BaseModel):
    """Schema for model search"""

    query: Optional[str] = None  # Search in name, description
    group_id: Optional[UUID] = None  # Changed to UUID
    state: Optional[ModelState] = None
    architecture_type: Optional[str] = None
    dataset_type: Optional[str] = None
    created_by_user_id: Optional[UUID] = None  # Changed to UUID
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ModelStateUpdate(BaseModel):
    """Schema for model state transition"""

    state: ModelState
    validation_message: Optional[str] = None


class ModelVersionCreate(BaseModel):
    """Schema for creating a new model version"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    parent_model_id: int = Field(..., gt=0)
    version: str = Field(..., max_length=50)
    metadata: Optional[Dict[str, Any]] = None

    @validator("version")
    def validate_version(cls, v):
        # Simple semantic version check
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError("Version must be in format X.Y.Z")
        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be integers")
        return v


class ModelUsageStats(BaseModel):
    """Schema for model usage statistics"""

    model_id: int
    model_name: str
    total_jobs: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_downloads: int
    first_used_at: Optional[datetime]
    last_used_at: Optional[datetime]


class UploadUrlResponse(BaseModel):
    """Schema for signed upload URL"""

    upload_url: str
    model_id: int
    gcs_path: str
    expires_in_seconds: int = 3600
