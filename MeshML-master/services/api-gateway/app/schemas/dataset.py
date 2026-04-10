"""Dataset schemas"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl


class DatasetUploadResponse(BaseModel):
    """Response after dataset upload"""

    dataset_id: str
    name: str
    format: str
    status: str
    total_size_bytes: Optional[int] = None
    file_count: Optional[int] = None
    num_samples: Optional[int] = None
    num_classes: Optional[int] = None
    local_path: Optional[str] = None
    gcs_path: Optional[str] = None
    message: str
    uploaded_at: str


class DatasetFromURLRequest(BaseModel):
    """Request to download dataset from URL"""

    url: str
    name: Optional[str] = None
    format: Optional[str] = None  # imagefolder, coco, csv

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://drive.google.com/file/d/1abc123xyz/view",
                "name": "My CIFAR-10 Dataset",
                "format": "imagefolder",
            }
        }


class DatasetResponse(BaseModel):
    """Dataset information response"""

    id: str
    name: str
    format: str
    upload_type: str
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    gcs_path: Optional[str] = None
    total_size_bytes: Optional[int] = None
    file_count: Optional[int] = None
    num_samples: Optional[int] = None
    num_classes: Optional[int] = None
    num_shards: Optional[int] = None
    shard_strategy: Optional[str] = None
    sharded_at: Optional[datetime] = None
    status: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    uploaded_by: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """List of datasets response"""

    datasets: List[DatasetResponse]
    total: int
