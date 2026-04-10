"""Job schemas"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class JobCreateRequest(BaseModel):
    """Create job request"""

    group_id: uuid.UUID
    model_id: Optional[str] = None
    dataset_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class JobResponse(BaseModel):
    """Job response"""

    id: uuid.UUID
    group_id: uuid.UUID
    model_id: Optional[str]
    dataset_id: Optional[str]
    config: Optional[Dict[str, Any]]
    status: str
    progress: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_by: uuid.UUID
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class JobProgressResponse(BaseModel):
    """Job training progress"""

    job_id: uuid.UUID
    status: str
    current_epoch: int
    total_epochs: int
    current_batch: int
    total_batches: int
    loss: float
    accuracy: float
    worker_count: int
