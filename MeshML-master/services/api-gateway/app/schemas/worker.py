"""Worker schemas"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr


class WorkerRegisterRequest(BaseModel):
    """Worker registration request"""

    worker_id: str
    user_email: Optional[EmailStr] = None
    capabilities: Dict[str, Any]
    status: str = "idle"


class WorkerResponse(BaseModel):
    """Worker response"""

    id: uuid.UUID
    worker_id: str
    user_email: Optional[str]
    capabilities: Optional[Dict[str, Any]]
    status: str
    last_heartbeat: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class WorkerUpdateCapabilitiesRequest(BaseModel):
    """Update worker capabilities request"""

    capabilities: Dict[str, Any]
