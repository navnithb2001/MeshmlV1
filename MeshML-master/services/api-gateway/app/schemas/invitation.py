"""Invitation schemas"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CreateInvitationRequest(BaseModel):
    """Create invitation request"""

    max_uses: Optional[int] = Field(None, ge=1)
    expires_in_hours: int = Field(24, ge=1, le=168)  # 1 hour to 1 week


class InvitationResponse(BaseModel):
    """Invitation response"""

    code: str
    group_id: uuid.UUID
    max_uses: Optional[int]
    current_uses: int
    expires_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AcceptInvitationRequest(BaseModel):
    """Accept invitation request"""

    worker_id: str
    invitation_code: str
