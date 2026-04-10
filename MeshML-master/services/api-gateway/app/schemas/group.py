"""Group schemas"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from app.schemas.auth import UserResponse


class GroupCreateRequest(BaseModel):
    """Create group request"""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: bool = False


class GroupUpdateRequest(BaseModel):
    """Update group details request"""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: Optional[bool] = None


class GroupResponse(BaseModel):
    """Group response"""

    id: uuid.UUID
    name: str
    description: Optional[str]
    is_public: bool
    owner_id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


class GroupMemberResponse(BaseModel):
    """Group member response"""

    id: uuid.UUID
    group_id: uuid.UUID
    user_id: Optional[uuid.UUID]
    worker_id: Optional[str]
    role: str
    joined_at: datetime
    user: Optional["UserResponse"] = None

    class Config:
        from_attributes = True


class JoinGroupRequest(BaseModel):
    """Join group request"""

    worker_id: str


class UpdateMemberRoleRequest(BaseModel):
    """Update member role request"""

    role: str = Field(..., pattern="^(owner|admin|member|worker)$")
