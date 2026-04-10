"""Authentication schemas"""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """User registration request"""

    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLoginRequest(BaseModel):
    """User login request"""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response"""

    id: uuid.UUID
    email: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True  # Pydantic v2


class TokenResponse(BaseModel):
    """JWT token response"""

    access_token: str
    token_type: str
    user: UserResponse


class UserChangePasswordRequest(BaseModel):
    """Request to change password"""

    old_password: str
    new_password: str = Field(..., min_length=8)
