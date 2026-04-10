"""Pydantic schemas package"""

from .auth import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse
from .group import (
    GroupCreateRequest,
    GroupMemberResponse,
    GroupResponse,
    JoinGroupRequest,
    UpdateMemberRoleRequest,
)
from .invitation import AcceptInvitationRequest, CreateInvitationRequest, InvitationResponse
from .job import JobCreateRequest, JobProgressResponse, JobResponse
from .worker import WorkerRegisterRequest, WorkerResponse, WorkerUpdateCapabilitiesRequest

__all__ = [
    # Auth
    "UserRegisterRequest",
    "UserLoginRequest",
    "UserResponse",
    "TokenResponse",
    # Group
    "GroupCreateRequest",
    "GroupResponse",
    "GroupMemberResponse",
    "JoinGroupRequest",
    "UpdateMemberRoleRequest",
    # Invitation
    "CreateInvitationRequest",
    "InvitationResponse",
    "AcceptInvitationRequest",
    # Worker
    "WorkerRegisterRequest",
    "WorkerResponse",
    "WorkerUpdateCapabilitiesRequest",
    # Job
    "JobCreateRequest",
    "JobResponse",
    "JobProgressResponse",
]
