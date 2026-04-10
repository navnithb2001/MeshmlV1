"""
Group invitation endpoints

Endpoints:
- POST /api/groups/{group_id}/invitations - Create invitation
- POST /api/invitations/accept - Accept invitation
- GET /api/invitations/{code} - Get invitation details
- DELETE /api/invitations/{code} - Revoke invitation
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from app.models.group import Group, GroupInvitation, GroupMember
from app.models.user import User
from app.models.worker import Worker
from app.routers.auth import get_current_user
from app.schemas.invitation import (
    AcceptInvitationRequest,
    CreateInvitationRequest,
    InvitationResponse,
)
from app.utils.database import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


def generate_invitation_code() -> str:
    """Generate unique invitation code"""
    return f"inv_{secrets.token_urlsafe(16)}"


@router.post("/{group_id}/invitations", response_model=InvitationResponse)
async def create_invitation(
    group_id: str,
    request: CreateInvitationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create invitation for group

    Args:
        group_id: Group ID
        request: Invitation parameters (max_uses, expires_in_hours)
        current_user: Authenticated user
        db: Database session

    Returns:
        Invitation code and details

    Raises:
        403: Not authorized (only owner/admin)
        404: Group not found
    """
    logger.info(f"Creating invitation for group {group_id}")

    # Check if user is owner/admin
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == group_id, GroupMember.user_id == current_user.id
        )
    )
    member = member_result.scalar_one_or_none()

    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can create invitations",
        )

    # Generate invitation
    code = generate_invitation_code()
    expires_at = datetime.utcnow() + timedelta(hours=request.expires_in_hours)

    invitation = GroupInvitation(
        code=code,
        group_id=group_id,
        created_by=current_user.id,
        max_uses=request.max_uses,
        expires_at=expires_at,
    )

    db.add(invitation)
    await db.commit()
    await db.refresh(invitation)

    logger.info(f"Invitation created: {code}")

    return InvitationResponse(
        code=code,
        group_id=group_id,
        max_uses=request.max_uses,
        current_uses=0,
        expires_at=expires_at,
        created_at=invitation.created_at,
    )


@router.post("/accept", response_model=dict)
async def accept_invitation(
    request: AcceptInvitationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept group invitation (requires authentication)

    Worker uses their user credentials to authenticate, then joins the group
    with both their user_id (for dashboard access) and worker_id (for training tasks).
    This links the user account to the worker device.

    Args:
        request: Invitation code and worker_id
        current_user: Authenticated user (from login)
        db: Database session

    Returns:
        Group information

    Raises:
        404: Invalid invitation code
        400: Invitation expired or used up
        401: Not authenticated
    """
    logger.info(
        f"User {current_user.email} (worker {request.worker_id}) accepting invitation: {request.invitation_code[:12]}..."
    )

    # Get invitation
    result = await db.execute(
        select(GroupInvitation).where(GroupInvitation.code == request.invitation_code)
    )
    invitation = result.scalar_one_or_none()

    if not invitation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invalid invitation code")

    # Check expiration
    if invitation.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invitation has expired"
        )

    # Check usage limit
    if invitation.max_uses and invitation.current_uses >= invitation.max_uses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invitation has reached maximum uses"
        )

    # Get group
    group_result = await db.execute(select(Group).where(Group.id == invitation.group_id))
    group = group_result.scalar_one_or_none()

    if not group:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found")

    # Check if already member (by user_id or worker_id)
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == invitation.group_id,
            (GroupMember.user_id == current_user.id) | (GroupMember.worker_id == request.worker_id),
        )
    )
    existing_member = member_result.scalar_one_or_none()

    if existing_member:
        logger.info(
            f"User {current_user.email} (worker {request.worker_id}) already member of group"
        )
        return {
            "group_id": invitation.group_id,
            "group_name": group.name,
            "role": existing_member.role,
            "joined_at": existing_member.joined_at.isoformat(),
        }

    # Add as member with both user_id and worker_id
    # This links the user account (for dashboard) to the worker device (for training)
    member = GroupMember(
        group_id=invitation.group_id,
        user_id=current_user.id,  # User account for authentication and dashboard access
        worker_id=request.worker_id,  # Worker device for training tasks
        role="worker",
    )

    db.add(member)

    # Ensure worker exists in workers table as a placeholder if not already registered
    worker_result = await db.execute(select(Worker).where(Worker.worker_id == request.worker_id))
    worker = worker_result.scalar_one_or_none()

    if not worker:
        worker = Worker(
            worker_id=request.worker_id,
            user_email=current_user.email,
            status="offline",
        )
        db.add(worker)

    # Increment usage count
    invitation.current_uses += 1

    await db.commit()

    logger.info(
        f"User {current_user.email} (worker {request.worker_id}) joined group {invitation.group_id} via invitation"
    )

    return {
        "group_id": invitation.group_id,
        "group_name": group.name,
        "role": "worker",
        "joined_at": datetime.utcnow().isoformat(),
    }


@router.get("/{code}", response_model=InvitationResponse)
async def get_invitation_details(code: str, db: AsyncSession = Depends(get_db)):
    """
    Get invitation details

    Args:
        code: Invitation code
        db: Database session

    Returns:
        Invitation information

    Raises:
        404: Invitation not found
    """
    result = await db.execute(select(GroupInvitation).where(GroupInvitation.code == code))
    invitation = result.scalar_one_or_none()

    if not invitation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    return InvitationResponse(
        code=invitation.code,
        group_id=invitation.group_id,
        max_uses=invitation.max_uses,
        current_uses=invitation.current_uses,
        expires_at=invitation.expires_at,
        created_at=invitation.created_at,
    )


@router.delete("/{code}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invitation(
    code: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Revoke invitation

    Args:
        code: Invitation code
        current_user: Authenticated user
        db: Database session

    Raises:
        403: Not authorized
        404: Invitation not found
    """
    # Get invitation
    result = await db.execute(select(GroupInvitation).where(GroupInvitation.code == code))
    invitation = result.scalar_one_or_none()

    if not invitation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    # Check if user is owner/admin
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == invitation.group_id, GroupMember.user_id == current_user.id
        )
    )
    member = member_result.scalar_one_or_none()

    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can revoke invitations",
        )

    await db.delete(invitation)
    await db.commit()

    logger.info(f"Invitation revoked: {code}")
