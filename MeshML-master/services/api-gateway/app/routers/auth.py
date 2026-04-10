"""
Authentication router - User registration, login, JWT tokens

Endpoints:
- POST /api/auth/register - Create new user account
- POST /api/auth/login - Login and get JWT token
- GET /api/auth/me - Get current user info
- POST /api/auth/refresh - Refresh JWT token
"""

import logging
from datetime import datetime, timedelta

from app.models.user import User
from app.schemas.auth import TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse, UserChangePasswordRequest
from app.utils.database import get_db
from app.utils.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register new user account

    Args:
        request: User registration data (email, password, name)
        db: Database session

    Returns:
        Created user information

    Raises:
        400: Email already registered
    """
    logger.info(f"Registering new user: {request.email}")

    # Check if email already exists
    result = await db.execute(select(User).where(User.email == request.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    hashed_password = hash_password(request.password)
    user = User(email=request.email, hashed_password=hashed_password, full_name=request.full_name)

    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info(f"User registered successfully: {user.id}")
    return user


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Login and get JWT access token

    Args:
        request: Login credentials (email, password)
        db: Database session

    Returns:
        JWT access token and user info

    Raises:
        401: Invalid credentials
    """
    logger.info(f"Login attempt: {request.email}")

    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        logger.warning(f"Failed login attempt: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Generate access token
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})

    logger.info(f"User logged in successfully: {user.id}")

    return TokenResponse(
        access_token=access_token, token_type="bearer", user=UserResponse.from_orm(user)
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency to get current authenticated user

    Args:
        credentials: JWT token from Authorization header
        db: Database session

    Returns:
        Current user

    Raises:
        401: Invalid or expired token
    """
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )

    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )

    # Get user from database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user information

    Args:
        current_user: Injected current user

    Returns:
        User information
    """
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(current_user: User = Depends(get_current_user)):
    """
    Refresh JWT access token

    Args:
        current_user: Current authenticated user

    Returns:
        New JWT access token
    """
    # Generate new access token
    access_token = create_access_token(
        data={"sub": str(current_user.id), "email": current_user.email}
    )

    return TokenResponse(
        access_token=access_token, token_type="bearer", user=UserResponse.from_orm(current_user)
    )


@router.post("/password", status_code=status.HTTP_200_OK)
async def change_password(
    request: UserChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Change user password
    """
    # Verify old password
    if not verify_password(request.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect old password"
        )
    
    # Hash new password
    hashed_password = hash_password(request.new_password)
    
    # Update user in db
    current_user.hashed_password = hashed_password
    db.add(current_user)
    await db.commit()
    
    return {"message": "Password updated successfully"}
