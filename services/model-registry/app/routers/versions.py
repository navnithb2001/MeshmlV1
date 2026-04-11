"""
Versions router - Model versioning management
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..models import Model
from ..schemas import ModelResponse, ModelVersionCreate
from ..versioning import VersionManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Test user UUID (in production, this would come from authentication)
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model_version(
    version_data: ModelVersionCreate, db: AsyncSession = Depends(get_db_session)
):
    """
    Create a new version of an existing model
    Parent model must be in READY state
    """
    version_manager = VersionManager(db)

    try:
        model = await version_manager.create_version(
            parent_model_id=version_data.parent_model_id,
            name=version_data.name,
            version=version_data.version,
            created_by_user_id=TEST_USER_ID,  # Using test user UUID
            description=version_data.description,
            metadata=version_data.metadata,
        )

        logger.info(f"Created version {model.version} of model {version_data.parent_model_id}")

        return model

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{parent_model_id}", response_model=List[ModelResponse])
async def get_model_versions(
    parent_model_id: int,
    include_deprecated: bool = False,
    db: AsyncSession = Depends(get_db_session),
):
    """Get all versions of a model"""
    version_manager = VersionManager(db)

    versions = await version_manager.get_versions(parent_model_id, include_deprecated)

    return versions


@router.get("/{parent_model_id}/latest", response_model=Optional[ModelResponse])
async def get_latest_version(parent_model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Get the latest ready version of a model"""
    version_manager = VersionManager(db)

    latest = await version_manager.get_latest_version(parent_model_id)

    if not latest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ready versions found for model {parent_model_id}",
        )

    return latest


@router.get("/{model_id}/history", response_model=List[ModelResponse])
async def get_version_history(model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Get full version history (all ancestors and descendants)"""
    version_manager = VersionManager(db)

    try:
        history = await version_manager.get_version_history(model_id)
        return history
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/{parent_model_id}/count")
async def get_version_count(parent_model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Get count of versions for a model"""
    version_manager = VersionManager(db)

    count = await version_manager.get_version_count(parent_model_id)

    return {"parent_model_id": parent_model_id, "version_count": count}


@router.get("/{parent_model_id}/suggest-next")
async def suggest_next_version(parent_model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Suggest the next version number based on existing versions"""
    version_manager = VersionManager(db)

    next_version = await version_manager.suggest_next_version(parent_model_id)

    return {"parent_model_id": parent_model_id, "suggested_version": next_version}


@router.post("/increment")
async def increment_version(current_version: str, part: str = "patch"):  # major, minor, or patch
    """
    Utility endpoint to increment a version number
    Useful for clients to calculate next version
    """
    version_manager = VersionManager(None)  # No DB needed for this utility

    try:
        new_version = version_manager.increment_version(current_version, part)
        return {
            "current_version": current_version,
            "incremented_part": part,
            "new_version": new_version,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
