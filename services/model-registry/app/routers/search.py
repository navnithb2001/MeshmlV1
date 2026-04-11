"""
Search router - Model discovery and filtering
Implements TASK-11.3 (model search & discovery)
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..models import Model, ModelState, ModelUsage
from ..schemas import ModelListResponse, ModelResponse, ModelSearchQuery, ModelUsageStats

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models", response_model=ModelListResponse)
async def search_models(
    query: Optional[str] = Query(None, description="Search in name and description"),
    group_id: Optional[int] = Query(None, description="Filter by group"),
    state: Optional[ModelState] = Query(None, description="Filter by state"),
    architecture_type: Optional[str] = Query(None, description="Filter by architecture"),
    dataset_type: Optional[str] = Query(None, description="Filter by dataset type"),
    created_by_user_id: Optional[int] = Query(None, description="Filter by creator"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db_session),
):
    """
    Search and filter models with pagination
    Supports text search and multiple filters
    """
    # Build base query
    stmt = select(Model)
    conditions = []

    # Text search in name and description
    if query:
        search_term = f"%{query}%"
        conditions.append(or_(Model.name.ilike(search_term), Model.description.ilike(search_term)))

    # Filter by group
    if group_id is not None:
        conditions.append(Model.group_id == group_id)

    # Filter by state
    if state is not None:
        conditions.append(Model.state == state)

    # Filter by architecture type
    if architecture_type:
        conditions.append(Model.architecture_type == architecture_type)

    # Filter by dataset type
    if dataset_type:
        conditions.append(Model.dataset_type == dataset_type)

    # Filter by creator
    if created_by_user_id is not None:
        conditions.append(Model.created_by_user_id == created_by_user_id)

    # Apply all conditions
    if conditions:
        stmt = stmt.where(and_(*conditions))

    # Get total count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar()

    # Apply pagination and ordering
    offset = (page - 1) * page_size
    stmt = stmt.order_by(Model.created_at.desc()).limit(page_size).offset(offset)

    # Execute query
    result = await db.execute(stmt)
    models = result.scalars().all()

    # Calculate pagination metadata
    has_next = (offset + page_size) < total
    has_prev = page > 1

    logger.info(f"Search returned {len(models)} models (page {page}, total {total})")

    return ModelListResponse(
        models=models,
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
        has_prev=has_prev,
    )


@router.get("/groups/{group_id}/models", response_model=List[ModelResponse])
async def list_group_models(
    group_id: int,
    state: Optional[ModelState] = Query(None, description="Filter by state"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db_session),
):
    """
    List all models belonging to a specific group
    Ordered by creation date (newest first)
    """
    stmt = select(Model).where(Model.group_id == group_id)

    if state is not None:
        stmt = stmt.where(Model.state == state)

    stmt = stmt.order_by(Model.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    models = result.scalars().all()

    logger.info(f"Listed {len(models)} models for group {group_id}")

    return models


@router.get("/users/{user_id}/models", response_model=List[ModelResponse])
async def list_user_models(
    user_id: int,
    state: Optional[ModelState] = Query(None, description="Filter by state"),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db_session),
):
    """List all models created by a specific user"""
    stmt = select(Model).where(Model.created_by_user_id == user_id)

    if state is not None:
        stmt = stmt.where(Model.state == state)

    stmt = stmt.order_by(Model.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    models = result.scalars().all()

    logger.info(f"Listed {len(models)} models for user {user_id}")

    return models


@router.get("/architecture-types")
async def list_architecture_types(db: AsyncSession = Depends(get_db_session)):
    """Get all unique architecture types in the registry"""
    result = await db.execute(
        select(Model.architecture_type)
        .distinct()
        .where(Model.architecture_type.isnot(None))
        .order_by(Model.architecture_type)
    )

    types = result.scalars().all()

    return {"architecture_types": types}


@router.get("/dataset-types")
async def list_dataset_types(db: AsyncSession = Depends(get_db_session)):
    """Get all unique dataset types in the registry"""
    result = await db.execute(
        select(Model.dataset_type)
        .distinct()
        .where(Model.dataset_type.isnot(None))
        .order_by(Model.dataset_type)
    )

    types = result.scalars().all()

    return {"dataset_types": types}


@router.get("/models/{model_id}/usage", response_model=ModelUsageStats)
async def get_model_usage(model_id: int, db: AsyncSession = Depends(get_db_session)):
    """
    Get usage statistics for a model
    Shows which jobs are using this model
    """
    # Verify model exists
    model_result = await db.execute(select(Model).where(Model.id == model_id))
    model = model_result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    # Get usage records
    usage_result = await db.execute(select(ModelUsage).where(ModelUsage.model_id == model_id))
    usage_records = usage_result.scalars().all()

    # Calculate statistics
    total_jobs = len(usage_records)
    active_jobs = len([u for u in usage_records if u.completed_at is None])
    completed_jobs = len([u for u in usage_records if u.completed_at is not None])
    failed_jobs = 0  # Would need job status to determine

    first_used = min([u.started_at for u in usage_records]) if usage_records else None
    last_used = max([u.started_at for u in usage_records]) if usage_records else None

    return ModelUsageStats(
        model_id=model_id,
        model_name=model.name,
        total_jobs=total_jobs,
        active_jobs=active_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        total_downloads=model.download_count,
        first_used_at=first_used,
        last_used_at=last_used,
    )


@router.get("/popular")
async def get_popular_models(
    limit: int = Query(10, ge=1, le=50), db: AsyncSession = Depends(get_db_session)
):
    """
    Get most popular models by usage count
    Only includes READY models
    """
    result = await db.execute(
        select(Model)
        .where(Model.state == ModelState.READY)
        .order_by(Model.usage_count.desc(), Model.download_count.desc())
        .limit(limit)
    )

    models = result.scalars().all()

    return {
        "popular_models": [
            {
                "id": m.id,
                "name": m.name,
                "usage_count": m.usage_count,
                "download_count": m.download_count,
                "architecture_type": m.architecture_type,
                "created_at": m.created_at,
            }
            for m in models
        ]
    }


@router.get("/recent")
async def get_recent_models(
    limit: int = Query(10, ge=1, le=50),
    state: Optional[ModelState] = Query(ModelState.READY),
    db: AsyncSession = Depends(get_db_session),
):
    """Get recently created models"""
    stmt = select(Model).order_by(Model.created_at.desc()).limit(limit)

    if state is not None:
        stmt = stmt.where(Model.state == state)

    result = await db.execute(stmt)
    models = result.scalars().all()

    return {
        "recent_models": [
            {
                "id": m.id,
                "name": m.name,
                "state": m.state,
                "architecture_type": m.architecture_type,
                "created_at": m.created_at,
            }
            for m in models
        ]
    }
