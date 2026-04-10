"""
Worker registration and management endpoints

Endpoints:
- POST /api/workers/register - Register new worker
- GET /api/workers - List all workers
- GET /api/workers/{worker_id} - Get worker details
- PUT /api/workers/{worker_id}/capabilities - Update worker capabilities
- DELETE /api/workers/{worker_id} - Deregister worker
- POST /api/workers/{worker_id}/heartbeat - Manual heartbeat
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from app.models.group import GroupMember
from app.models.worker import Worker
from app.schemas.worker import (
    WorkerRegisterRequest,
    WorkerResponse,
    WorkerUpdateCapabilitiesRequest,
)
from app.utils.database import get_db
from app.utils.security import create_access_token
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
async def register_worker(request: WorkerRegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Register new worker

    Args:
        request: Worker registration data
        db: Database session

    Returns:
        Worker ID and authentication token
    """
    logger.info(f"Registering worker: {request.worker_id}")

    # Check if worker already exists
    result = await db.execute(select(Worker).where(Worker.worker_id == request.worker_id))
    existing_worker = result.scalar_one_or_none()

    if existing_worker:
        # Update existing worker
        existing_worker.user_email = request.user_email
        existing_worker.capabilities = request.capabilities
        existing_worker.status = request.status
        existing_worker.last_heartbeat = datetime.utcnow()

        await db.commit()
        await db.refresh(existing_worker)

        worker = existing_worker
        logger.info(f"Worker updated: {worker.worker_id}")
    else:
        # Create new worker
        worker = Worker(
            worker_id=request.worker_id,
            user_email=request.user_email,
            capabilities=request.capabilities,
            status=request.status,
            last_heartbeat=datetime.utcnow(),
        )

        db.add(worker)
        await db.commit()
        await db.refresh(worker)

        logger.info(f"Worker registered: {worker.worker_id}")

    # Generate auth token
    auth_token = create_access_token(
        data={"sub": worker.worker_id, "type": "worker", "email": request.user_email}
    )

    return {
        "worker_id": worker.worker_id,
        "auth_token": auth_token,
        "registered_at": worker.created_at.isoformat(),
    }


@router.get("", response_model=List[WorkerResponse])
async def list_workers(status: Optional[str] = None, group_id: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    """
    List all registered workers

    Args:
        status: Filter by status (idle/training/offline)
        group_id: Filter by group ID
        db: Database session

    Returns:
        List of workers
    """
    query = select(Worker)

    if group_id:
        query = query.join(
            GroupMember, GroupMember.worker_id == Worker.worker_id
        ).where(GroupMember.group_id == group_id)

    if status:
        query = query.where(Worker.status == status)

    result = await db.execute(query)
    workers = result.scalars().all()

    logger.info(f"Retrieved {len(workers)} workers")
    return workers


@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(worker_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get worker details by ID

    Args:
        worker_id: Worker ID
        db: Database session

    Returns:
        Worker information

    Raises:
        404: Worker not found
    """
    result = await db.execute(select(Worker).where(Worker.worker_id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker {worker_id} not found"
        )

    return worker


@router.put("/{worker_id}/capabilities", response_model=WorkerResponse)
async def update_worker_capabilities(
    worker_id: str, request: WorkerUpdateCapabilitiesRequest, db: AsyncSession = Depends(get_db)
):
    """
    Update worker capabilities

    Args:
        worker_id: Worker ID
        request: New capabilities
        db: Database session

    Returns:
        Updated worker information

    Raises:
        404: Worker not found
    """
    result = await db.execute(select(Worker).where(Worker.worker_id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker {worker_id} not found"
        )

    # Update capabilities
    worker.capabilities = request.capabilities
    worker.last_heartbeat = datetime.utcnow()

    await db.commit()
    await db.refresh(worker)

    logger.info(f"Updated capabilities for worker: {worker_id}")
    return worker


@router.post("/{worker_id}/heartbeat", response_model=dict)
async def worker_heartbeat(worker_id: str, db: AsyncSession = Depends(get_db)):
    """
    Manual heartbeat update

    Args:
        worker_id: Worker ID
        db: Database session

    Returns:
        Heartbeat acknowledgment

    Raises:
        404: Worker not found
    """
    result = await db.execute(select(Worker).where(Worker.worker_id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker {worker_id} not found"
        )

    # Update heartbeat
    worker.last_heartbeat = datetime.utcnow()
    await db.commit()

    return {
        "worker_id": worker_id,
        "status": "ok",
        "last_heartbeat": worker.last_heartbeat.isoformat(),
    }


@router.delete("/{worker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deregister_worker(worker_id: str, db: AsyncSession = Depends(get_db)):
    """
    Deregister worker

    Args:
        worker_id: Worker ID
        db: Database session

    Raises:
        404: Worker not found
    """
    result = await db.execute(select(Worker).where(Worker.worker_id == worker_id))
    worker = result.scalar_one_or_none()

    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker {worker_id} not found"
        )

    await db.delete(worker)
    await db.commit()

    logger.info(f"Worker deregistered: {worker_id}")
