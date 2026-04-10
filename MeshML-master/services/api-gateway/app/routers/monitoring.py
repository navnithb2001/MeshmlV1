"""
Monitoring and metrics endpoints

Endpoints:
- GET /api/monitoring/health - System health check
- GET /api/monitoring/metrics/realtime - Real-time system stats
- GET /api/monitoring/workers - Worker status
- GET /api/monitoring/groups/{group_id}/stats - Group statistics
"""

import logging
from typing import Any, Dict

from app.models.group import Group, GroupMember
from app.models.job import Job
from app.models.worker import Worker
from app.utils.database import get_db
from app.utils.redis_client import get_redis
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    """
    System health check

    Returns:
        Health status of all components
    """
    health = {"status": "healthy", "components": {}}

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        health["components"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health["components"]["database"] = "unhealthy"
        health["status"] = "degraded"

    # Check Redis
    try:
        await redis.ping()
        health["components"]["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health["components"]["redis"] = "unhealthy"
        health["status"] = "degraded"

    return health


@router.get("/metrics/realtime")
async def get_realtime_metrics(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Get real-time system metrics

    Returns:
        Current system statistics
    """
    # Count workers by status
    worker_result = await db.execute(
        select(Worker.status, func.count(Worker.id).label("count")).group_by(Worker.status)
    )
    worker_stats = {row[0]: row[1] for row in worker_result.all()}

    # Count jobs by status
    job_result = await db.execute(
        select(Job.status, func.count(Job.id).label("count")).group_by(Job.status)
    )
    job_stats = {row[0]: row[1] for row in job_result.all()}

    # Count groups
    group_result = await db.execute(select(func.count(Group.id)))
    total_groups = group_result.scalar()

    return {
        "workers": {
            "total": sum(worker_stats.values()),
            "idle": worker_stats.get("idle", 0),
            "training": worker_stats.get("training", 0),
            "offline": worker_stats.get("offline", 0),
        },
        "jobs": {
            "total": sum(job_stats.values()),
            "pending": job_stats.get("pending", 0),
            "running": job_stats.get("running", 0),
            "completed": job_stats.get("completed", 0),
            "failed": job_stats.get("failed", 0),
        },
        "groups": {"total": total_groups},
    }


@router.get("/workers")
async def get_worker_status(db: AsyncSession = Depends(get_db)):
    """
    Get status of all workers

    Returns:
        List of workers with status
    """
    result = await db.execute(select(Worker))
    workers = result.scalars().all()

    return {
        "workers": [
            {
                "worker_id": w.worker_id,
                "status": w.status,
                "capabilities": w.capabilities,
                "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
            }
            for w in workers
        ],
        "total": len(workers),
    }


@router.get("/groups/{group_id}/stats")
async def get_group_stats(group_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get statistics for specific group

    Args:
        group_id: Group ID
        db: Database session

    Returns:
        Group statistics
    """
    # Get member count
    member_result = await db.execute(
        select(func.count(GroupMember.id)).where(GroupMember.group_id == group_id)
    )
    member_count = member_result.scalar()

    # Get job counts
    job_result = await db.execute(
        select(Job.status, func.count(Job.id).label("count"))
        .where(Job.group_id == group_id)
        .group_by(Job.status)
    )
    job_stats = {row[0]: row[1] for row in job_result.all()}

    return {
        "group_id": group_id,
        "members": member_count,
        "jobs": {
            "total": sum(job_stats.values()),
            "pending": job_stats.get("pending", 0),
            "running": job_stats.get("running", 0),
            "completed": job_stats.get("completed", 0),
            "failed": job_stats.get("failed", 0),
        },
    }
