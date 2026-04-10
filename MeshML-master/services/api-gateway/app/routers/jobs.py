"""
Job management endpoints

Endpoints:
- POST /api/jobs - Submit new training job
- GET /api/jobs - List jobs
- GET /api/jobs/{job_id} - Get job details
- DELETE /api/jobs/{job_id} - Cancel job
- GET /api/jobs/{job_id}/progress - Get training progress
"""

import logging
import uuid
from typing import List, Optional

import redis.asyncio as redis
from app.clients.dataset_sharder_client import DatasetSharderClient
from app.clients.task_orchestrator_client import TaskOrchestratorClient
from app.models.dataset import Dataset
from app.models.group import GroupMember
from app.models.job import Job
from app.proto import dataset_sharder_pb2
from app.models.user import User
from app.proto import task_orchestrator_pb2
from app.routers.auth import get_current_user
from app.schemas.job import JobCreateRequest, JobProgressResponse, JobResponse
from app.utils.database import get_db, AsyncSessionLocal
from app.utils.redis_client import get_redis
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    request: JobCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    cache: redis.Redis = Depends(get_redis),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Submit new training job

    Args:
        request: Job configuration
        current_user: Authenticated user
        db: Database session

    Returns:
        Created job information

    Raises:
        403: Not member of group
    """
    logger.info(f"Creating job for group {request.group_id}")

    # Verify user is member of group
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == request.group_id, GroupMember.user_id == current_user.id
        )
    )
    member = member_result.scalar_one_or_none()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this group"
        )

    # Create job
    job_config = request.config.copy() if request.config else {}
    if request.dataset_id:
        dataset_result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
        dataset = dataset_result.scalar_one_or_none()
        if dataset and "dataset_format" not in job_config:
            job_config["dataset_format"] = dataset.format
    final_version = (job_config or {}).get("final_version")
    if final_version is not None:
        try:
            final_version_int = int(final_version)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="final_version must be an integer greater than 0.",
            )
        if final_version_int <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="final_version must be greater than 0.",
            )
        job_config["final_version"] = final_version_int

    job = Job(
        id=str(uuid.uuid4()),
        group_id=request.group_id,
        model_id=request.model_id,
        dataset_id=request.dataset_id,
        config=job_config if job_config else None,
        created_by=current_user.id,
        status="pending",
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Queue background processing to unblock the UI instantly
    if background_tasks:
        background_tasks.add_task(
            _initialize_job_background,
            job_id=str(job.id),
            group_id=str(job.group_id),
            model_id=str(job.model_id) if job.model_id else None,
            dataset_id=str(job.dataset_id) if job.dataset_id else None,
            config=job.config,
            current_user_id=str(current_user.id),
        )

    logger.info(f"Fast Job creation completed: queued sharding/orchestration for job {job.id}")
    return job

async def _initialize_job_background(
    job_id: str,
    group_id: str,
    model_id: Optional[str],
    dataset_id: Optional[str],
    config: dict,
    current_user_id: str,
):
    """Background task to shard dataset and deploy training pods."""
    config = config or {}
    
    async with AsyncSessionLocal() as db:
        try:
            import asyncio
            def _get_int(value, default):
                try: return int(value)
                except (TypeError, ValueError): return default

            def _get_float(value, default):
                try: return float(value)
                except (TypeError, ValueError): return default

            requirements_cfg = config.get("requirements", {})
            if not isinstance(requirements_cfg, dict):
                requirements_cfg = {}

            requirements = task_orchestrator_pb2.JobRequirements(
                min_gpu_count=_get_int(requirements_cfg.get("min_gpu_count"), 0),
                min_gpu_memory_gb=_get_float(requirements_cfg.get("min_gpu_memory_gb"), 0.0),
                min_cpu_count=_get_int(requirements_cfg.get("min_cpu_count"), 1),
                min_ram_gb=_get_float(requirements_cfg.get("min_ram_gb"), 1.0),
                requires_cuda=bool(requirements_cfg.get("requires_cuda", False)),
                requires_mps=bool(requirements_cfg.get("requires_mps", False)),
                max_execution_time_seconds=_get_int(requirements_cfg.get("max_execution_time_seconds"), 3600),
            )

            tags = {}
            config_tags = config.get("tags")
            if isinstance(config_tags, dict):
                tags.update({str(k): str(v) for k, v in config_tags.items()})

            for key in ["dataset_format", "num_shards", "shard_strategy"]:
                if key in config:
                    tags[key] = str(config.get(key))

            priority_value = config.get("priority", "MEDIUM")
            if isinstance(priority_value, str):
                priority_key = f"JOB_PRIORITY_{priority_value.upper()}"
                if priority_key in task_orchestrator_pb2.JobPriority.keys():
                    priority = task_orchestrator_pb2.JobPriority.Value(priority_key)
                else:
                    priority = task_orchestrator_pb2.JobPriority.Value("JOB_PRIORITY_MEDIUM")
            elif isinstance(priority_value, int):
                priority = priority_value if 0 <= priority_value <= 3 else 1
            else:
                priority = task_orchestrator_pb2.JobPriority.Value("JOB_PRIORITY_MEDIUM")

            submission = task_orchestrator_pb2.JobSubmission(
                job_id=job_id,
                group_id=group_id,
                model_id=model_id or "",
                dataset_id=dataset_id or "",
                user_id=current_user_id,
                batch_size=_get_int(config.get("batch_size"), 32),
                num_epochs=_get_int(config.get("num_epochs"), 10),
                learning_rate=_get_float(config.get("learning_rate"), 0.001),
                optimizer=str(config.get("optimizer", "adam")),
                requirements=requirements,
                tags=tags,
                description=str(config.get("description", "")),
                priority=priority,
            )

            if model_id:
                try:
                    from app.utils.redis_client import redis_client
                    if redis_client:
                        await redis_client.set(f"job:{job_id}:model_id", str(model_id))
                except Exception:
                    pass

            # Async Polling Loop: Wait for dataset extraction/upload to finish
            if not dataset_id:
                raise RuntimeError("No dataset provided for job")
                
            while True:
                dataset_result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
                dataset = dataset_result.scalar_one_or_none()
                
                if not dataset:
                    raise RuntimeError("Dataset is missing from database")
                    
                if dataset.status in ["processing", "pending", "uploading"]:
                    logger.info(f"Job {job_id} waiting for dataset {dataset_id} extraction/upload to complete... (status: {dataset.status})")
                    await asyncio.sleep(3)
                    continue
                    
                if dataset.status == "failed":
                    raise RuntimeError(f"Dataset {dataset_id} failed to upload/extract.")
                    
                if dataset.status in ["uploaded", "available"]:
                    logger.info(f"Dataset {dataset_id} is ready (status: {dataset.status}). Proceeding with task orchestration for Job {job_id}.")
                    break
                    
                logger.warning(f"Unknown dataset status: {dataset.status}. Attempting to shard anyway.")
                break

            if not dataset.gcs_path:
                raise RuntimeError("Dataset has no storage path for sharding")

            dataset_format = str(config.get("dataset_format") or dataset.format or "").lower()
            num_shards = _get_int(config.get("num_shards"), 10)
            if dataset.num_samples and dataset.num_samples > 0:
                num_shards = max(1, min(num_shards, int(dataset.num_samples)))
            shard_strategy = str(config.get("shard_strategy", "stratified"))
            batch_size = _get_int(config.get("batch_size"), 32)
            dataset_path = str(dataset.gcs_path)
            if dataset_path and not dataset_path.endswith("/"):
                dataset_path = f"{dataset_path}/"
                
            sharder = DatasetSharderClient()
            shard_response = await sharder.shard_dataset(
                dataset_sharder_pb2.ShardDatasetRequest(
                    dataset_id=dataset_id,
                    job_id=job_id,
                    model_id=model_id or "",
                    dataset_path=dataset_path,
                    format=dataset_format,
                    num_shards=num_shards,
                    strategy=shard_strategy,
                    batch_size=batch_size,
                    seed=42,
                )
            )
            if not shard_response.success:
                raise RuntimeError(shard_response.message or "Dataset sharding failed")
            
            logger.info("Sharding completed for job %s: dataset=%s batches=%s", job_id, dataset_id, shard_response.total_batches)

            orchestrator = TaskOrchestratorClient()
            response = await orchestrator.initiate_training(submission)
            if not response.success:
                raise RuntimeError(response.message or "submission failed")

            logger.info(f"Job submitted to Task Orchestrator: {job_id}")

        except Exception as e:
            logger.exception(f"Failed to submit job {job_id} to internal services: {e}")
            job_result = await db.execute(select(Job).where(Job.id == job_id))
            job = job_result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.error_message = f"Job submission failed: {e}"[:1000]
                await db.commit()


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    group_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List jobs (filtered by group and/or status)

    Args:
        group_id: Filter by group
        status: Filter by status
        current_user: Authenticated user
        db: Database session

    Returns:
        List of jobs
    """
    query = select(Job)

    if group_id:
        # Verify membership
        member_result = await db.execute(
            select(GroupMember).where(
                GroupMember.group_id == group_id, GroupMember.user_id == current_user.id
            )
        )
        if not member_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not a member of this group"
            )
        query = query.where(Job.group_id == group_id)

    if status:
        query = query.where(Job.status == status)

    result = await db.execute(query)
    jobs = result.scalars().all()

    logger.info(f"Retrieved {len(jobs)} jobs")
    return jobs


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Get job details

    Args:
        job_id: Job ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Job information

    Raises:
        404: Job not found
        403: Not authorized
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    # Verify membership
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == job.group_id, GroupMember.user_id == current_user.id
        )
    )
    if not member_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this job"
        )

    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(
    job_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Cancel running job

    Args:
        job_id: Job ID
        current_user: Authenticated user
        db: Database session

    Raises:
        404: Job not found
        403: Not authorized
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    # Only creator can cancel
    if str(job.created_by) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only job creator can cancel"
        )

    job.status = "cancelled"
    await db.commit()

    logger.info(f"Job cancelled: {job_id}")


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
async def get_job_progress(
    job_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Get training progress for job

    Args:
        job_id: Job ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Progress metrics

    Raises:
        404: Job not found
        403: Not authorized
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    # Verify membership
    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == job.group_id, GroupMember.user_id == current_user.id
        )
    )
    if not member_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this job"
        )

    # TODO: Get actual progress from Redis/metrics service
    progress = job.progress or {}

    return JobProgressResponse(
        job_id=job_id,
        status=job.status,
        current_epoch=progress.get("current_epoch", 0),
        total_epochs=progress.get("total_epochs", 0),
        current_batch=progress.get("current_batch", 0),
        total_batches=progress.get("total_batches", 0),
        loss=progress.get("loss", 0.0),
        accuracy=progress.get("accuracy", 0.0),
        worker_count=progress.get("worker_count", 0),
    )


@router.get("/{job_id}/status")
async def get_job_status(
    job_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Fallback status endpoint for clients without WebSocket support.
    Returns current job status and batch progress.
    """
    try:
        job_uuid = uuid.UUID(job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid job_id")

    result = await db.execute(select(Job).where(Job.id == job_uuid))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    member_result = await db.execute(
        select(GroupMember).where(
            GroupMember.group_id == job.group_id, GroupMember.user_id == current_user.id
        )
    )
    if not member_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this job"
        )

    total_batches_result = await db.execute(
        text("SELECT COUNT(*) FROM data_batches WHERE job_id = :job_id"), {"job_id": job_id}
    )
    total_batches = int(total_batches_result.scalar() or 0)

    completed_batches_result = await db.execute(
        text("SELECT COUNT(*) FROM data_batches WHERE job_id = :job_id AND status = 'COMPLETED'"),
        {"job_id": job_id},
    )
    completed_batches = int(completed_batches_result.scalar() or 0)

    return {
        "job_id": str(job.id),
        "status": job.status,
        "total_batches": total_batches,
        "completed_batches": completed_batches,
    }
