"""API router for data distribution endpoints.

Provides HTTP endpoints for workers to discover and download assigned batches.
"""

import logging
import pickle
from typing import List, Optional

from app.config import settings
from app.core.storage import get_dataset_storage
from app.services.batch_storage import BatchManager, create_storage_backend
from app.services.data_distribution import AssignmentStatus, DataDistributor, DistributionStrategy
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/distribution", tags=["data-distribution"])

# Global instances (would be dependency-injected in production)
_batch_manager = None
_distributor = None


def get_batch_manager() -> BatchManager:
    """Dependency to get BatchManager instance."""
    global _batch_manager
    if _batch_manager is None:
        storage = create_storage_backend(storage_type="local")
        _batch_manager = BatchManager(storage_backend=storage)
    return _batch_manager


def get_distributor() -> DataDistributor:
    """Dependency to get DataDistributor instance."""
    global _distributor
    if _distributor is None:
        batch_manager = get_batch_manager()
        _distributor = DataDistributor(
            batch_manager=batch_manager, strategy=DistributionStrategy.SHARD_PER_WORKER
        )
    return _distributor


# Request/Response Models


class AssignBatchesRequest(BaseModel):
    """Request to assign batches to workers."""

    worker_ids: List[str] = Field(..., min_items=1, description="List of worker IDs")
    shard_id: Optional[int] = Field(None, description="Optional shard ID filter")
    strategy: Optional[str] = Field(
        "shard_per_worker",
        description="Distribution strategy: shard_per_worker, round_robin, load_balanced",
    )


class WorkerAssignmentResponse(BaseModel):
    """Response containing worker's batch assignments."""

    worker_id: str
    shard_id: int
    assigned_batches: List[str]
    total_samples: int
    progress: float
    is_complete: bool


class BatchAssignmentResponse(BaseModel):
    """Response for batch assignment details."""

    assignment_id: str
    batch_id: str
    worker_id: str
    shard_id: int
    batch_index: int
    status: str
    assigned_at: str
    downloaded_at: Optional[str] = None
    failed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    retry_count: int


class DownloadStatusUpdate(BaseModel):
    """Request to update download status."""

    status: str = Field(..., description="Status: downloading, completed, failed")
    failure_reason: Optional[str] = Field(None, description="Reason if status is failed")


class ReassignRequest(BaseModel):
    """Request to reassign a failed batch."""

    batch_id: str
    new_worker_id: str


class DistributionStatsResponse(BaseModel):
    """Response containing distribution statistics."""

    total_assignments: int
    total_workers: int
    status_counts: dict
    worker_stats: dict
    strategy: str


class DownloadUrlResponse(BaseModel):
    """Response containing signed download URL."""

    download_url: str
    storage_path: str
    expires_in_seconds: int = 3600


# Endpoints


@router.post("/assign", response_model=dict)
async def assign_batches_to_workers(
    request: AssignBatchesRequest, distributor: DataDistributor = Depends(get_distributor)
):
    """
    Assign batches to workers based on distribution strategy.

    This endpoint creates assignments for all specified workers,
    distributing available batches according to the chosen strategy.
    """
    try:
        # Parse strategy
        if request.strategy:
            try:
                strategy = DistributionStrategy(request.strategy)
                distributor.strategy = strategy
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")

        # Assign batches
        assignments = distributor.assign_batches_to_workers(
            worker_ids=request.worker_ids, shard_id=request.shard_id
        )

        # Format response
        response = {
            worker_id: {
                "worker_id": assignment.worker_id,
                "shard_id": assignment.shard_id,
                "assigned_batches": assignment.assigned_batches,
                "total_samples": assignment.total_samples,
                "progress": assignment.get_progress(),
                "is_complete": assignment.is_complete(),
            }
            for worker_id, assignment in assignments.items()
        }

        return {
            "success": True,
            "message": f"Assigned batches to {len(assignments)} workers",
            "assignments": response,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to assign batches: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers/{worker_id}/assignment", response_model=WorkerAssignmentResponse)
async def get_worker_assignment(
    worker_id: str, distributor: DataDistributor = Depends(get_distributor)
):
    """
    Get batch assignment for a specific worker.

    Returns the list of batches assigned to this worker and progress.
    """
    assignment = distributor.get_worker_assignment(worker_id)

    if not assignment:
        raise HTTPException(status_code=404, detail=f"No assignment found for worker {worker_id}")

    return WorkerAssignmentResponse(
        worker_id=assignment.worker_id,
        shard_id=assignment.shard_id,
        assigned_batches=assignment.assigned_batches,
        total_samples=assignment.total_samples,
        progress=assignment.get_progress(),
        is_complete=assignment.is_complete(),
    )


@router.get("/workers/{worker_id}/batches", response_model=List[str])
async def list_worker_batches(
    worker_id: str, distributor: DataDistributor = Depends(get_distributor)
):
    """
    List all batch IDs assigned to a worker.

    Useful for workers to know what batches they need to download.
    """
    assignment = distributor.get_worker_assignment(worker_id)

    if not assignment:
        raise HTTPException(status_code=404, detail=f"No assignment found for worker {worker_id}")

    return assignment.assigned_batches


@router.get("/batches/{batch_id}/download-url", response_model=DownloadUrlResponse)
async def get_batch_download_url(
    batch_id: str, batch_manager: BatchManager = Depends(get_batch_manager)
):
    """
    Get a signed download URL for a batch stored in GCS.
    """
    try:
        _, metadata = batch_manager.load_batch(batch_id)
        storage_path = metadata.storage_path
        if not storage_path.startswith("gs://"):
            raise HTTPException(status_code=400, detail="Batch not stored in GCS")

        blob_path = storage_path.replace(f"gs://{settings.GCS_BUCKET_DATASETS}/", "")
        storage_client = get_dataset_storage()
        download_url = storage_client.generate_presigned_download_url(blob_path)
        return DownloadUrlResponse(
            download_url=download_url, storage_path=storage_path, expires_in_seconds=3600
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Batch not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batches/{batch_id}/assignment", response_model=BatchAssignmentResponse)
async def get_batch_assignment(
    batch_id: str, distributor: DataDistributor = Depends(get_distributor)
):
    """Get assignment details for a specific batch."""
    assignment = distributor.get_batch_assignment(batch_id)

    if not assignment:
        raise HTTPException(status_code=404, detail=f"No assignment found for batch {batch_id}")

    return BatchAssignmentResponse(
        assignment_id=assignment.assignment_id,
        batch_id=assignment.batch_id,
        worker_id=assignment.worker_id,
        shard_id=assignment.shard_id,
        batch_index=assignment.batch_index,
        status=assignment.status.value,
        assigned_at=assignment.assigned_at,
        downloaded_at=assignment.downloaded_at,
        failed_at=assignment.failed_at,
        failure_reason=assignment.failure_reason,
        retry_count=assignment.retry_count,
    )


@router.get("/workers/{worker_id}/batches/{batch_id}/download")
async def download_batch(
    worker_id: str,
    batch_id: str,
    batch_manager: BatchManager = Depends(get_batch_manager),
    distributor: DataDistributor = Depends(get_distributor),
):
    """
    Download a specific batch.

    Returns the serialized batch data as streaming response.
    Workers should call this endpoint to download their assigned batches.
    """
    # Verify assignment
    assignment = distributor.get_batch_assignment(batch_id)

    if not assignment:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not assigned")

    if assignment.worker_id != worker_id:
        raise HTTPException(
            status_code=403, detail=f"Batch {batch_id} not assigned to worker {worker_id}"
        )

    # Mark download as started
    distributor.mark_download_started(worker_id, batch_id)

    try:
        # Load batch
        samples, metadata = batch_manager.load_batch(batch_id)

        # Serialize for download
        batch_data = {"samples": samples, "metadata": metadata.to_dict()}

        serialized = pickle.dumps(batch_data, protocol=pickle.HIGHEST_PROTOCOL)

        # Mark as completed (will be marked after successful download)
        # Note: In production, you'd want confirmation from worker

        def iterfile():
            """Stream file in chunks."""
            chunk_size = 65536  # 64KB chunks
            for i in range(0, len(serialized), chunk_size):
                yield serialized[i : i + chunk_size]

        return StreamingResponse(
            iterfile(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={batch_id}.pkl",
                "Content-Length": str(len(serialized)),
                "X-Batch-ID": batch_id,
                "X-Shard-ID": str(metadata.shard_id),
                "X-Num-Samples": str(metadata.num_samples),
            },
        )

    except FileNotFoundError:
        # Mark as failed
        distributor.mark_download_failed(worker_id, batch_id, "Batch file not found in storage")
        raise HTTPException(status_code=404, detail="Batch file not found")

    except Exception as e:
        # Mark as failed
        distributor.mark_download_failed(worker_id, batch_id, f"Download error: {str(e)}")
        logger.error(f"Failed to download batch {batch_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workers/{worker_id}/batches/{batch_id}/status")
async def update_download_status(
    worker_id: str,
    batch_id: str,
    status_update: DownloadStatusUpdate,
    distributor: DataDistributor = Depends(get_distributor),
):
    """
    Update download status for a batch.

    Workers should call this after download completes or fails.
    """
    status = status_update.status.lower()

    success = False

    if status == "downloading":
        success = distributor.mark_download_started(worker_id, batch_id)
    elif status == "completed":
        success = distributor.mark_download_completed(worker_id, batch_id)
    elif status == "failed":
        reason = status_update.failure_reason or "Unknown error"
        success = distributor.mark_download_failed(worker_id, batch_id, reason)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if not success:
        raise HTTPException(status_code=400, detail="Failed to update status")

    return {
        "success": True,
        "message": f"Status updated to {status}",
        "worker_id": worker_id,
        "batch_id": batch_id,
    }


@router.post("/batches/{batch_id}/reassign")
async def reassign_batch(
    batch_id: str, request: ReassignRequest, distributor: DataDistributor = Depends(get_distributor)
):
    """
    Reassign a failed batch to a different worker.

    Useful for handling worker failures or load rebalancing.
    """
    new_assignment = distributor.reassign_failed_batch(
        batch_id=batch_id, new_worker_id=request.new_worker_id
    )

    if not new_assignment:
        raise HTTPException(
            status_code=400, detail="Failed to reassign batch (may have exceeded max retries)"
        )

    return {
        "success": True,
        "message": f"Batch {batch_id} reassigned to {request.new_worker_id}",
        "assignment": BatchAssignmentResponse(
            assignment_id=new_assignment.assignment_id,
            batch_id=new_assignment.batch_id,
            worker_id=new_assignment.worker_id,
            shard_id=new_assignment.shard_id,
            batch_index=new_assignment.batch_index,
            status=new_assignment.status.value,
            assigned_at=new_assignment.assigned_at,
            downloaded_at=new_assignment.downloaded_at,
            failed_at=new_assignment.failed_at,
            failure_reason=new_assignment.failure_reason,
            retry_count=new_assignment.retry_count,
        ),
    }


@router.post("/reassign-failed")
async def auto_reassign_failed_batches(
    worker_ids: List[str], distributor: DataDistributor = Depends(get_distributor)
):
    """
    Automatically reassign all failed batches to available workers.

    Useful for bulk recovery after worker failures.
    """
    new_assignments = distributor.auto_reassign_failed_batches(worker_ids)

    return {
        "success": True,
        "message": f"Reassigned {len(new_assignments)} failed batches",
        "reassigned_count": len(new_assignments),
        "assignments": [
            {"batch_id": a.batch_id, "new_worker_id": a.worker_id, "retry_count": a.retry_count}
            for a in new_assignments
        ],
    }


@router.get("/stats", response_model=DistributionStatsResponse)
async def get_distribution_stats(distributor: DataDistributor = Depends(get_distributor)):
    """
    Get overall distribution statistics.

    Returns counts of assignments by status, per-worker progress, etc.
    """
    stats = distributor.get_distribution_stats()

    return DistributionStatsResponse(
        total_assignments=stats["total_assignments"],
        total_workers=stats["total_workers"],
        status_counts=stats["status_counts"],
        worker_stats=stats["worker_stats"],
        strategy=stats["strategy"],
    )


@router.get("/health")
async def distribution_health_check(
    batch_manager: BatchManager = Depends(get_batch_manager),
    distributor: DataDistributor = Depends(get_distributor),
):
    """Health check for distribution service."""
    try:
        # Check batch storage
        batch_stats = batch_manager.get_batch_stats()

        # Check distributor
        dist_stats = distributor.get_distribution_stats()

        return {
            "status": "healthy",
            "batch_storage": {
                "total_batches": batch_stats.get("total_batches", 0),
                "total_size_mb": batch_stats.get("total_size_mb", 0),
            },
            "distribution": {
                "total_workers": dist_stats["total_workers"],
                "total_assignments": dist_stats["total_assignments"],
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {"status": "unhealthy", "error": str(e)}
