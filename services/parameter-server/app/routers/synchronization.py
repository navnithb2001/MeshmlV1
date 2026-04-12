"""
Synchronization API Router

RESTful endpoints for managing synchronization strategies and worker coordination.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.routers.gradients import TensorData, tensor_data_to_torch
from app.services.gradient_aggregation import (
    AggregationConfig,
    AggregationStrategy,
    ClippingStrategy,
    GradientAggregationService,
    GradientUpdate,
)
from app.services.synchronization import SyncConfig, SynchronizationService, SyncMode, WorkerState
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sync", tags=["synchronization"])

# Global service instances
gradient_service = GradientAggregationService()
sync_service = SynchronizationService(gradient_service)


# ==================== Request/Response Models ====================


class SyncConfigRequest(BaseModel):
    """Request to configure synchronization"""

    mode: SyncMode = Field(default=SyncMode.ASYNCHRONOUS, description="Synchronization mode")

    # Synchronous settings
    min_workers: int = Field(default=1, gt=0, description="Minimum workers required")
    max_workers: Optional[int] = Field(default=None, description="Maximum workers expected")
    sync_timeout_seconds: float = Field(default=60.0, gt=0, description="Sync timeout")

    # Semi-synchronous settings
    worker_quorum: float = Field(default=0.8, ge=0.0, le=1.0, description="Worker quorum fraction")
    max_staleness: int = Field(default=10, ge=0, description="Max staleness")
    quorum_timeout_seconds: float = Field(default=30.0, gt=0, description="Quorum timeout")

    # Asynchronous settings
    async_batch_size: int = Field(default=1, gt=0, description="Async batch size")
    async_timeout_seconds: Optional[float] = Field(default=None, description="Async timeout")

    # Worker management
    worker_timeout_seconds: float = Field(default=300.0, gt=0, description="Worker timeout")
    auto_exclude_timeouts: bool = Field(default=True, description="Auto-exclude timed out workers")

    # Aggregation settings
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.FEDAVG, description="Aggregation strategy"
    )
    clipping_strategy: ClippingStrategy = Field(
        default=ClippingStrategy.NONE, description="Clipping strategy"
    )
    clip_value: float = Field(default=1.0, gt=0, description="Clip value")
    clip_norm: float = Field(default=1.0, gt=0, description="Clip norm")
    staleness_weight_decay: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Staleness decay"
    )
    normalize_gradients: bool = Field(default=False, description="Normalize gradients")


class WorkerRegisterRequest(BaseModel):
    """Request to register a worker"""

    worker_id: str = Field(..., description="Worker identifier")
    model_id: str = Field(..., description="Model identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Worker metadata")


class GradientSubmitSyncRequest(BaseModel):
    """Request to submit gradient with synchronization"""

    worker_id: str = Field(..., description="Worker identifier")
    model_id: str = Field(..., description="Model identifier")
    version_id: int = Field(..., description="Parameter version")
    gradients: Dict[str, TensorData] = Field(..., description="Gradient tensors")
    num_samples: int = Field(..., gt=0, description="Number of samples")
    loss: Optional[float] = Field(default=None, description="Training loss")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    sync_config: Optional[SyncConfigRequest] = Field(default=None, description="Custom sync config")


class WorkerInfoResponse(BaseModel):
    """Response for worker information"""

    worker_id: str
    model_id: str
    state: str
    last_seen: datetime
    last_gradient_version: Optional[int]
    current_round: int
    total_gradients: int
    total_samples: int
    metadata: Dict[str, Any]


class SyncRoundResponse(BaseModel):
    """Response for synchronization round"""

    round_id: int
    model_id: str
    target_version: int
    expected_workers: List[str]
    received_workers: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    timed_out: bool
    num_workers: Optional[int] = None
    total_samples: Optional[int] = None


class SyncStatisticsResponse(BaseModel):
    """Response for synchronization statistics"""

    total_workers: int
    active_workers: int
    timed_out_workers: int
    total_rounds: int
    completed_rounds: int
    timed_out_rounds: int
    avg_round_duration_seconds: float
    active_rounds: int


# ==================== Helper Functions ====================


def sync_config_from_request(req: SyncConfigRequest) -> SyncConfig:
    """Convert request to SyncConfig"""
    aggregation_config = AggregationConfig(
        strategy=req.aggregation_strategy,
        clipping_strategy=req.clipping_strategy,
        clip_value=req.clip_value,
        clip_norm=req.clip_norm,
        staleness_weight_decay=req.staleness_weight_decay,
        max_staleness=req.max_staleness,
        normalize_gradients=req.normalize_gradients,
    )

    return SyncConfig(
        mode=req.mode,
        min_workers=req.min_workers,
        max_workers=req.max_workers,
        sync_timeout_seconds=req.sync_timeout_seconds,
        worker_quorum=req.worker_quorum,
        max_staleness=req.max_staleness,
        quorum_timeout_seconds=req.quorum_timeout_seconds,
        async_batch_size=req.async_batch_size,
        async_timeout_seconds=req.async_timeout_seconds,
        worker_timeout_seconds=req.worker_timeout_seconds,
        auto_exclude_timeouts=req.auto_exclude_timeouts,
        aggregation_config=aggregation_config,
    )


# ==================== Endpoints ====================


@router.post("/workers/register", response_model=WorkerInfoResponse)
async def register_worker(request: WorkerRegisterRequest) -> WorkerInfoResponse:
    """
    Register a worker with the synchronization service.

    Workers must register before submitting gradients in synchronous mode.
    """
    try:
        worker = sync_service.register_worker(
            worker_id=request.worker_id, model_id=request.model_id, metadata=request.metadata
        )

        return WorkerInfoResponse(
            worker_id=worker.worker_id,
            model_id=worker.model_id,
            state=worker.state.value,
            last_seen=worker.last_seen,
            last_gradient_version=worker.last_gradient_version,
            current_round=worker.current_round,
            total_gradients=worker.total_gradients,
            total_samples=worker.total_samples,
            metadata=worker.metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workers/{worker_id}", response_model=Dict[str, Any])
async def unregister_worker(worker_id: str) -> Dict[str, Any]:
    """
    Unregister a worker.

    The worker will be excluded from future synchronization rounds.
    """
    try:
        success = sync_service.unregister_worker(worker_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

        return {
            "status": "success",
            "message": f"Worker {worker_id} unregistered",
            "worker_id": worker_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers/{worker_id}", response_model=WorkerInfoResponse)
async def get_worker_info(worker_id: str) -> WorkerInfoResponse:
    """Get information about a specific worker"""
    try:
        worker = sync_service.get_worker_info(worker_id)

        if worker is None:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

        return WorkerInfoResponse(
            worker_id=worker.worker_id,
            model_id=worker.model_id,
            state=worker.state.value,
            last_seen=worker.last_seen,
            last_gradient_version=worker.last_gradient_version,
            current_round=worker.current_round,
            total_gradients=worker.total_gradients,
            total_samples=worker.total_samples,
            metadata=worker.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workers", response_model=List[WorkerInfoResponse])
async def list_workers(
    model_id: Optional[str] = None, state: Optional[WorkerState] = None
) -> List[WorkerInfoResponse]:
    """
    List all workers with optional filtering.

    Query parameters:
    - model_id: Filter by model (optional)
    - state: Filter by worker state (optional)
    """
    try:
        workers = sync_service.list_workers(model_id=model_id, state=state)

        return [
            WorkerInfoResponse(
                worker_id=w.worker_id,
                model_id=w.model_id,
                state=w.state.value,
                last_seen=w.last_seen,
                last_gradient_version=w.last_gradient_version,
                current_round=w.current_round,
                total_gradients=w.total_gradients,
                total_samples=w.total_samples,
                metadata=w.metadata,
            )
            for w in workers
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit", response_model=Dict[str, Any])
async def submit_gradient_sync(request: GradientSubmitSyncRequest) -> Dict[str, Any]:
    """
    Submit gradient with synchronization handling.

    This endpoint handles gradient submission with the configured
    synchronization strategy (sync/async/semi-sync).

    Returns aggregation result if aggregation occurred.
    """
    try:
        # Convert gradients
        gradients = {
            name: tensor_data_to_torch(tensor_data)
            for name, tensor_data in request.gradients.items()
        }

        # Create gradient update
        gradient_update = GradientUpdate(
            worker_id=request.worker_id,
            model_id=request.model_id,
            version_id=request.version_id,
            gradients=gradients,
            num_samples=request.num_samples,
            loss=request.loss,
            metrics=request.metrics,
            metadata=request.metadata,
        )

        # Get sync config
        sync_config = None
        if request.sync_config:
            sync_config = sync_config_from_request(request.sync_config)

        # Submit with synchronization
        result = await sync_service.submit_gradient(gradient_update, sync_config)

        response = {
            "status": "success",
            "message": "Gradient submitted",
            "worker_id": request.worker_id,
            "model_id": request.model_id,
            "version_id": request.version_id,
            "aggregated": result is not None,
        }

        # Add aggregation info if available
        if result:
            response["aggregation"] = {
                "num_workers": result.num_workers,
                "total_samples": result.total_samples,
                "worker_ids": result.worker_ids,
                "strategy": result.strategy.value,
                "target_version": result.target_version_id,
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/current/{model_id}", response_model=Optional[SyncRoundResponse])
async def get_current_round(model_id: str) -> Optional[SyncRoundResponse]:
    """
    Get current synchronization round for a model.

    Returns None if no active round.
    """
    try:
        round_info = sync_service.get_current_round(model_id)

        if round_info is None:
            return None

        response = SyncRoundResponse(
            round_id=round_info.round_id,
            model_id=round_info.model_id,
            target_version=round_info.target_version,
            expected_workers=list(round_info.expected_workers),
            received_workers=list(round_info.received_workers),
            started_at=round_info.started_at,
            completed_at=round_info.completed_at,
            timed_out=round_info.timed_out,
        )

        if round_info.aggregation_result:
            response.num_workers = round_info.aggregation_result.num_workers
            response.total_samples = round_info.aggregation_result.total_samples

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rounds/history", response_model=List[SyncRoundResponse])
async def get_round_history(
    model_id: Optional[str] = None, limit: Optional[int] = None
) -> List[SyncRoundResponse]:
    """
    Get synchronization round history.

    Query parameters:
    - model_id: Filter by model (optional)
    - limit: Maximum records (optional)
    """
    try:
        rounds = sync_service.get_round_history(model_id=model_id, limit=limit)

        return [
            SyncRoundResponse(
                round_id=r.round_id,
                model_id=r.model_id,
                target_version=r.target_version,
                expected_workers=list(r.expected_workers),
                received_workers=list(r.received_workers),
                started_at=r.started_at,
                completed_at=r.completed_at,
                timed_out=r.timed_out,
                num_workers=r.aggregation_result.num_workers if r.aggregation_result else None,
                total_samples=r.aggregation_result.total_samples if r.aggregation_result else None,
            )
            for r in rounds
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modes", response_model=Dict[str, List[str]])
async def get_sync_modes() -> Dict[str, List[str]]:
    """
    Get available synchronization modes and worker states.
    """
    return {
        "sync_modes": [mode.value for mode in SyncMode],
        "worker_states": [state.value for state in WorkerState],
    }


@router.get("/stats/summary", response_model=SyncStatisticsResponse)
async def get_statistics() -> SyncStatisticsResponse:
    """
    Get synchronization service statistics.
    """
    try:
        stats = sync_service.get_statistics()
        return SyncStatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for synchronization service.
    """
    stats = sync_service.get_statistics()

    return {
        "status": "healthy",
        "service": "synchronization",
        "active_workers": stats["active_workers"],
        "active_rounds": stats["active_rounds"],
    }
