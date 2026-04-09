"""
Parameter Distribution API Router

RESTful endpoints for parameter distribution to workers.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.routers.gradients import TensorData, torch_to_tensor_data
from app.services.parameter_distribution import (
    CompressionType,
    DistributionConfig,
    DistributionMode,
    DistributionRequest,
    ParameterDistributionService,
    ParameterFormat,
)
from app.services.parameter_storage import ParameterStorageService
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/distribution", tags=["distribution"])

# Global service instances
parameter_storage = ParameterStorageService()
distribution_service = ParameterDistributionService(parameter_storage)


# ==================== Request/Response Models ====================


class PullParametersRequest(BaseModel):
    """Request to pull parameters"""

    model_id: str = Field(..., description="Model identifier")
    worker_id: str = Field(..., description="Worker identifier")
    current_version: Optional[int] = Field(default=None, description="Current version worker has")
    requested_version: Optional[int] = Field(
        default=None, description="Requested version (None = latest)"
    )
    delta_only: bool = Field(default=False, description="Request delta if beneficial")
    compression: CompressionType = Field(
        default=CompressionType.NONE, description="Compression type"
    )
    format_type: ParameterFormat = Field(
        default=ParameterFormat.PYTORCH, description="Parameter format"
    )
    parameter_names: Optional[List[str]] = Field(
        default=None, description="Specific parameters (None = all)"
    )


class BroadcastParametersRequest(BaseModel):
    """Request to broadcast parameters"""

    model_id: str = Field(..., description="Model identifier")
    worker_ids: List[str] = Field(..., description="Worker identifiers")
    version_id: Optional[int] = Field(
        default=None, description="Version to broadcast (None = latest)"
    )
    compression: CompressionType = Field(
        default=CompressionType.NONE, description="Compression type"
    )
    format_type: ParameterFormat = Field(
        default=ParameterFormat.PYTORCH, description="Parameter format"
    )


class SubscribeRequest(BaseModel):
    """Request to subscribe to parameter updates"""

    model_id: str = Field(..., description="Model identifier")
    worker_id: str = Field(..., description="Worker identifier")


class ParameterPackageResponse(BaseModel):
    """Response containing parameter package"""

    model_id: str
    version_id: int
    parameters: Dict[str, TensorData]
    parameter_names: List[str]
    is_delta: bool
    base_version: Optional[int]
    checksum: str
    size_bytes: int
    compressed: bool
    compression_type: Optional[str]
    format_type: str
    metadata: Dict[str, Any]
    created_at: datetime


class DistributionRecordResponse(BaseModel):
    """Response for distribution record"""

    record_id: str
    model_id: str
    version_id: int
    worker_ids: List[str]
    is_delta: bool
    size_bytes: int
    compression_type: str
    distributed_at: datetime
    metadata: Dict[str, Any]


class DistributionStatisticsResponse(BaseModel):
    """Response for distribution statistics"""

    total_distributions: int
    delta_distributions: int
    full_distributions: int
    total_bytes_transferred: int
    total_mb_transferred: float
    unique_workers: int
    total_subscriptions: int
    models_with_subscriptions: int


# ==================== Helper Functions ====================


def package_to_response(package) -> ParameterPackageResponse:
    """Convert ParameterPackage to response model"""
    # Convert parameters to TensorData
    parameters_data = {}
    for name, tensor in package.parameters.items():
        if hasattr(tensor, "shape"):  # PyTorch tensor or NumPy array
            parameters_data[name] = torch_to_tensor_data(tensor)
        else:
            # For other types, skip or handle differently
            continue

    return ParameterPackageResponse(
        model_id=package.model_id,
        version_id=package.version_id,
        parameters=parameters_data,
        parameter_names=package.parameter_names,
        is_delta=package.is_delta,
        base_version=package.base_version,
        checksum=package.checksum,
        size_bytes=package.size_bytes,
        compressed=package.compressed,
        compression_type=package.compression_type.value if package.compression_type else None,
        format_type=package.format_type.value,
        metadata=package.metadata,
        created_at=package.created_at,
    )


# ==================== Endpoints ====================


@router.post("/pull", response_model=ParameterPackageResponse)
async def pull_parameters(request: PullParametersRequest) -> ParameterPackageResponse:
    """
    Pull parameters from server (pull mode).

    Workers request the latest (or specific version) parameters.
    Supports delta compression and various formats.
    """
    try:
        dist_request = DistributionRequest(
            model_id=request.model_id,
            worker_id=request.worker_id,
            current_version=request.current_version,
            requested_version=request.requested_version,
            delta_only=request.delta_only,
            compression=request.compression,
            format_type=request.format_type,
            parameter_names=request.parameter_names,
        )

        package = distribution_service.distribute_to_worker(
            worker_id=request.worker_id, request=dist_request
        )

        # Decompress if needed for response
        if package.compressed:
            package = distribution_service.decompress_package(package)

        return package_to_response(package)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast", response_model=Dict[str, Any])
async def broadcast_parameters(request: BroadcastParametersRequest) -> Dict[str, Any]:
    """
    Broadcast parameters to multiple workers (push mode).

    Server pushes parameters to specified workers.
    """
    try:
        # Create temporary config for this broadcast
        config = DistributionConfig(
            mode=DistributionMode.PUSH,
            default_compression=request.compression,
            default_format=request.format_type,
        )

        packages = distribution_service.broadcast_to_workers(
            model_id=request.model_id,
            worker_ids=request.worker_ids,
            version_id=request.version_id,
            config=config,
        )

        return {
            "status": "success",
            "message": f"Parameters broadcast to {len(packages)} workers",
            "model_id": request.model_id,
            "version_id": request.version_id or "latest",
            "num_workers": len(packages),
            "worker_ids": request.worker_ids,
            "total_size_mb": sum(p.size_bytes for p in packages.values()) / 1024 / 1024,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subscribe", response_model=Dict[str, Any])
async def subscribe_to_updates(request: SubscribeRequest) -> Dict[str, Any]:
    """
    Subscribe a worker to receive parameter updates (push mode).

    Worker will be notified when parameters are updated.
    """
    try:
        newly_subscribed = distribution_service.subscribe_worker(
            model_id=request.model_id, worker_id=request.worker_id
        )

        return {
            "status": "success",
            "message": (
                "Subscribed to parameter updates" if newly_subscribed else "Already subscribed"
            ),
            "model_id": request.model_id,
            "worker_id": request.worker_id,
            "newly_subscribed": newly_subscribed,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/subscribe", response_model=Dict[str, Any])
async def unsubscribe_from_updates(request: SubscribeRequest) -> Dict[str, Any]:
    """
    Unsubscribe a worker from parameter updates.
    """
    try:
        was_subscribed = distribution_service.unsubscribe_worker(
            model_id=request.model_id, worker_id=request.worker_id
        )

        if not was_subscribed:
            raise HTTPException(
                status_code=404,
                detail=f"Worker {request.worker_id} not subscribed to {request.model_id}",
            )

        return {
            "status": "success",
            "message": "Unsubscribed from parameter updates",
            "model_id": request.model_id,
            "worker_id": request.worker_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subscriptions/{model_id}", response_model=Dict[str, Any])
async def get_subscriptions(model_id: str) -> Dict[str, Any]:
    """
    Get list of workers subscribed to a model.
    """
    try:
        workers = distribution_service.get_subscribed_workers(model_id)

        return {"model_id": model_id, "num_subscribers": len(workers), "worker_ids": workers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[DistributionRecordResponse])
async def get_distribution_history(
    model_id: Optional[str] = None, worker_id: Optional[str] = None, limit: Optional[int] = None
) -> List[DistributionRecordResponse]:
    """
    Get parameter distribution history.

    Query parameters:
    - model_id: Filter by model (optional)
    - worker_id: Filter by worker (optional)
    - limit: Maximum records (optional)
    """
    try:
        history = distribution_service.get_distribution_history(
            model_id=model_id, worker_id=worker_id, limit=limit
        )

        return [
            DistributionRecordResponse(
                record_id=r.record_id,
                model_id=r.model_id,
                version_id=r.version_id,
                worker_ids=r.worker_ids,
                is_delta=r.is_delta,
                size_bytes=r.size_bytes,
                compression_type=r.compression_type.value,
                distributed_at=r.distributed_at,
                metadata=r.metadata,
            )
            for r in history
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats", response_model=Dict[str, List[str]])
async def get_available_formats() -> Dict[str, List[str]]:
    """
    Get available distribution modes, formats, and compression types.
    """
    return {
        "distribution_modes": [mode.value for mode in DistributionMode],
        "parameter_formats": [fmt.value for fmt in ParameterFormat],
        "compression_types": [comp.value for comp in CompressionType],
    }


@router.get("/stats/summary", response_model=DistributionStatisticsResponse)
async def get_statistics() -> DistributionStatisticsResponse:
    """
    Get parameter distribution statistics.
    """
    try:
        stats = distribution_service.get_statistics()
        return DistributionStatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for distribution service.
    """
    stats = distribution_service.get_statistics()

    return {
        "status": "healthy",
        "service": "parameter_distribution",
        "total_distributions": stats["total_distributions"],
        "total_subscriptions": stats["total_subscriptions"],
    }
