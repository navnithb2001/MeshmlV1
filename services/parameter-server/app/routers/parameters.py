"""
Parameter Storage API Router

HTTP endpoints for parameter storage, versioning, and checkpoint management.
"""

import pickle
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.services.model_registry_client import ModelRegistryClient
from app.services.parameter_storage import CheckpointType, ParameterFormat, ParameterStorageService
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/parameters", tags=["Parameter Storage"])


# ==================== Pydantic Models ====================


class CheckpointTypeEnum(str, Enum):
    """Checkpoint type options"""

    MANUAL = "manual"
    AUTO = "auto"
    BEST = "best"
    FINAL = "final"


class ParameterFormatEnum(str, Enum):
    """Parameter format options"""

    PYTORCH = "pytorch"
    NUMPY = "numpy"


class CreateCheckpointRequest(BaseModel):
    """Request to create a checkpoint"""

    checkpoint_type: CheckpointTypeEnum = Field(CheckpointTypeEnum.MANUAL)
    checkpoint_id: Optional[str] = Field(
        None, description="Custom checkpoint ID (auto-generated if not provided)"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Model metrics (loss, accuracy, etc.)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        schema_extra = {
            "example": {
                "checkpoint_type": "best",
                "metrics": {"loss": 0.25, "accuracy": 0.95},
                "metadata": {"epoch": 10, "learning_rate": 0.001},
            }
        }


class ParameterVersionResponse(BaseModel):
    """Parameter version information"""

    version_id: int
    model_id: str
    created_at: str
    checksum: str
    num_parameters: int
    total_size_bytes: int
    metadata: Dict[str, Any]


class CheckpointResponse(BaseModel):
    """Checkpoint information"""

    checkpoint_id: str
    model_id: str
    version_id: int
    checkpoint_type: str
    created_at: str
    checksum: str
    num_parameters: int
    size_bytes: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class LearningRateRequest(BaseModel):
    """Learning rate update request"""

    learning_rate: float = Field(..., gt=0, description="Learning rate value")


class ParameterDeltaResponse(BaseModel):
    """Delta between two versions"""

    from_version: int
    to_version: int
    changed_keys: List[str]
    delta_size_bytes: int
    compression_ratio: float


class StorageStatisticsResponse(BaseModel):
    """Storage statistics"""

    total_models: int
    total_versions: int
    total_checkpoints: int
    total_parameters: int
    total_size_bytes: int
    redis_enabled: bool


# ==================== Dependency ====================

_parameter_storage_service: Optional[ParameterStorageService] = None


def get_parameter_storage_service() -> ParameterStorageService:
    """Dependency to get parameter storage service"""
    if _parameter_storage_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Parameter storage service not initialized",
        )
    return _parameter_storage_service


def set_parameter_storage_service(service: ParameterStorageService):
    """Set global service instance"""
    global _parameter_storage_service
    _parameter_storage_service = service


# ==================== Endpoints ====================


@router.get(
    "/{model_id}",
    summary="Get current parameters",
    description="Get current parameter values for a model (metadata only, not actual tensors)",
)
async def get_parameters(
    model_id: str,
    version_id: Optional[int] = None,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Get current parameters"""
    # Get parameter names only (not actual tensors)
    param_names = service.get_parameter_names(model_id)

    if param_names is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
        )

    current_version = service.get_current_version(model_id)

    return {
        "model_id": model_id,
        "current_version": current_version,
        "requested_version": version_id,
        "parameter_names": param_names,
        "num_parameters": len(param_names),
    }


@router.get(
    "/{model_id}/names",
    response_model=List[str],
    summary="Get parameter names",
    description="Get list of parameter names for a model",
)
async def get_parameter_names(
    model_id: str, service: ParameterStorageService = Depends(get_parameter_storage_service)
):
    """Get parameter names"""
    param_names = service.get_parameter_names(model_id)

    if param_names is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
        )

    return param_names


@router.get(
    "/{model_id}/version",
    summary="Get current version",
    description="Get current version ID for a model",
)
async def get_current_version(
    model_id: str, service: ParameterStorageService = Depends(get_parameter_storage_service)
):
    """Get current version"""
    version_id = service.get_current_version(model_id)

    if version_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
        )

    return {"model_id": model_id, "current_version": version_id}


@router.get(
    "/{model_id}/versions",
    response_model=List[ParameterVersionResponse],
    summary="Get version history",
    description="Get version history for a model",
)
async def get_version_history(
    model_id: str,
    limit: Optional[int] = None,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Get version history"""
    versions = service.get_version_history(model_id, limit)

    return [
        ParameterVersionResponse(
            version_id=v.version_id,
            model_id=v.model_id,
            created_at=v.created_at.isoformat(),
            checksum=v.checksum,
            num_parameters=v.num_parameters,
            total_size_bytes=v.total_size_bytes,
            metadata=v.metadata,
        )
        for v in versions
    ]


@router.post(
    "/{model_id}/checkpoints",
    response_model=CheckpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a checkpoint",
    description="Create a checkpoint of current parameters",
)
async def create_checkpoint(
    model_id: str,
    request: CreateCheckpointRequest,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Create a checkpoint"""
    try:
        checkpoint = service.create_checkpoint(
            model_id=model_id,
            checkpoint_type=CheckpointType(request.checkpoint_type.value),
            checkpoint_id=request.checkpoint_id,
            metrics=request.metrics,
            metadata=request.metadata,
        )

        try:
            parameters = service.get_parameters(model_id)
            if parameters is not None:
                payload = pickle.dumps(parameters)
                client = ModelRegistryClient()
                await client.upload_checkpoint(
                    model_id=int(model_id),
                    checkpoint_type=checkpoint.checkpoint_type.value,
                    state_dict=payload,
                )
        except Exception:
            pass

        return CheckpointResponse(
            checkpoint_id=checkpoint.checkpoint_id,
            model_id=checkpoint.model_id,
            version_id=checkpoint.version_id,
            checkpoint_type=checkpoint.checkpoint_type.value,
            created_at=checkpoint.created_at.isoformat(),
            checksum=checkpoint.checksum,
            num_parameters=checkpoint.num_parameters,
            size_bytes=checkpoint.size_bytes,
            metrics=checkpoint.metrics,
            metadata=checkpoint.metadata,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create checkpoint: {str(e)}",
        )


@router.put("/{model_id}/learning-rate")
async def update_learning_rate(
    model_id: str,
    request: LearningRateRequest,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    try:
        if not service.enable_redis or not service.redis_client:
            raise HTTPException(status_code=503, detail="Redis unavailable")
        service.redis_client.set(f"lr:{model_id}", str(request.learning_rate))
        return {"model_id": model_id, "learning_rate": request.learning_rate}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{model_id}/checkpoints",
    response_model=List[CheckpointResponse],
    summary="List checkpoints",
    description="List all checkpoints for a model",
)
async def list_checkpoints(
    model_id: str,
    checkpoint_type: Optional[CheckpointTypeEnum] = None,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """List checkpoints"""
    checkpoint_type_filter = None
    if checkpoint_type:
        checkpoint_type_filter = CheckpointType(checkpoint_type.value)

    checkpoints = service.list_checkpoints(model_id, checkpoint_type_filter)

    return [
        CheckpointResponse(
            checkpoint_id=c.checkpoint_id,
            model_id=c.model_id,
            version_id=c.version_id,
            checkpoint_type=c.checkpoint_type.value,
            created_at=c.created_at.isoformat(),
            checksum=c.checksum,
            num_parameters=c.num_parameters,
            size_bytes=c.size_bytes,
            metrics=c.metrics,
            metadata=c.metadata,
        )
        for c in checkpoints
    ]


@router.post(
    "/{model_id}/checkpoints/{checkpoint_id}/restore",
    summary="Restore from checkpoint",
    description="Restore parameters from a checkpoint",
)
async def restore_checkpoint(
    model_id: str,
    checkpoint_id: str,
    restore_to_current: bool = True,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Restore from checkpoint"""
    parameters = service.load_checkpoint(
        model_id=model_id, checkpoint_id=checkpoint_id, restore_to_current=restore_to_current
    )

    if parameters is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint not found: {checkpoint_id}"
        )

    current_version = service.get_current_version(model_id)

    return {
        "model_id": model_id,
        "checkpoint_id": checkpoint_id,
        "restored": True,
        "current_version": current_version,
    }


@router.delete(
    "/{model_id}/checkpoints/{checkpoint_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a checkpoint",
    description="Delete a checkpoint from storage",
)
async def delete_checkpoint(
    model_id: str,
    checkpoint_id: str,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Delete a checkpoint"""
    deleted = service.delete_checkpoint(model_id, checkpoint_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint not found: {checkpoint_id}"
        )

    return None


@router.get(
    "/{model_id}/delta",
    response_model=ParameterDeltaResponse,
    summary="Calculate parameter delta",
    description="Calculate delta between two parameter versions",
)
async def calculate_delta(
    model_id: str,
    from_version: int,
    to_version: int,
    service: ParameterStorageService = Depends(get_parameter_storage_service),
):
    """Calculate parameter delta"""
    delta = service.calculate_delta(model_id, from_version, to_version)

    if delta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Versions not found: {from_version} or {to_version}",
        )

    return ParameterDeltaResponse(
        from_version=delta.from_version,
        to_version=delta.to_version,
        changed_keys=delta.changed_keys,
        delta_size_bytes=delta.delta_size_bytes,
        compression_ratio=delta.compression_ratio,
    )


@router.get(
    "/stats/summary",
    response_model=StorageStatisticsResponse,
    summary="Get storage statistics",
    description="Get statistics about parameter storage",
)
async def get_statistics(service: ParameterStorageService = Depends(get_parameter_storage_service)):
    """Get storage statistics"""
    stats = service.get_statistics()

    return StorageStatisticsResponse(**stats)


@router.get(
    "/health", summary="Health check", description="Check if parameter storage service is healthy"
)
async def health_check(service: ParameterStorageService = Depends(get_parameter_storage_service)):
    """Health check endpoint"""
    stats = service.get_statistics()

    return {
        "status": "healthy",
        "total_models": stats["total_models"],
        "redis_enabled": stats["redis_enabled"],
    }
