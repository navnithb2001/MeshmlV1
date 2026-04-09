"""
Gradient Aggregation API Router

RESTful endpoints for gradient submission, aggregation, and management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from app.services.gradient_aggregation import (
    AggregationConfig,
    AggregationStrategy,
    ClippingStrategy,
    GradientAggregationService,
    GradientUpdate,
)
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/gradients", tags=["gradients"])

# Global service instance
gradient_service = GradientAggregationService()


# ==================== Request/Response Models ====================


class TensorData(BaseModel):
    """Tensor data for API transfer"""

    shape: List[int]
    data: List[float]  # Flattened tensor data
    dtype: str = "float32"


class GradientSubmitRequest(BaseModel):
    """Request to submit gradients"""

    worker_id: str = Field(..., description="Worker identifier")
    model_id: str = Field(..., description="Model identifier")
    version_id: int = Field(..., description="Parameter version used for gradient computation")
    gradients: Dict[str, TensorData] = Field(..., description="Gradient tensors")
    num_samples: int = Field(..., gt=0, description="Number of samples used")
    loss: Optional[float] = Field(None, description="Training loss")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Additional metrics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GradientAggregateRequest(BaseModel):
    """Request to aggregate gradients"""

    model_id: str = Field(..., description="Model identifier")
    current_version: int = Field(..., description="Current parameter version")
    strategy: AggregationStrategy = Field(
        default=AggregationStrategy.FEDAVG, description="Aggregation strategy"
    )
    clipping_strategy: ClippingStrategy = Field(
        default=ClippingStrategy.NONE, description="Gradient clipping strategy"
    )
    clip_value: float = Field(default=1.0, gt=0, description="Clip value threshold")
    clip_norm: float = Field(default=1.0, gt=0, description="Clip norm threshold")
    staleness_weight_decay: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Staleness weight decay factor"
    )
    max_staleness: int = Field(default=10, ge=0, description="Maximum staleness allowed")
    normalize_gradients: bool = Field(default=False, description="Normalize gradients")
    momentum_factor: float = Field(default=0.9, ge=0.0, le=1.0, description="Momentum factor")
    adaptive_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Adaptive quality threshold"
    )
    clear_buffer: bool = Field(default=True, description="Clear buffer after aggregation")


class GradientUpdateResponse(BaseModel):
    """Response for gradient update info"""

    worker_id: str
    model_id: str
    version_id: int
    num_samples: int
    loss: Optional[float]
    metrics: Dict[str, float]
    received_at: datetime
    parameter_count: int


class AggregatedGradientResponse(BaseModel):
    """Response for aggregated gradients"""

    model_id: str
    target_version_id: int
    num_workers: int
    total_samples: int
    worker_ids: List[str]
    strategy: str
    staleness_weights: Dict[str, float]
    created_at: datetime
    parameter_count: int
    metadata: Dict[str, Any]


class PendingGradientsResponse(BaseModel):
    """Response for pending gradients"""

    model_id: str
    pending_count: int
    gradients: List[GradientUpdateResponse]


class StatisticsResponse(BaseModel):
    """Response for aggregation statistics"""

    total_aggregations: int
    strategy_counts: Dict[str, int]
    pending_gradients: int
    models_with_pending: int
    models_with_momentum: int


# ==================== Helper Functions ====================


def tensor_data_to_torch(tensor_data: TensorData) -> torch.Tensor:
    """Convert TensorData to PyTorch tensor"""
    tensor = torch.tensor(tensor_data.data, dtype=getattr(torch, tensor_data.dtype))
    tensor = tensor.reshape(tensor_data.shape)
    return tensor


def torch_to_tensor_data(tensor: torch.Tensor) -> TensorData:
    """Convert PyTorch tensor to TensorData"""
    return TensorData(
        shape=list(tensor.shape),
        data=tensor.flatten().tolist(),
        dtype=str(tensor.dtype).replace("torch.", ""),
    )


# ==================== Endpoints ====================


@router.post("/submit", response_model=Dict[str, Any])
async def submit_gradients(request: GradientSubmitRequest) -> Dict[str, Any]:
    """
    Submit gradient update from a worker.

    The worker sends gradients computed from a specific parameter version.
    These gradients are buffered for later aggregation.
    """
    try:
        # Convert gradient tensors
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

        # Submit to service
        gradient_service.submit_gradient(gradient_update)

        return {
            "status": "success",
            "message": f"Gradient accepted from {request.worker_id}",
            "worker_id": request.worker_id,
            "model_id": request.model_id,
            "version_id": request.version_id,
            "num_samples": request.num_samples,
            "parameter_count": len(gradients),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/aggregate", response_model=AggregatedGradientResponse)
async def aggregate_gradients(request: GradientAggregateRequest) -> AggregatedGradientResponse:
    """
    Aggregate pending gradients for a model.

    Combines gradients from multiple workers using the specified strategy,
    applies staleness weighting and gradient clipping/normalization.
    """
    try:
        # Create aggregation config
        config = AggregationConfig(
            strategy=request.strategy,
            clipping_strategy=request.clipping_strategy,
            clip_value=request.clip_value,
            clip_norm=request.clip_norm,
            staleness_weight_decay=request.staleness_weight_decay,
            max_staleness=request.max_staleness,
            normalize_gradients=request.normalize_gradients,
            momentum_factor=request.momentum_factor,
            adaptive_threshold=request.adaptive_threshold,
        )

        # Aggregate
        result = gradient_service.aggregate_gradients(
            model_id=request.model_id,
            current_version=request.current_version,
            config=config,
            clear_buffer=request.clear_buffer,
        )

        if result is None:
            raise HTTPException(
                status_code=404, detail=f"No gradients to aggregate for {request.model_id}"
            )

        return AggregatedGradientResponse(
            model_id=result.model_id,
            target_version_id=result.target_version_id,
            num_workers=result.num_workers,
            total_samples=result.total_samples,
            worker_ids=result.worker_ids,
            strategy=result.strategy.value,
            staleness_weights=result.staleness_weights,
            created_at=result.created_at,
            parameter_count=len(result.aggregated_gradients),
            metadata=result.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pending/{model_id}", response_model=PendingGradientsResponse)
async def get_pending_gradients(model_id: str) -> PendingGradientsResponse:
    """
    Get pending gradient updates for a model.

    Returns gradients that have been submitted but not yet aggregated.
    """
    try:
        pending = gradient_service.get_pending_gradients(model_id)

        gradient_responses = [
            GradientUpdateResponse(
                worker_id=update.worker_id,
                model_id=update.model_id,
                version_id=update.version_id,
                num_samples=update.num_samples,
                loss=update.loss,
                metrics=update.metrics,
                received_at=update.received_at,
                parameter_count=len(update.gradients),
            )
            for update in pending
        ]

        return PendingGradientsResponse(
            model_id=model_id, pending_count=len(pending), gradients=gradient_responses
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pending/{model_id}", response_model=Dict[str, Any])
async def clear_pending_gradients(model_id: str) -> Dict[str, Any]:
    """
    Clear pending gradients for a model.

    Removes all buffered gradients without aggregating them.
    """
    try:
        count = gradient_service.clear_buffer(model_id)

        return {
            "status": "success",
            "message": f"Cleared {count} pending gradients",
            "model_id": model_id,
            "cleared_count": count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[AggregatedGradientResponse])
async def get_aggregation_history(
    model_id: Optional[str] = None, limit: Optional[int] = None
) -> List[AggregatedGradientResponse]:
    """
    Get aggregation history.

    Query parameters:
    - model_id: Filter by model (optional)
    - limit: Maximum number of records (optional)
    """
    try:
        history = gradient_service.get_aggregation_history(model_id=model_id, limit=limit)

        return [
            AggregatedGradientResponse(
                model_id=agg.model_id,
                target_version_id=agg.target_version_id,
                num_workers=agg.num_workers,
                total_samples=agg.total_samples,
                worker_ids=agg.worker_ids,
                strategy=agg.strategy.value,
                staleness_weights=agg.staleness_weights,
                created_at=agg.created_at,
                parameter_count=len(agg.aggregated_gradients),
                metadata=agg.metadata,
            )
            for agg in history
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies", response_model=Dict[str, List[str]])
async def get_strategies() -> Dict[str, List[str]]:
    """
    Get available aggregation and clipping strategies.
    """
    return {
        "aggregation_strategies": [s.value for s in AggregationStrategy],
        "clipping_strategies": [s.value for s in ClippingStrategy],
    }


@router.get("/stats/summary", response_model=StatisticsResponse)
async def get_statistics() -> StatisticsResponse:
    """
    Get gradient aggregation statistics.
    """
    try:
        stats = gradient_service.get_statistics()
        return StatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for gradient aggregation service.
    """
    stats = gradient_service.get_statistics()

    return {
        "status": "healthy",
        "service": "gradient_aggregation",
        "total_aggregations": stats["total_aggregations"],
        "pending_gradients": stats["pending_gradients"],
    }
