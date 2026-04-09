"""
Convergence Detection API Router

RESTful endpoints for monitoring training convergence and early stopping.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.services.convergence_detection import (
    ConvergenceConfig,
    ConvergenceCriterion,
    ConvergenceDetectionService,
    MetricDirection,
    TrainingMetrics,
    TrainingPhase,
)
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/convergence", tags=["convergence"])

# Global service instance
convergence_service = ConvergenceDetectionService()


# ==================== Request/Response Models ====================


class MetricsUpdateRequest(BaseModel):
    """Request to update training metrics"""

    model_id: str = Field(..., description="Model identifier")
    iteration: int = Field(..., ge=0, description="Training iteration")
    loss: float = Field(..., description="Training loss")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Additional metrics")
    gradient_norm: Optional[float] = Field(default=None, description="Gradient norm")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    num_samples: int = Field(default=0, ge=0, description="Number of samples")


class ConvergenceConfigRequest(BaseModel):
    """Request to configure convergence detection"""

    loss_threshold: Optional[float] = Field(
        default=None, description="Loss threshold for convergence"
    )
    loss_patience: int = Field(default=10, ge=1, description="Loss patience (iterations)")
    loss_min_delta: float = Field(default=1e-4, ge=0, description="Minimum loss improvement")

    target_metrics: Dict[str, Tuple[float, str]] = Field(
        default_factory=dict, description="Target metrics: {name: (value, direction)}"
    )
    metric_patience: int = Field(default=10, ge=1, description="Metric patience")
    metric_min_delta: float = Field(default=1e-4, ge=0, description="Minimum metric improvement")

    enable_plateau_detection: bool = Field(default=True, description="Enable plateau detection")
    plateau_patience: int = Field(default=20, ge=1, description="Plateau patience")
    plateau_threshold: float = Field(default=1e-3, ge=0, description="Plateau variance threshold")

    gradient_norm_threshold: Optional[float] = Field(
        default=None, description="Gradient norm threshold"
    )

    max_iterations: Optional[int] = Field(default=None, description="Maximum iterations")
    warmup_iterations: int = Field(default=5, ge=0, description="Warmup iterations")
    window_size: int = Field(default=10, ge=1, description="Analysis window size")

    enable_early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stop_patience: int = Field(default=50, ge=1, description="Early stop patience")


class TrainingMetricsResponse(BaseModel):
    """Response for training metrics"""

    iteration: int
    loss: float
    metrics: Dict[str, float]
    gradient_norm: Optional[float]
    learning_rate: Optional[float]
    num_samples: int
    timestamp: datetime


class ConvergenceResultResponse(BaseModel):
    """Response for convergence check result"""

    converged: bool
    should_stop: bool
    phase: str
    criteria_met: List[str]
    current_iteration: int
    best_iteration: int
    best_loss: float
    best_metrics: Dict[str, float]
    iterations_without_improvement: int
    estimated_iterations_remaining: Optional[int]
    message: str
    metadata: Dict[str, Any]


class ConvergenceSummaryResponse(BaseModel):
    """Response for convergence summary"""

    model_id: str
    phase: str
    current_iteration: int
    best_iteration: int
    current_loss: float
    best_loss: float
    best_metrics: Dict[str, float]
    iterations_without_improvement: int
    total_iterations: int
    improvement_rate: float
    is_plateaued: bool


class ConvergenceStatisticsResponse(BaseModel):
    """Response for convergence statistics"""

    total_models_tracked: int
    phase_distribution: Dict[str, int]
    recent_convergences: int
    recent_early_stops: int
    total_convergence_checks: int


# ==================== Helper Functions ====================


def config_from_request(req: ConvergenceConfigRequest) -> ConvergenceConfig:
    """Convert request to ConvergenceConfig"""
    # Convert target_metrics to proper format
    target_metrics = {}
    for name, (value, direction_str) in req.target_metrics.items():
        direction = MetricDirection(direction_str)
        target_metrics[name] = (value, direction)

    return ConvergenceConfig(
        loss_threshold=req.loss_threshold,
        loss_patience=req.loss_patience,
        loss_min_delta=req.loss_min_delta,
        target_metrics=target_metrics,
        metric_patience=req.metric_patience,
        metric_min_delta=req.metric_min_delta,
        enable_plateau_detection=req.enable_plateau_detection,
        plateau_patience=req.plateau_patience,
        plateau_threshold=req.plateau_threshold,
        gradient_norm_threshold=req.gradient_norm_threshold,
        max_iterations=req.max_iterations,
        warmup_iterations=req.warmup_iterations,
        window_size=req.window_size,
        enable_early_stopping=req.enable_early_stopping,
        early_stop_patience=req.early_stop_patience,
    )


# ==================== Endpoints ====================


@router.post("/metrics", response_model=ConvergenceResultResponse)
async def update_metrics(
    request: MetricsUpdateRequest, config: Optional[ConvergenceConfigRequest] = None
) -> ConvergenceResultResponse:
    """
    Update training metrics and check convergence.

    This endpoint should be called after each training iteration
    to track progress and detect convergence.
    """
    try:
        # Create TrainingMetrics
        metrics = TrainingMetrics(
            iteration=request.iteration,
            loss=request.loss,
            metrics=request.metrics,
            gradient_norm=request.gradient_norm,
            learning_rate=request.learning_rate,
            num_samples=request.num_samples,
        )

        # Get config
        conv_config = None
        if config:
            conv_config = config_from_request(config)

        # Update and check convergence
        result = convergence_service.update_metrics(
            model_id=request.model_id, metrics=metrics, config=conv_config
        )

        return ConvergenceResultResponse(
            converged=result.converged,
            should_stop=result.should_stop,
            phase=result.phase.value,
            criteria_met=[c.value for c in result.criteria_met],
            current_iteration=result.current_iteration,
            best_iteration=result.best_iteration,
            best_loss=result.best_loss,
            best_metrics=result.best_metrics,
            iterations_without_improvement=result.iterations_without_improvement,
            estimated_iterations_remaining=result.estimated_iterations_remaining,
            message=result.message,
            metadata=result.metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary/{model_id}", response_model=ConvergenceSummaryResponse)
async def get_convergence_summary(model_id: str) -> ConvergenceSummaryResponse:
    """
    Get convergence summary for a model.

    Provides overview of training progress and convergence status.
    """
    try:
        summary = convergence_service.get_convergence_summary(model_id)

        if "status" in summary:
            # Model not started or no metrics yet
            raise HTTPException(status_code=404, detail=f"No convergence data for {model_id}")

        return ConvergenceSummaryResponse(**summary)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{model_id}", response_model=List[TrainingMetricsResponse])
async def get_metrics_history(
    model_id: str, limit: Optional[int] = None
) -> List[TrainingMetricsResponse]:
    """
    Get training metrics history for a model.

    Query parameters:
    - limit: Maximum number of records (optional)

    Returns metrics in reverse chronological order (newest first).
    """
    try:
        history = convergence_service.get_metrics_history(model_id=model_id, limit=limit)

        return [
            TrainingMetricsResponse(
                iteration=m.iteration,
                loss=m.loss,
                metrics=m.metrics,
                gradient_norm=m.gradient_norm,
                learning_rate=m.learning_rate,
                num_samples=m.num_samples,
                timestamp=m.timestamp,
            )
            for m in history
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset/{model_id}", response_model=Dict[str, Any])
async def reset_training(model_id: str) -> Dict[str, Any]:
    """
    Reset training state for a model.

    Clears all metrics history and convergence tracking.
    """
    try:
        success = convergence_service.reset_training(model_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        return {
            "status": "success",
            "message": f"Training state reset for {model_id}",
            "model_id": model_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phases", response_model=Dict[str, List[str]])
async def get_phases_and_criteria() -> Dict[str, List[str]]:
    """
    Get available training phases and convergence criteria.
    """
    return {
        "training_phases": [phase.value for phase in TrainingPhase],
        "convergence_criteria": [criterion.value for criterion in ConvergenceCriterion],
        "metric_directions": [direction.value for direction in MetricDirection],
    }


@router.get("/stats/summary", response_model=ConvergenceStatisticsResponse)
async def get_statistics() -> ConvergenceStatisticsResponse:
    """
    Get convergence detection service statistics.
    """
    try:
        stats = convergence_service.get_statistics()
        return ConvergenceStatisticsResponse(**stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for convergence detection service.
    """
    stats = convergence_service.get_statistics()

    return {
        "status": "healthy",
        "service": "convergence_detection",
        "models_tracked": stats["total_models_tracked"],
        "total_checks": stats["total_convergence_checks"],
    }
