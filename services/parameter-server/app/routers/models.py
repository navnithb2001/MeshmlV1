"""
Model Initialization API Router

HTTP endpoints for model initialization and management in Parameter Server.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.services.model_initializer import (
    InitializationStrategy,
    ModelConfig,
    ModelInitializerService,
    ModelStatus,
)
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/models", tags=["Model Initialization"])


# ==================== Pydantic Models ====================


class InitializationStrategyEnum(str, Enum):
    """Initialization strategy options"""

    RANDOM = "random"
    PRETRAINED = "pretrained"
    ZEROS = "zeros"
    ONES = "ones"
    XAVIER = "xavier_uniform"
    KAIMING = "kaiming_normal"


class InitializeModelRequest(BaseModel):
    """Request to initialize a model"""

    model_id: str = Field(..., description="Unique model identifier")
    gcs_model_path: str = Field(..., description="GCS path to model.py (gs://bucket/path/model.py)")
    initialization_strategy: InitializationStrategyEnum = Field(
        InitializationStrategyEnum.RANDOM, description="Weight initialization strategy"
    )
    pretrained_weights_path: Optional[str] = Field(
        None, description="GCS path to pretrained weights (.pt/.pth file)"
    )
    device: str = Field("cpu", description="Device for model (cpu, cuda, mps)")
    seed: Optional[int] = Field(None, description="Random seed for initialization")
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for create_model() function"
    )

    class Config:
        schema_extra = {
            "example": {
                "model_id": "resnet18_v1",
                "gcs_model_path": "gs://my-bucket/models/resnet18/model.py",
                "initialization_strategy": "random",
                "device": "cuda",
                "seed": 42,
                "model_kwargs": {"num_classes": 10, "pretrained": False},
            }
        }


class ReinitializeModelRequest(BaseModel):
    """Request to reinitialize model weights"""

    initialization_strategy: Optional[InitializationStrategyEnum] = Field(
        None, description="New initialization strategy (optional)"
    )


class ModelInfoResponse(BaseModel):
    """Model information response"""

    model_id: str
    status: str
    metadata: Optional[Dict[str, Any]]
    num_parameters: int
    device: str
    checksum: Optional[str]
    created_at: str
    updated_at: str
    error_message: Optional[str]


class ModelStatisticsResponse(BaseModel):
    """Service statistics response"""

    total_models: int
    status_counts: Dict[str, int]
    total_parameters: int
    default_device: str


# ==================== Dependency ====================

# Global service instance (initialized by main.py)
_model_initializer_service: Optional[ModelInitializerService] = None


def get_model_initializer_service() -> ModelInitializerService:
    """Dependency to get model initializer service"""
    if _model_initializer_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model initializer service not initialized",
        )
    return _model_initializer_service


def set_model_initializer_service(service: ModelInitializerService):
    """Set global service instance"""
    global _model_initializer_service
    _model_initializer_service = service


# ==================== Endpoints ====================


@router.post(
    "/initialize",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize a model from GCS",
    description="""
    Initialize a PyTorch model from custom code in GCS.
    
    Workflow:
    1. Download model.py from GCS
    2. Import and validate create_model() and MODEL_METADATA
    3. Instantiate model
    4. Initialize weights (random or pretrained)
    5. Move to device (CPU/GPU)
    6. Register in parameter server
    """,
)
async def initialize_model(
    request: InitializeModelRequest,
    service: ModelInitializerService = Depends(get_model_initializer_service),
):
    """Initialize a model from GCS"""
    try:
        # Create config
        config = ModelConfig(
            model_id=request.model_id,
            gcs_model_path=request.gcs_model_path,
            initialization_strategy=InitializationStrategy(request.initialization_strategy.value),
            pretrained_weights_path=request.pretrained_weights_path,
            device=request.device,
            seed=request.seed,
            model_kwargs=request.model_kwargs,
        )

        # Initialize model
        initialized_model = await service.initialize_model(config)

        # Return model info
        model_info = service.get_model_info(request.model_id)

        return ModelInfoResponse(**model_info)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid model configuration: {str(e)}"
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model initialization failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected error: {str(e)}"
        )


@router.get(
    "/{model_id}",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get detailed information about an initialized model (without weights)",
)
async def get_model(
    model_id: str, service: ModelInitializerService = Depends(get_model_initializer_service)
):
    """Get model information"""
    model_info = service.get_model_info(model_id)

    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
        )

    return ModelInfoResponse(**model_info)


@router.get(
    "/",
    response_model=List[ModelInfoResponse],
    summary="List all models",
    description="List all initialized models in the parameter server",
)
async def list_models(service: ModelInitializerService = Depends(get_model_initializer_service)):
    """List all models"""
    models = service.list_models()

    return [ModelInfoResponse(**service.get_model_info(model_id)) for model_id in models.keys()]


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a model",
    description="Delete a model from the parameter server registry",
)
async def delete_model(
    model_id: str, service: ModelInitializerService = Depends(get_model_initializer_service)
):
    """Delete a model"""
    deleted = service.delete_model(model_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model not found: {model_id}"
        )

    return None


@router.post(
    "/{model_id}/reinitialize",
    response_model=ModelInfoResponse,
    summary="Reinitialize model weights",
    description="""
    Reinitialize model weights with a new strategy.
    
    Useful for:
    - Resetting model to initial state
    - Trying different initialization strategies
    - Re-running experiments
    """,
)
async def reinitialize_model(
    model_id: str,
    request: ReinitializeModelRequest,
    service: ModelInitializerService = Depends(get_model_initializer_service),
):
    """Reinitialize model weights"""
    try:
        new_strategy = None
        if request.initialization_strategy:
            new_strategy = InitializationStrategy(request.initialization_strategy.value)

        initialized_model = await service.reinitialize_model(model_id, new_strategy)

        model_info = service.get_model_info(model_id)

        return ModelInfoResponse(**model_info)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reinitialization failed: {str(e)}",
        )


@router.get(
    "/stats/summary",
    response_model=ModelStatisticsResponse,
    summary="Get service statistics",
    description="Get statistics about initialized models and service status",
)
async def get_statistics(service: ModelInitializerService = Depends(get_model_initializer_service)):
    """Get service statistics"""
    stats = service.get_statistics()

    return ModelStatisticsResponse(**stats)


@router.get(
    "/health", summary="Health check", description="Check if model initializer service is healthy"
)
async def health_check(service: ModelInitializerService = Depends(get_model_initializer_service)):
    """Health check endpoint"""
    stats = service.get_statistics()

    return {
        "status": "healthy",
        "total_models": stats["total_models"],
        "default_device": stats["default_device"],
    }
