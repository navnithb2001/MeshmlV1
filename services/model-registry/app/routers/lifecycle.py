"""
Lifecycle router - Model state management
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..lifecycle import LifecycleManager
from ..models import Model, ModelState
from ..schemas import ModelResponse, ModelStateUpdate

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/states")
async def get_available_states():
    """Get all available model states"""
    return {
        "states": [state.value for state in ModelState],
        "transitions": LifecycleManager.STATE_TRANSITIONS,
    }


@router.get("/state/{state}", response_model=List[ModelResponse])
async def get_models_by_state(
    state: ModelState, limit: int = 100, offset: int = 0, db: AsyncSession = Depends(get_db_session)
):
    """Get all models in a specific state"""
    lifecycle = LifecycleManager(db)
    models = await lifecycle.get_models_by_state(state, limit, offset)

    return models


@router.post("/{model_id}/validate", response_model=ModelResponse)
async def start_validation(model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Transition model to VALIDATING state"""
    lifecycle = LifecycleManager(db)

    try:
        model = await lifecycle.start_validation(model_id)
        return model
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{model_id}/mark-ready", response_model=ModelResponse)
async def mark_model_ready(model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Transition model to READY state (validation passed)"""
    lifecycle = LifecycleManager(db)

    try:
        model = await lifecycle.mark_ready(model_id)
        logger.info(f"Model {model_id} marked as READY")
        return model
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{model_id}/mark-failed", response_model=ModelResponse)
async def mark_model_failed(
    model_id: int, error_message: str, db: AsyncSession = Depends(get_db_session)
):
    """Transition model to FAILED state (validation failed)"""
    lifecycle = LifecycleManager(db)

    try:
        model = await lifecycle.mark_failed(model_id, error_message)
        logger.warning(f"Model {model_id} marked as FAILED: {error_message}")
        return model
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{model_id}/deprecate", response_model=ModelResponse)
async def deprecate_model(
    model_id: int, reason: str = "Deprecated by user", db: AsyncSession = Depends(get_db_session)
):
    """Transition model to DEPRECATED state"""
    lifecycle = LifecycleManager(db)

    try:
        model = await lifecycle.deprecate(model_id, reason)
        logger.info(f"Model {model_id} deprecated: {reason}")
        return model
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{model_id}/can-transition/{target_state}")
async def check_transition_validity(
    model_id: int, target_state: ModelState, db: AsyncSession = Depends(get_db_session)
):
    """Check if a state transition is valid"""
    lifecycle = LifecycleManager(db)

    try:
        can_transition = await lifecycle.can_transition(model_id, target_state)
        current_state = await lifecycle.get_state(model_id)

        return {
            "model_id": model_id,
            "current_state": current_state,
            "target_state": target_state,
            "can_transition": can_transition,
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
