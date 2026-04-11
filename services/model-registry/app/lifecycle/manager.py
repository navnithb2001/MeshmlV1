"""
Model lifecycle management service
Handles state transitions and validation workflow
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Model, ModelState
from ..schemas import ModelStateUpdate

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages model lifecycle state transitions"""

    # Valid state transitions
    STATE_TRANSITIONS = {
        ModelState.UPLOADING: [ModelState.VALIDATING, ModelState.FAILED],
        ModelState.VALIDATING: [ModelState.READY, ModelState.FAILED],
        ModelState.READY: [ModelState.DEPRECATED],
        ModelState.FAILED: [ModelState.UPLOADING],  # Allow re-upload
        ModelState.DEPRECATED: [],  # Terminal state
    }

    def __init__(self, db: AsyncSession):
        self.db = db

    async def transition_state(
        self, model_id: int, new_state: ModelState, validation_message: Optional[str] = None
    ) -> Model:
        """
        Transition model to new state with validation

        Args:
            model_id: Model ID
            new_state: Target state
            validation_message: Optional message (especially for FAILED state)

        Returns:
            Updated Model object

        Raises:
            ValueError: If transition is invalid
        """
        # Get current model
        result = await self.db.execute(select(Model).where(Model.id == model_id))
        model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Check if transition is valid
        current_state = ModelState(model.state)
        if new_state not in self.STATE_TRANSITIONS.get(current_state, []):
            raise ValueError(f"Invalid state transition: {current_state} -> {new_state}")

        # Update state
        model.state = new_state
        model.validation_message = validation_message
        model.updated_at = datetime.utcnow()

        # Set deprecated timestamp if transitioning to DEPRECATED
        if new_state == ModelState.DEPRECATED:
            model.deprecated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(model)

        logger.info(f"Model {model_id} transitioned: {current_state} -> {new_state}")

        return model

    async def start_upload(self, model_id: int) -> Model:
        """Mark model as uploading"""
        return await self.transition_state(model_id, ModelState.UPLOADING)

    async def start_validation(self, model_id: int) -> Model:
        """Mark model as validating"""
        return await self.transition_state(model_id, ModelState.VALIDATING)

    async def mark_ready(self, model_id: int) -> Model:
        """Mark model as ready for use"""
        return await self.transition_state(model_id, ModelState.READY)

    async def mark_failed(self, model_id: int, error_message: str) -> Model:
        """Mark model as failed validation"""
        return await self.transition_state(
            model_id, ModelState.FAILED, validation_message=error_message
        )

    async def deprecate(self, model_id: int, reason: Optional[str] = None) -> Model:
        """Deprecate a model"""
        return await self.transition_state(
            model_id, ModelState.DEPRECATED, validation_message=reason
        )

    async def get_state(self, model_id: int) -> ModelState:
        """Get current model state"""
        result = await self.db.execute(select(Model.state).where(Model.id == model_id))
        state = result.scalar_one_or_none()

        if state is None:
            raise ValueError(f"Model {model_id} not found")

        return ModelState(state)

    async def can_transition(self, model_id: int, target_state: ModelState) -> bool:
        """Check if transition to target state is valid"""
        current_state = await self.get_state(model_id)
        return target_state in self.STATE_TRANSITIONS.get(current_state, [])

    async def get_models_by_state(
        self, state: ModelState, limit: int = 100, offset: int = 0
    ) -> list[Model]:
        """Get all models in a specific state"""
        result = await self.db.execute(
            select(Model)
            .where(Model.state == state)
            .limit(limit)
            .offset(offset)
            .order_by(Model.updated_at.desc())
        )
        return result.scalars().all()
