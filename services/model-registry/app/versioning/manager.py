"""
Model versioning service
Handles version management and parent-child relationships
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Model, ModelState
from ..schemas import ModelVersionCreate

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages model versioning and relationships"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_version(
        self,
        parent_model_id: int,
        name: str,
        version: str,
        created_by_user_id: UUID,  # Changed to UUID
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Model:
        """
        Create a new version of an existing model

        Args:
            parent_model_id: ID of parent model
            name: Name for new version
            version: Version string (e.g., "1.1.0")
            created_by_user_id: User creating the version
            description: Optional description
            metadata: Optional metadata

        Returns:
            New Model object
        """
        # Validate parent model exists and is ready
        result = await self.db.execute(select(Model).where(Model.id == parent_model_id))
        parent = result.scalar_one_or_none()

        if not parent:
            raise ValueError(f"Parent model {parent_model_id} not found")

        if parent.state != ModelState.READY:
            raise ValueError(f"Parent model must be in READY state, currently {parent.state}")

        # Check if version already exists for this parent
        existing = await self.db.execute(
            select(Model).where(
                and_(Model.parent_model_id == parent_model_id, Model.version == version)
            )
        )

        if existing.scalar_one_or_none():
            raise ValueError(f"Version {version} already exists for model {parent_model_id}")

        # Create new model version
        new_model = Model(
            name=name,
            description=description or parent.description,
            group_id=parent.group_id,  # Inherit group
            created_by_user_id=created_by_user_id,
            architecture_type=parent.architecture_type,  # Inherit architecture
            dataset_type=parent.dataset_type,  # Inherit dataset type
            framework=parent.framework,
            model_metadata=metadata or parent.model_metadata,
            version=version,
            parent_model_id=parent_model_id,
            state=ModelState.UPLOADING,
        )

        self.db.add(new_model)
        await self.db.commit()
        await self.db.refresh(new_model)

        logger.info(f"Created version {version} of model {parent_model_id} -> {new_model.id}")

        return new_model

    async def get_versions(
        self, parent_model_id: int, include_deprecated: bool = False
    ) -> List[Model]:
        """
        Get all versions of a model

        Args:
            parent_model_id: Parent model ID
            include_deprecated: Include deprecated versions

        Returns:
            List of model versions
        """
        query = select(Model).where(Model.parent_model_id == parent_model_id)

        if not include_deprecated:
            query = query.where(Model.state != ModelState.DEPRECATED)

        query = query.order_by(Model.created_at.desc())

        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_latest_version(self, parent_model_id: int) -> Optional[Model]:
        """
        Get the latest ready version of a model

        Args:
            parent_model_id: Parent model ID

        Returns:
            Latest model version or None
        """
        result = await self.db.execute(
            select(Model)
            .where(and_(Model.parent_model_id == parent_model_id, Model.state == ModelState.READY))
            .order_by(Model.created_at.desc())
            .limit(1)
        )

        return result.scalar_one_or_none()

    async def get_version_history(self, model_id: int) -> List[Model]:
        """
        Get full version history (all ancestors and descendants)

        Args:
            model_id: Model ID

        Returns:
            List of all related model versions
        """
        # Get the model
        result = await self.db.execute(select(Model).where(Model.id == model_id))
        model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Find root (original) model
        root_id = model.parent_model_id or model.id

        # Get all versions with this root
        result = await self.db.execute(
            select(Model)
            .where((Model.id == root_id) | (Model.parent_model_id == root_id))
            .order_by(Model.created_at.asc())
        )

        return result.scalars().all()

    async def get_version_count(self, parent_model_id: int) -> int:
        """Get count of versions for a model"""
        result = await self.db.execute(
            select(Model).where(Model.parent_model_id == parent_model_id)
        )
        return len(result.scalars().all())

    def parse_version(self, version_str: str) -> tuple[int, int, int]:
        """Parse semantic version string"""
        try:
            parts = version_str.split(".")
            if len(parts) != 3:
                raise ValueError("Version must be in format X.Y.Z")
            return tuple(int(p) for p in parts)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid version format: {version_str}")

    def increment_version(
        self, current_version: str, part: str = "patch"  # major, minor, or patch
    ) -> str:
        """
        Increment a semantic version

        Args:
            current_version: Current version string
            part: Which part to increment (major/minor/patch)

        Returns:
            New version string
        """
        major, minor, patch = self.parse_version(current_version)

        if part == "major":
            return f"{major + 1}.0.0"
        elif part == "minor":
            return f"{major}.{minor + 1}.0"
        elif part == "patch":
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Invalid version part: {part}")

    async def suggest_next_version(self, parent_model_id: int) -> str:
        """Suggest next version number based on existing versions"""
        versions = await self.get_versions(parent_model_id)

        if not versions:
            # Get parent's version and increment
            result = await self.db.execute(select(Model.version).where(Model.id == parent_model_id))
            parent_version = result.scalar_one_or_none()

            if parent_version:
                return self.increment_version(parent_version, "minor")
            return "1.0.0"

        # Get highest version and increment
        latest_version = max(versions, key=lambda m: self.parse_version(m.version)).version

        return self.increment_version(latest_version, "patch")
