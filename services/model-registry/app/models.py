"""
Database models for Model Registry
Extends the existing database models from Phase 1
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ModelState(str, enum.Enum):
    """Model lifecycle states"""

    UPLOADING = "uploading"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class Model(Base):
    """Model registry table - extends Phase 1 model"""

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)

    # Ownership - using UUID to match existing database schema
    # Note: Foreign keys are defined in the database, not in SQLAlchemy to avoid circular dependencies
    group_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    created_by_user_id = Column(UUID(as_uuid=True), nullable=False)

    # Storage
    gcs_path = Column(String(512), nullable=True)  # gs://meshml-models/models/{model_id}/model.py
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(64), nullable=True)  # SHA-256 hash

    # State & Lifecycle
    state = Column(
        String(50), default="uploading", index=True
    )  # Use String instead of Enum for compatibility
    validation_message = Column(Text, nullable=True)  # Error message if validation fails

    # Metadata
    architecture_type = Column(String(100), nullable=True, index=True)  # e.g., "CNN", "Transformer"
    dataset_type = Column(String(100), nullable=True, index=True)  # e.g., "ImageNet", "CIFAR-10"
    framework = Column(String(50), default="PyTorch")
    model_metadata = Column(JSON, nullable=True)  # MODEL_METADATA from uploaded file

    # Versioning
    version = Column(String(50), default="1.0.0")
    parent_model_id = Column(
        Integer, nullable=True, index=True
    )  # Self-referential, no FK constraint
    checkpoint_version = Column(Integer, default=0)  # Incremented on each checkpoint upload

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    deprecated_at = Column(DateTime, nullable=True)

    # Usage tracking
    usage_count = Column(Integer, default=0)  # How many jobs use this model
    download_count = Column(Integer, default=0)  # Download statistics


class ModelUsage(Base):
    """Track which jobs use which models"""

    __tablename__ = "model_usage"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False, index=True)  # No FK to avoid circular dependency
    user_id = Column(
        UUID(as_uuid=True), nullable=True, index=True
    )  # No FK to avoid circular dependency
    action = Column(String(50), nullable=True)  # 'download', 'use_in_job', etc.
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
