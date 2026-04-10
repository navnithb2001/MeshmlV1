"""Dataset database model"""

import uuid
from datetime import datetime

from app.utils.database import Base
from sqlalchemy import JSON, BigInteger, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship


class Dataset(Base):
    """Dataset model for uploaded/managed datasets"""

    __tablename__ = "datasets"

    id = Column(String(255), primary_key=True)  # UUID or custom ID
    name = Column(String(255), nullable=False, index=True)
    format = Column(String(50), nullable=False)  # imagefolder, coco, csv, etc.
    upload_type = Column(String(50), nullable=False)  # file_upload, url_download
    source_url = Column(Text, nullable=True)  # Original URL if downloaded
    local_path = Column(Text, nullable=True)  # Local filesystem path
    gcs_path = Column(Text, nullable=True)  # GCS path

    # Size and content info
    total_size_bytes = Column(BigInteger, nullable=True)
    file_count = Column(Integer, nullable=True)
    num_samples = Column(Integer, nullable=True)  # Total number of samples
    num_classes = Column(Integer, nullable=True)  # Number of classes

    # Sharding info
    num_shards = Column(Integer, nullable=True)  # Number of shards created
    shard_strategy = Column(String(50), nullable=True)  # random, stratified, non_iid
    sharded_at = Column(DateTime, nullable=True)  # When sharding completed

    # Status tracking
    status = Column(String(50), nullable=False, default="uploading", index=True)
    # Status values: uploading, downloaded, uploaded, validating, sharding, ready, failed

    error_message = Column(Text, nullable=True)  # Error details if failed

    # Metadata
    dataset_metadata = Column(JSON, nullable=True)  # Additional info (class names, etc.)

    # Ownership
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    uploader = relationship("User", back_populates="datasets")

    def __repr__(self):
        return f"<Dataset {self.id} - {self.name} ({self.status})>"
