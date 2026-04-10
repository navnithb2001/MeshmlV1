"""Job database model"""

import uuid
from datetime import datetime

from app.utils.database import Base
from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship


class Job(Base):
    """Training job model"""

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey("groups.id"), nullable=False, index=True)
    model_id = Column(String(255), nullable=True)  # Reference to model registry
    dataset_id = Column(String(255), nullable=True)  # Reference to dataset
    config = Column(JSON, nullable=True)  # Training configuration
    status = Column(
        String(50), nullable=False, default="pending"
    )  # pending, running, completed, failed, cancelled
    progress = Column(JSON, nullable=True)  # {current_epoch, loss, accuracy, etc.}
    error_message = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    group = relationship("Group", back_populates="jobs")
    creator = relationship("User", back_populates="created_jobs")

    def __repr__(self):
        return f"<Job {self.id} - {self.status}>"
