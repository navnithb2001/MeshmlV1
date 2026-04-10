"""Task Orchestrator DB models."""

from datetime import datetime

from app.db import Base
from sqlalchemy import Column, DateTime, Integer, String


class DataBatch(Base):
    """Represents a data batch ready for assignment."""

    __tablename__ = "data_batches"

    id = Column(String(255), primary_key=True)
    job_id = Column(String(255), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    gcs_path = Column(String(1024), nullable=True)
    status = Column(
        String(50), nullable=False, index=True
    )  # AVAILABLE, ASSIGNED, COMPLETED, FAILED
    assigned_worker_id = Column(String(255), nullable=True, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
