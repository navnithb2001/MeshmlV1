"""Worker database model"""

import uuid
from datetime import datetime

from app.utils.database import Base
from sqlalchemy import JSON, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID


class Worker(Base):
    """Worker registration and status model"""

    __tablename__ = "workers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    worker_id = Column(String(255), unique=True, nullable=False, index=True)
    user_email = Column(String(255), nullable=True, index=True)
    capabilities = Column(JSON, nullable=True)  # {device, gpu_name, ram_gb, etc.}
    status = Column(String(50), nullable=False, default="idle")  # idle, training, offline
    last_heartbeat = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Worker {self.worker_id} - {self.status}>"
