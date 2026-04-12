"""Database models for Metrics Service."""

from app.db import Base
from sqlalchemy import Column, DateTime, Float, Integer, String, func


class MetricPoint(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, nullable=False, index=True)
    step = Column(Integer, nullable=False, index=True)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    timestamp = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
