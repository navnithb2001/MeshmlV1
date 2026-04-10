from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from app.db import Base
from app.grpc_server import MetricsService
from app.models import MetricPoint as MetricPointModel
from app.proto import metrics_pb2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


class FakeSession:
    def __init__(self):
        self.executed = 0
        self.committed = 0

    async def execute(self, _stmt):
        self.executed += 1

    async def commit(self):
        self.committed += 1


class FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _metric_stream():
    yield metrics_pb2.MetricPoint(
        job_id="job-1",
        step=1,
        loss=0.5,
        accuracy=0.8,
        timestamp_ms=0,
        worker_id="worker-1",
    )


@pytest.mark.asyncio
async def test_stream_metrics_publishes_and_persists():
    fake_session = FakeSession()
    redis_client = SimpleNamespace(publish=AsyncMock())

    with patch("app.grpc_server.AsyncSessionLocal", return_value=FakeSessionContext(fake_session)):
        service = MetricsService(redis_client=redis_client)

        response = await service.StreamMetrics(_metric_stream(), context=None)

        assert response.success is True
        redis_client.publish.assert_awaited_once()
        assert fake_session.executed == 1
        assert fake_session.committed == 1


@pytest.mark.asyncio
async def test_metric_model_works_with_sqlite_in_memory() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with session_maker() as session:
            session.add(MetricPointModel(job_id="job-2", step=2, loss=0.2, accuracy=0.9))
            await session.commit()

        async with session_maker() as session:
            result = await session.execute(
                select(MetricPointModel).where(MetricPointModel.job_id == "job-2")
            )
            row = result.scalar_one()
            assert row.step == 2
    finally:
        await engine.dispose()
