"""gRPC server for Metrics Service."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import grpc
import redis.asyncio as redis
from app.db import AsyncSessionLocal
from app.models import MetricPoint as MetricPointModel
from app.proto import metrics_pb2, metrics_pb2_grpc
from sqlalchemy import insert

logger = logging.getLogger(__name__)


class MetricsService(metrics_pb2_grpc.MetricsServiceServicer):
    def __init__(self, redis_client: Optional[redis.Redis]):
        self.redis_client = redis_client

    async def StreamMetrics(self, request_iterator, context):
        try:
            async for metric in request_iterator:
                payload = {
                    "job_id": metric.job_id,
                    "step": metric.step,
                    "loss": metric.loss,
                    "accuracy": metric.accuracy,
                    "timestamp_ms": metric.timestamp_ms,
                    "worker_id": metric.worker_id,
                }

                if self.redis_client:
                    channel = f"live_stats:{metric.job_id}"
                    await self.redis_client.publish(channel, json.dumps(payload))

                async with AsyncSessionLocal() as session:
                    if metric.timestamp_ms:
                        timestamp = datetime.fromtimestamp(
                            metric.timestamp_ms / 1000.0, tz=timezone.utc
                        )
                    else:
                        timestamp = datetime.now(tz=timezone.utc)
                    await session.execute(
                        insert(MetricPointModel).values(
                            job_id=metric.job_id,
                            step=metric.step,
                            loss=metric.loss,
                            accuracy=metric.accuracy,
                            timestamp=timestamp,
                        )
                    )
                    await session.commit()

            return metrics_pb2.MetricsAck(success=True, message="metrics ingested")
        except Exception as e:
            logger.error(f"Metrics stream failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


async def start_grpc_server(app, host: str, port: int) -> None:
    server = grpc.aio.server()
    redis_client = getattr(app.state, "redis_client", None)
    metrics_pb2_grpc.add_MetricsServiceServicer_to_server(MetricsService(redis_client), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    app.state.grpc_server = server
    asyncio.create_task(server.wait_for_termination())
