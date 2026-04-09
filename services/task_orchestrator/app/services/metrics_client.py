"""Metrics Service gRPC client for Task Orchestrator."""

import logging
import os
import time

import grpc
from app.proto import metrics_pb2, metrics_pb2_grpc

logger = logging.getLogger(__name__)


class MetricsClient:
    def __init__(self, grpc_url: str | None = None):
        self.grpc_url = grpc_url or os.getenv("METRICS_SERVICE_GRPC_URL", "metrics-service:50055")

    async def send_job_finished(self, job_id: str) -> None:
        try:
            async with grpc.aio.insecure_channel(self.grpc_url) as channel:
                stub = metrics_pb2_grpc.MetricsServiceStub(channel)
                call = stub.StreamMetrics()
                await call.write(
                    metrics_pb2.MetricPoint(
                        job_id=job_id,
                        step=-1,
                        loss=0.0,
                        accuracy=0.0,
                        timestamp_ms=int(time.time() * 1000),
                        worker_id="JOB_FINISHED",
                    )
                )
                await call.done_writing()
                await call
        except Exception as e:
            logger.warning(f"Failed to send JOB_FINISHED metric: {e}")
