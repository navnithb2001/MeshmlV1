"""Metrics Service gRPC client."""

import asyncio
import logging
from typing import Optional

import grpc
from meshml_worker.proto import metrics_pb2, metrics_pb2_grpc

logger = logging.getLogger(__name__)


class MetricsClient:
    def __init__(self, grpc_url: str):
        self.grpc_url = grpc_url
        self._channel: Optional[grpc.aio.Channel] = None
        self._call = None
        self._job_id: Optional[str] = None
        self._worker_id: Optional[str] = None
        self._lock = asyncio.Lock()

    async def start(self, job_id: str, worker_id: str) -> None:
        async with self._lock:
            if self._call:
                return
            self._job_id = job_id
            self._worker_id = worker_id
            self._channel = grpc.aio.insecure_channel(self.grpc_url)
            stub = metrics_pb2_grpc.MetricsServiceStub(self._channel)
            self._call = stub.StreamMetrics()
            logger.info(f"Metrics stream opened for job {job_id}")

    async def send(self, step: int, loss: float, accuracy: float, timestamp_ms: int) -> None:
        if not self._call or not self._job_id:
            return
        try:
            await self._call.write(
                metrics_pb2.MetricPoint(
                    job_id=self._job_id,
                    step=step,
                    loss=loss,
                    accuracy=accuracy,
                    timestamp_ms=timestamp_ms,
                    worker_id=self._worker_id or "",
                )
            )
        except Exception as e:
            logger.warning(f"Failed to send metrics: {e}")

    async def close(self) -> None:
        async with self._lock:
            if not self._call:
                return
            try:
                await self._call.done_writing()
                await self._call
            except Exception as e:
                logger.debug(f"Metrics stream close error: {e}")
            self._call = None
            if self._channel:
                await self._channel.close()
                self._channel = None
