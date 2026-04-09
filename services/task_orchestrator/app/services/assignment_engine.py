"""Assignment engine for Task Orchestrator streaming."""

import asyncio
import logging
from typing import Optional

from app.db import AsyncSessionLocal
from app.models import DataBatch
from app.services.dataset_sharder_client import DatasetSharderClient
from app.services.model_registry_client import ModelRegistryClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class AssignmentEngine:
    def __init__(self, stream_manager, poll_interval: float = 2.0):
        self.stream_manager = stream_manager
        self.poll_interval = poll_interval
        self.model_registry = ModelRegistryClient()
        self.dataset_sharder = DatasetSharderClient()
        self._visibility_logged = False

    async def _fetch_available_batch(self, session: AsyncSession) -> Optional[DataBatch]:
        result = await session.execute(
            select(DataBatch)
            .where(DataBatch.status == "AVAILABLE")
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _assign_batch(self, batch: DataBatch, worker_id: str, session: AsyncSession) -> None:
        batch.status = "ASSIGNED"
        batch.assigned_worker_id = worker_id
        await session.commit()

    async def _build_assignment_payload(self, batch: DataBatch) -> dict:
        model_id_int = int(batch.model_id)
        model_artifact = await self.model_registry.get_model_artifact(model_id_int)
        model_url = model_artifact.download_url if model_artifact.found else ""

        batch_url_resp = await self.dataset_sharder.get_batch_download_url(batch.id)
        data_url = batch_url_resp.get("download_url", "")

        total_batches = 0
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(func.count()).select_from(DataBatch).where(DataBatch.job_id == batch.job_id)
            )
            total_batches = int(result.scalar() or 0)

        return {
            "job_id": batch.job_id,
            "batch_id": batch.id,
            "model_id": batch.model_id,
            "dataset_id": None,
            "model_url": model_url,
            "data_url": data_url,
            "total_batches": total_batches,
        }

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                worker_id = self.stream_manager.get_idle_worker()
                if not worker_id:
                    await asyncio.sleep(self.poll_interval)
                    continue

                async with AsyncSessionLocal() as session:
                    if not self._visibility_logged:
                        total_result = await session.execute(
                            select(func.count()).select_from(DataBatch)
                        )
                        available_result = await session.execute(
                            select(func.count())
                            .select_from(DataBatch)
                            .where(DataBatch.status == "AVAILABLE")
                        )
                        total_rows = int(total_result.scalar() or 0)
                        available_rows = int(available_result.scalar() or 0)
                        logger.info(
                            "Pre-assignment data_batches visibility check: total=%s available=%s",
                            total_rows,
                            available_rows,
                        )
                        if total_rows == 0:
                            logger.warning(
                                "No rows found in data_batches yet; waiting for Dataset Sharder persistence"
                            )
                        self._visibility_logged = True

                    batch = await self._fetch_available_batch(session)
                    if not batch:
                        await asyncio.sleep(self.poll_interval)
                        continue

                    await self._assign_batch(batch, worker_id, session)
                    payload = await self._build_assignment_payload(batch)
                    self.stream_manager.assign_batch(worker_id, batch.id)
                    await self.stream_manager.push_assignment(worker_id, payload)
            except Exception as e:
                logger.exception(
                    "Assignment engine error (model_registry=%s, dataset_sharder=%s): %s",
                    self.model_registry.grpc_url,
                    self.dataset_sharder.grpc_url,
                    e,
                )
                await asyncio.sleep(self.poll_interval)
