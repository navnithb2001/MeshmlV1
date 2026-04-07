"""Persistence loop for checkpointing weights to Model Registry."""

import asyncio
import io
import logging
from typing import Iterable, Optional

import torch
from app.services.model_registry_client import ModelRegistryClient
from app.services.parameter_storage import ParameterStorageService

logger = logging.getLogger(__name__)


class PersistenceLoop:
    def __init__(
        self,
        storage: ParameterStorageService,
        model_registry: ModelRegistryClient,
        checkpoint_interval: int = 50,
        final_version: int = 500,
        poll_interval: float = 5.0,
    ):
        self.storage = storage
        self.model_registry = model_registry
        self.checkpoint_interval = checkpoint_interval
        self.final_version = final_version
        self.poll_interval = poll_interval

    def _iter_job_ids(self) -> Iterable[str]:
        if not self.storage.enable_redis or not self.storage.redis_client:
            return []
        for key in self.storage.redis_client.scan_iter("params:*:current_version"):
            key_str = key.decode() if isinstance(key, bytes) else str(key)
            parts = key_str.split(":")
            if len(parts) >= 3:
                yield parts[1]

    def _get_current_version(self, job_id: str) -> int:
        key = f"params:{job_id}:current_version"
        raw = self.storage.redis_client.get(key)
        if raw is None:
            return 0
        try:
            return int(raw)
        except Exception:
            return 0

    def _get_last_checkpoint(self, job_id: str) -> int:
        key = f"checkpoint:{job_id}:last_saved"
        raw = self.storage.redis_client.get(key)
        if raw is None:
            return 0
        try:
            return int(raw)
        except Exception:
            return 0

    def _set_last_checkpoint(self, job_id: str, version_id: int) -> None:
        key = f"checkpoint:{job_id}:last_saved"
        self.storage.redis_client.set(key, str(version_id))

    def _final_saved(self, job_id: str) -> bool:
        key = f"checkpoint:{job_id}:final_saved"
        raw = self.storage.redis_client.get(key)
        return raw is not None

    def _set_final_saved(self, job_id: str) -> None:
        key = f"checkpoint:{job_id}:final_saved"
        self.storage.redis_client.set(key, "1")

    def _load_weights(self, job_id: str, version_id: int) -> Optional[dict]:
        data = self.storage.redis_client.get(f"params:{job_id}:v{version_id}")
        if not data:
            return None
        buffer = io.BytesIO(data)
        return torch.load(buffer)

    def _serialize_weights(self, weights: dict) -> bytes:
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        buffer.seek(0)
        return buffer.read()

    def _get_model_id_for_job(self, job_id: str) -> Optional[int]:
        key = f"job:{job_id}:model_id"
        raw = self.storage.redis_client.get(key)
        if raw is None:
            return None
        try:
            return int(raw)
        except Exception:
            return None

    async def _process_job(self, job_id: str) -> None:
        current_version = self._get_current_version(job_id)
        if current_version <= 0:
            return

        model_id = self._get_model_id_for_job(job_id)
        if model_id is None:
            return

        last_checkpoint = self._get_last_checkpoint(job_id)
        if current_version - last_checkpoint >= self.checkpoint_interval:
            import asyncio

            weights = await asyncio.to_thread(self._load_weights, job_id, current_version)
            if weights is not None:
                payload = await asyncio.to_thread(self._serialize_weights, weights)
                await self.model_registry.upload_checkpoint(
                    model_id=model_id, checkpoint_type=f"v{current_version}", state_dict=payload
                )
                self._set_last_checkpoint(job_id, current_version)
                logger.info(
                    f"Checkpoint saved for job {job_id} -> model {model_id} v{current_version}"
                )

        if current_version >= self.final_version and not self._final_saved(job_id):
            import asyncio

            weights = await asyncio.to_thread(self._load_weights, job_id, current_version)
            if weights is not None:
                payload = await asyncio.to_thread(self._serialize_weights, weights)
                await self.model_registry.upload_final_model(model_id=model_id, state_dict=payload)
                self._set_final_saved(job_id)
                logger.info(
                    f"Final model saved for job {job_id} -> model {model_id} v{current_version}"
                )

    async def run(self, stop_event: asyncio.Event) -> None:
        if not self.storage.enable_redis or not self.storage.redis_client:
            logger.warning("Persistence loop disabled: Redis not available")
            return

        while not stop_event.is_set():
            try:
                for job_id in self._iter_job_ids():
                    await self._process_job(job_id)
            except Exception as e:
                logger.error(f"Persistence loop error: {e}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                continue
