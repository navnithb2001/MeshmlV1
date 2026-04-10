"""In-memory worker registry for Task Orchestrator."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

from app.services.worker_discovery import WorkerCapabilities, WorkerStatus

logger = logging.getLogger(__name__)


@dataclass
class RegistryWorker:
    worker_id: str
    hostname: str
    ip_address: str
    port: int
    capabilities: WorkerCapabilities
    group_id: Optional[str] = None
    version: str = "1.0.0"
    tags: Dict[str, str] = field(default_factory=dict)
    status: WorkerStatus = WorkerStatus.IDLE
    assigned_job_id: Optional[str] = None
    assigned_shard_ids: List[int] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WorkerRegistry:
    """Simple in-memory worker registry."""

    def __init__(self):
        self._workers: Dict[str, RegistryWorker] = {}
        self._lock = Lock()

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        ip_address: str,
        port: int,
        capabilities,
        group_id: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[Dict[str, str]] = None,
    ) -> RegistryWorker:
        with self._lock:
            if isinstance(capabilities, dict):
                caps = WorkerCapabilities.from_dict(capabilities)
            else:
                caps = capabilities

            worker = RegistryWorker(
                worker_id=worker_id,
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                capabilities=caps,
                group_id=group_id,
                version=version,
                tags=tags or {},
            )
            self._workers[worker_id] = worker
            logger.info(f"Worker registered: {worker_id}")
            return worker

    def remove_worker(self, worker_id: str) -> bool:
        with self._lock:
            removed = self._workers.pop(worker_id, None) is not None
            if removed:
                logger.info(f"Worker removed: {worker_id}")
            return removed

    def list_workers(
        self, group_id: Optional[str] = None, min_gpu_count: int = 0
    ) -> List[RegistryWorker]:
        with self._lock:
            workers = list(self._workers.values())

        if group_id:
            workers = [w for w in workers if w.group_id == group_id]

        if min_gpu_count > 0:
            workers = [w for w in workers if w.capabilities.gpu_count >= min_gpu_count]

        return workers

    def assign_job(self, worker_id: str, job_id: str, shard_id: Optional[int] = None) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            worker.assigned_job_id = job_id
            worker.assigned_shard_ids = [shard_id] if shard_id is not None else []
            worker.status = WorkerStatus.BUSY
            return True

    def update_heartbeat(self, worker_id: str, status: Optional[str] = None) -> bool:
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False
            if status:
                try:
                    worker.status = WorkerStatus(status)
                except ValueError:
                    worker.status = WorkerStatus.UNKNOWN
            worker.last_heartbeat = datetime.utcnow().isoformat()
            return True

    def get_worker(self, worker_id: str) -> Optional[RegistryWorker]:
        with self._lock:
            return self._workers.get(worker_id)
