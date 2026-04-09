"""gRPC server for Task Orchestrator."""

import asyncio
import json
import logging
import os
import uuid
from typing import List, Optional

import grpc
from app.db import AsyncSessionLocal
from app.models import DataBatch
from app.proto import task_orchestrator_pb2, task_orchestrator_pb2_grpc
from app.services.assignment_engine import AssignmentEngine
from app.services.dataset_sharder_client import DatasetSharderClient
from app.services.job_queue import JobMetadata, JobPriority, JobQueue, JobRequirements
from app.services.metrics_client import MetricsClient
from app.services.model_registry_client import ModelRegistryClient
from app.services.task_assignment import TaskAssignmentService
from app.services.worker_discovery import WorkerCapabilities, WorkerDiscoveryService
from app.services.worker_registry import WorkerRegistry
from redis import Redis
from sqlalchemy import func, select, text

logger = logging.getLogger(__name__)


def _bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


class StreamManager:
    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._status: dict[str, str] = {}
        self._assigned_batch: dict[str, str] = {}

    def register(self, worker_id: str, queue: asyncio.Queue) -> None:
        self._queues[worker_id] = queue
        self._status.setdefault(worker_id, "IDLE")

    def unregister(self, worker_id: str) -> None:
        self._queues.pop(worker_id, None)
        self._status.pop(worker_id, None)
        self._assigned_batch.pop(worker_id, None)

    def get_idle_worker(self) -> Optional[str]:
        for worker_id, status in self._status.items():
            if status == "IDLE" and worker_id in self._queues:
                return worker_id
        return None

    def assign_batch(self, worker_id: str, batch_id: str) -> None:
        self._status[worker_id] = "BUSY"
        self._assigned_batch[worker_id] = batch_id

    def clear_assignment(self, worker_id: str) -> None:
        self._status[worker_id] = "IDLE"
        self._assigned_batch.pop(worker_id, None)

    def get_assigned_batch(self, worker_id: str) -> Optional[str]:
        return self._assigned_batch.get(worker_id)

    async def push_assignment(self, worker_id: str, payload: dict) -> None:
        queue = self._queues.get(worker_id)
        if not queue:
            return
        hyper = {
            "model_id": payload.get("model_id"),
            "dataset_id": payload.get("dataset_id"),
            "total_batches": payload.get("total_batches"),
        }
        assignment = task_orchestrator_pb2.TaskAssignment(
            has_task=True,
            job_id=payload.get("job_id", ""),
            batch_id=payload.get("batch_id", ""),
            batch_gcs_path="",
            model_gcs_path="",
            current_epoch=0,
            hyperparameters=json.dumps(hyper).encode("utf-8"),
            message="assigned",
            model_url=payload.get("model_url", ""),
            data_url=payload.get("data_url", ""),
        )
        await queue.put(
            task_orchestrator_pb2.OrchestratorStreamResponse(
                worker_id=worker_id, assignment=assignment
            )
        )


class TaskOrchestratorServicer(task_orchestrator_pb2_grpc.TaskOrchestratorServicer):
    """gRPC servicer implementing Task Orchestrator APIs."""

    def __init__(
        self,
        worker_discovery: WorkerDiscoveryService,
        job_queue: JobQueue,
        task_assignment: TaskAssignmentService,
        worker_registry: WorkerRegistry,
        model_registry: Optional[ModelRegistryClient] = None,
        metrics_client: Optional[MetricsClient] = None,
    ):
        self.worker_discovery = worker_discovery
        self.job_queue = job_queue
        self.task_assignment = task_assignment
        self.worker_registry = worker_registry
        self.model_registry = model_registry
        self.metrics_client = metrics_client
        self.redis = job_queue.redis if hasattr(job_queue, "redis") else None
        self._batch_failures: dict[str, int] = {}
        self.strict_failure = os.getenv("STRICT_FAILURE", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._sharded_datasets = set()
        self._shard_lock = asyncio.Lock()
        self.stream_manager = StreamManager()
        self.assignment_engine = AssignmentEngine(self.stream_manager)
        self._assignment_stop = asyncio.Event()
        self._assignment_task = asyncio.create_task(
            self.assignment_engine.run(self._assignment_stop)
        )

    async def _resolve_worker_groups(self, user_id: str, worker_id: str) -> List[str]:
        groups: List[str] = []
        try:
            async with AsyncSessionLocal() as session:
                if user_id:
                    result = await session.execute(
                        text(
                            """
                            SELECT DISTINCT gm.group_id::text AS group_id
                            FROM group_members gm
                            WHERE gm.user_id::text = :user_id
                            ORDER BY gm.group_id::text
                            """
                        ),
                        {"user_id": user_id},
                    )
                    groups = [str(row.group_id) for row in result if getattr(row, "group_id", None)]

                if not groups and worker_id:
                    result = await session.execute(
                        text(
                            """
                            SELECT DISTINCT gm.group_id::text AS group_id
                            FROM group_members gm
                            WHERE gm.worker_id = :worker_id
                            ORDER BY gm.group_id::text
                            """
                        ),
                        {"worker_id": worker_id},
                    )
                    groups = [str(row.group_id) for row in result if getattr(row, "group_id", None)]
        except Exception as exc:
            logger.warning("Failed to resolve worker groups for user_id=%s: %s", user_id, exc)
        return groups

    async def InitiateTraining(self, request, context):
        try:
            requirements = JobRequirements(
                min_gpu_count=request.requirements.min_gpu_count,
                min_gpu_memory_gb=request.requirements.min_gpu_memory_gb,
                min_cpu_count=request.requirements.min_cpu_count,
                min_ram_gb=request.requirements.min_ram_gb,
                requires_cuda=request.requirements.requires_cuda,
                requires_mps=request.requirements.requires_mps,
                max_execution_time_seconds=request.requirements.max_execution_time_seconds or 3600,
            )

            metadata = JobMetadata(
                job_id=request.job_id,
                group_id=request.group_id,
                model_id=request.model_id,
                dataset_id=request.dataset_id,
                user_id=request.user_id,
                batch_size=request.batch_size or 32,
                num_epochs=request.num_epochs or 10,
                learning_rate=request.learning_rate or 0.001,
                optimizer=request.optimizer or "adam",
                requirements=requirements,
                tags=dict(request.tags),
                description=request.description or "",
            )

            try:
                priority = JobPriority(request.priority)
            except Exception:
                priority = JobPriority.MEDIUM

            self.job_queue.submit_job(metadata, priority)
            # No separate validator service is running in this deployment.
            # Promote submitted jobs into WAITING immediately.
            self.job_queue.mark_validation_complete(
                metadata.job_id,
                model_validation_passed=True,
                dataset_validation_passed=True,
                validation_errors=None,
            )
            logger.info("InitiateTraining accepted job_id=%s", metadata.job_id)
            return task_orchestrator_pb2.JobSubmissionAck(success=True, message="submitted")
        except Exception as e:
            logger.exception("InitiateTraining failed for job_id=%s: %s", request.job_id, e)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RegisterWorker(self, request, context):
        try:
            requested_id = (request.worker_name or "").strip()
            # Honor caller-provided worker identity when present so external
            # tooling can target an existing worker deterministically.
            worker_id = requested_id if requested_id else f"worker-{uuid.uuid4().hex[:8]}"
            groups = await self._resolve_worker_groups(request.user_id, worker_id)
            group_id = groups[0] if groups else (request.user_id or "default")

            gpu_count = len(request.gpus)
            gpu_mem_gb = max([_bytes_to_gb(g.memory_bytes) for g in request.gpus], default=0.0)
            gpu_type = request.gpus[0].name if request.gpus else "none"
            supports_cuda = any(g.cuda_available for g in request.gpus)
            supports_mps = any(g.metal_available for g in request.gpus)

            capabilities = WorkerCapabilities(
                gpu_count=gpu_count,
                gpu_memory_gb=gpu_mem_gb,
                gpu_type=gpu_type,
                cpu_count=request.cpu_cores,
                ram_gb=_bytes_to_gb(request.ram_bytes),
                network_speed_mbps=0.0,
                storage_gb=0.0,
                supports_cuda=supports_cuda,
                supports_mps=supports_mps,
                pytorch_version=request.frameworks.get("pytorch", "unknown"),
                python_version=request.frameworks.get("python", "unknown"),
            )

            self.worker_discovery.register_worker(
                worker_id=worker_id,
                hostname=request.worker_name or worker_id,
                ip_address=request.ip_address or "unknown",
                port=0,
                capabilities=capabilities,
                group_id=group_id,
                version="1.0.0",
                tags={"device_type": request.device_type},
            )

            # Sync with database
            try:
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        text("SELECT 1 FROM workers WHERE worker_id = :worker_id"),
                        {"worker_id": worker_id}
                    )
                    exists = result.scalar()
                    logger.info("Worker DB sync: exists=%s for %s", exists, worker_id)
                    if not exists:
                        logger.info("Inserting new worker %s into DB", worker_id)
                        await session.execute(
                            text("""
                                INSERT INTO workers (id, worker_id, user_email, capabilities, status, last_heartbeat, created_at, updated_at)
                                VALUES (gen_random_uuid(), :worker_id, :user_email, :caps, 'idle', NOW(), NOW(), NOW())
                                ON CONFLICT (worker_id) DO NOTHING
                            """),
                            {
                                "worker_id": worker_id, 
                                "user_email": getattr(request, "user_id", "") or "unknown", 
                                "caps": json.dumps(capabilities.to_dict())
                            }
                        )
                    else:
                        logger.info("Updating existing worker %s in DB", worker_id)
                        await session.execute(
                            text("""
                                UPDATE workers 
                                SET status = 'idle', capabilities = :caps, last_heartbeat = NOW(), updated_at = NOW() 
                                WHERE worker_id = :worker_id
                            """),
                            {"worker_id": worker_id, "caps": json.dumps(capabilities.to_dict())}
                        )
                    await session.commit()
                    logger.info("DB commit successful for worker %s", worker_id)
            except Exception as dberr:
                logger.error("Failed to sync worker registration to DB: %s", dberr, exc_info=True)

            return task_orchestrator_pb2.WorkerRegistration(
                worker_id=worker_id,
                groups=groups or [group_id],
                heartbeat_interval_seconds=self.worker_discovery.config.heartbeat_timeout_seconds,
                message="registered",
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def SendHeartbeat(self, request, context):
        try:
            success = self.worker_registry.update_heartbeat(request.worker_id, request.status)
            
            if success:
                try:
                    async with AsyncSessionLocal() as session:
                        await session.execute(
                            text("UPDATE workers SET status = :status, last_heartbeat = NOW(), updated_at = NOW() WHERE worker_id = :worker_id"),
                            {"status": request.status or "idle", "worker_id": request.worker_id}
                        )
                        await session.commit()
                except Exception as dberr:
                    logger.warning("Failed to sync worker heartbeat to DB: %s", dberr)
            return task_orchestrator_pb2.HeartbeatAck(
                success=success,
                message="ok" if success else "worker not found",
                server_timestamp=int(asyncio.get_event_loop().time()),
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def RequestTask(self, request, context):
        try:
            worker = self.worker_registry.get_worker(request.worker_id)
            if not worker:
                return task_orchestrator_pb2.TaskAssignment(
                    has_task=False, message="worker not registered"
                )

            # Use preferred job if provided
            job_info = None
            for job_id in request.preferred_job_ids:
                job_info = self.job_queue.get_job(job_id)
                if job_info and job_info.status.value == "waiting":
                    break
                job_info = None

            if not job_info:
                requirements = JobRequirements(
                    min_gpu_count=worker.capabilities.gpu_count,
                    min_gpu_memory_gb=worker.capabilities.gpu_memory_gb,
                    min_cpu_count=worker.capabilities.cpu_count,
                    min_ram_gb=worker.capabilities.ram_gb,
                    requires_cuda=worker.capabilities.supports_cuda,
                    requires_mps=worker.capabilities.supports_mps,
                )
                job_info = self.job_queue.get_next_job(requirements)

            if not job_info:
                return task_orchestrator_pb2.TaskAssignment(
                    has_task=False, message="no jobs available"
                )

            # Ensure sharding and assign batches for this worker
            batch_ids, shard_id = await self._ensure_shards_and_assign_batches(
                job_info, worker.worker_id
            )

            # Assign job to worker in registry/queue
            self.worker_discovery.assign_job_to_worker(
                job_id=job_info.job_id,
                worker_id=worker.worker_id,
                shard_ids=[shard_id] if shard_id is not None else [],
            )
            self.worker_registry.assign_job(
                worker_id=worker.worker_id, job_id=job_info.job_id, shard_id=shard_id
            )

            # Build hyperparameters payload (JSON)
            hyper = {
                "model_id": job_info.metadata.model_id,
                "dataset_id": job_info.metadata.dataset_id,
                "batch_ids": batch_ids,
            }

            if self.model_registry and job_info.metadata.model_id:
                try:
                    model_id_int = int(job_info.metadata.model_id)
                    artifact = await self.model_registry.get_model_artifact(model_id_int)
                    if artifact.found:
                        hyper["model_download_url"] = artifact.download_url
                        hyper["model_gcs_path"] = artifact.gcs_path
                        hyper["model_sha256"] = artifact.sha256
                except Exception:
                    pass

            batch_id = str(batch_ids[0]) if batch_ids else ""

            return task_orchestrator_pb2.TaskAssignment(
                has_task=True,
                job_id=job_info.job_id,
                batch_id=batch_id,
                batch_gcs_path="",
                model_gcs_path=hyper.get("model_gcs_path", ""),
                current_epoch=job_info.current_epoch,
                hyperparameters=json.dumps(hyper).encode("utf-8"),
                message="assigned",
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def StreamTasks(self, request_iterator, context):
        try:
            worker_id = None
            outbound = asyncio.Queue()
            logger.info("StreamTasks opened")

            async def _reader():
                nonlocal worker_id
                try:
                    async for request in request_iterator:
                        try:
                            worker_id = request.worker_id or worker_id
                            if not worker_id:
                                continue
                            self.stream_manager.register(worker_id, outbound)

                            if request.HasField("heartbeat"):
                                success = self.worker_registry.update_heartbeat(
                                    worker_id, request.heartbeat.status
                                )
                                
                                if success:
                                    try:
                                        async with AsyncSessionLocal() as session:
                                            result = await session.execute(
                                                text("UPDATE workers SET status = :status, last_heartbeat = NOW(), updated_at = NOW() WHERE worker_id = :worker_id"),
                                                {"status": request.heartbeat.status or "idle", "worker_id": worker_id}
                                            )
                                            logger.info("Stream heartbeat DB update for %s: rows matched=%s", worker_id, result.rowcount)
                                            await session.commit()
                                    except Exception as dberr:
                                        logger.error("Failed to sync stream heartbeat to DB: %s", dberr, exc_info=True)
                                await outbound.put(
                                    task_orchestrator_pb2.OrchestratorStreamResponse(
                                        worker_id=worker_id,
                                        heartbeat_ack=task_orchestrator_pb2.HeartbeatAck(
                                            success=success,
                                            message="ok" if success else "worker not found",
                                            server_timestamp=int(asyncio.get_event_loop().time()),
                                        ),
                                    )
                                )
                            elif request.HasField("task_result"):
                                await self._handle_task_result(request.task_result)
                            elif request.HasField("task_request"):
                                # Prime sharding/data_batches for streaming workers so the
                                # assignment engine has persisted rows to dispatch.
                                await self._ensure_stream_job_sharded(
                                    worker_id=worker_id,
                                    preferred_job_ids=list(request.task_request.preferred_job_ids),
                                )
                        except Exception:
                            logger.exception(
                                "StreamTasks request handling failed for worker %s", worker_id
                            )
                            continue
                except Exception:
                    logger.exception("StreamTasks reader failed for worker %s", worker_id)

            reader_task = asyncio.create_task(_reader())

            while True:
                try:
                    response = await asyncio.wait_for(outbound.get(), timeout=1.0)
                    yield response
                except asyncio.TimeoutError:
                    if reader_task.done():
                        # Keep stream open even if client finished request-side writes.
                        # Assignments can still be pushed on response stream.
                        try:
                            exc = reader_task.exception()
                        except asyncio.CancelledError:
                            exc = None
                        if exc:
                            logger.warning(
                                "StreamTasks reader ended with exception for worker %s: %s",
                                worker_id,
                                exc,
                            )
                        # If stream never received a worker id and reader is done,
                        # there is no active client to keep alive for.
                        if worker_id is None:
                            break
                    continue
                except asyncio.CancelledError:
                    break
            logger.info("StreamTasks context closed for worker %s", worker_id)
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        finally:
            try:
                if "reader_task" in locals():
                    reader_task.cancel()
            except Exception:
                pass
            if worker_id:
                await self._handle_disconnect(worker_id)
            logger.info("StreamTasks closed for worker %s", worker_id)

    async def _ensure_stream_job_sharded(
        self, worker_id: str, preferred_job_ids: Optional[List[str]] = None
    ) -> None:
        """
        Ensure sharding has run for at least one candidate job in stream mode.

        StreamTasks relies on AssignmentEngine reading data_batches; this hook
        reuses RequestTask's sharding path so batches are persisted before
        assignment polling attempts to dispatch work.
        """
        try:
            worker = self.worker_registry.get_worker(worker_id)
            if not worker:
                return

            job_info = None
            for job_id in preferred_job_ids or []:
                candidate = self.job_queue.get_job(job_id)
                if candidate and candidate.status.value == "waiting":
                    job_info = candidate
                    break

            if not job_info:
                requirements = JobRequirements(
                    min_gpu_count=worker.capabilities.gpu_count,
                    min_gpu_memory_gb=worker.capabilities.gpu_memory_gb,
                    min_cpu_count=worker.capabilities.cpu_count,
                    min_ram_gb=worker.capabilities.ram_gb,
                    requires_cuda=worker.capabilities.supports_cuda,
                    requires_mps=worker.capabilities.supports_mps,
                )
                job_info = self.job_queue.get_next_job(requirements)

            if not job_info:
                return

            await self._ensure_shards_and_assign_batches(job_info, worker_id)
        except Exception as exc:
            logger.warning("Stream sharding preflight failed for worker %s: %s", worker_id, exc)

    async def _handle_task_result(self, result):
        async with AsyncSessionLocal() as session:
            batch = await session.get(DataBatch, result.batch_id)
            if not batch:
                return
            job_id = batch.job_id
            model_id = batch.model_id

            if result.success:
                batch.status = "COMPLETED"
                batch.assigned_worker_id = None
                await session.commit()
                self.stream_manager.clear_assignment(result.worker_id)
                await self._update_job_progress(job_id, session)
                await self._maybe_complete_job(job_id, model_id, session)
                return

            if (
                self.strict_failure
                and result.error_message
                and "CRITICAL" in result.error_message.upper()
            ):
                batch.status = "FAILED"
                batch.assigned_worker_id = None
                await session.commit()
                self.stream_manager.clear_assignment(result.worker_id)
                await self._mark_job_failed(job_id, result.error_message, session)
                return

            failure_count = self._increment_batch_failure(job_id, batch.id)
            logger.warning(
                "TaskResult failure job=%s batch=%s worker=%s failure_count=%s error=%s",
                job_id,
                batch.id,
                result.worker_id,
                failure_count,
                (result.error_message or "")[:300],
            )
            if failure_count >= 3:
                batch.status = "FAILED"
                batch.assigned_worker_id = None
                await session.commit()
                self.stream_manager.clear_assignment(result.worker_id)
                await self._mark_job_failed(
                    job_id, f"Batch {batch.id} failed {failure_count} times", session
                )
            else:
                batch.status = "AVAILABLE"
                batch.assigned_worker_id = None
                await session.commit()
                self.stream_manager.clear_assignment(result.worker_id)

    async def _handle_disconnect(self, worker_id: str):
        batch_id = self.stream_manager.get_assigned_batch(worker_id)
        if batch_id:
            try:
                async with AsyncSessionLocal() as session:
                    batch = await session.get(DataBatch, batch_id)
                    if batch and batch.status == "ASSIGNED":
                        batch.status = "AVAILABLE"
                        batch.assigned_worker_id = None
                        await session.commit()
            except Exception as e:
                logger.error("Error unassigning batch %s on disconnect: %s", batch_id, e)
                
        # Mark worker as offline in DB
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    text("UPDATE workers SET status = 'offline', updated_at = NOW() WHERE worker_id = :worker_id"),
                    {"worker_id": worker_id}
                )
                await session.commit()
                logger.info("Worker %s marked as offline in DB", worker_id)
        except Exception as e:
            logger.error("Failed to mark worker %s offline in DB: %s", worker_id, e)
            
        self.stream_manager.unregister(worker_id)

    def _increment_batch_failure(self, job_id: str, batch_id: str) -> int:
        key = f"batch_failures:{job_id}:{batch_id}"
        if self.redis:
            try:
                return int(self.redis.incr(key))
            except Exception:
                pass
        current = self._batch_failures.get(key, 0) + 1
        self._batch_failures[key] = current
        return current

    async def _maybe_complete_job(self, job_id: str, model_id: str, session) -> None:
        total_result = await session.execute(
            select(func.count()).select_from(DataBatch).where(DataBatch.job_id == job_id)
        )
        total_batches = int(total_result.scalar() or 0)
        if total_batches == 0:
            return
        completed_result = await session.execute(
            select(func.count())
            .select_from(DataBatch)
            .where(DataBatch.job_id == job_id)
            .where(DataBatch.status == "COMPLETED")
        )
        completed_batches = int(completed_result.scalar() or 0)
        if completed_batches < total_batches:
            return

        artifact_uri = None
        if self.model_registry and model_id:
            try:
                response = await self.model_registry.get_final_model_download_url(int(model_id))
                if response and response.storage_path:
                    artifact_uri = response.storage_path
            except Exception:
                artifact_uri = None

        if artifact_uri:
            await session.execute(
                text(
                    "UPDATE jobs "
                    "SET status = 'completed', completed_at = NOW(), "
                    "config = ("
                    "  jsonb_set("
                    "    COALESCE(config::jsonb, '{}'::jsonb), "
                    "  '{model_artifact_uri}', "
                    "  to_jsonb(CAST(:uri AS text)), "
                    "  true"
                    "  )::json"
                    ") "
                    "WHERE id::text = :job_id"
                ),
                {
                    "job_id": job_id,
                    "uri": artifact_uri,
                },
            )
        else:
            await session.execute(
                text(
                    "UPDATE jobs "
                    "SET status = 'completed', completed_at = NOW() "
                    "WHERE id::text = :job_id"
                ),
                {"job_id": job_id},
            )
        await session.commit()

        if self.metrics_client:
            await self.metrics_client.send_job_finished(job_id)

    async def _mark_job_failed(self, job_id: str, message: str, session) -> None:
        await session.execute(
            text(
                "UPDATE jobs "
                "SET status = 'failed', error_message = :message, completed_at = NOW() "
                "WHERE id::text = :job_id"
            ),
            {
                "job_id": job_id,
                "message": message[:1000],
            },
        )
        await session.commit()

    async def _update_job_progress(self, job_id: str, session) -> None:
        total_result = await session.execute(
            select(func.count()).select_from(DataBatch).where(DataBatch.job_id == job_id)
        )
        total_batches = int(total_result.scalar() or 0)
        completed_result = await session.execute(
            select(func.count())
            .select_from(DataBatch)
            .where(DataBatch.job_id == job_id)
            .where(DataBatch.status == "COMPLETED")
        )
        completed_batches = int(completed_result.scalar() or 0)
        await session.execute(
            text(
                "UPDATE jobs "
                "SET progress = ("
                "  jsonb_set("
                "    jsonb_set(COALESCE(progress::jsonb, '{}'::jsonb), '{current_batch}', to_jsonb(CAST(:completed AS integer)), true),"
                "    '{total_batches}', to_jsonb(CAST(:total AS integer)), true"
                "  )::json"
                ") "
                "WHERE id::text = :job_id"
            ),
            {
                "completed": completed_batches,
                "total": total_batches,
                "job_id": job_id,
            },
        )
        await session.commit()

    async def _ensure_shards_and_assign_batches(
        self, job_info, worker_id: str
    ) -> tuple[List[int], Optional[int]]:
        dataset_id = job_info.metadata.dataset_id

        async with self._shard_lock:
            if dataset_id not in self._sharded_datasets:
                bucket = os.getenv("DATASET_GCS_BUCKET", "meshml-datasets")
                template = os.getenv("DATASET_PATH_TEMPLATE", "gs://{bucket}/{dataset_id}/")
                dataset_path = template.format(bucket=bucket, dataset_id=dataset_id)

                dataset_format = None
                if isinstance(job_info.metadata.tags, dict):
                    dataset_format = job_info.metadata.tags.get("dataset_format")

                num_shards = int(os.getenv("DATASET_DEFAULT_SHARDS", "10"))
                if isinstance(job_info.metadata.tags, dict):
                    num_shards = int(job_info.metadata.tags.get("num_shards", num_shards))

                shard_strategy = "stratified"
                if isinstance(job_info.metadata.tags, dict):
                    shard_strategy = job_info.metadata.tags.get("shard_strategy", shard_strategy)

                batch_size = job_info.metadata.batch_size

                client = DatasetSharderClient()
                await client.shard_dataset(
                    dataset_id=dataset_id,
                    job_id=job_info.job_id,
                    model_id=job_info.metadata.model_id,
                    dataset_path=dataset_path,
                    format=dataset_format,
                    num_shards=num_shards,
                    strategy=shard_strategy,
                    batch_size=batch_size,
                    seed=42,
                )

                self._sharded_datasets.add(dataset_id)

        client = DatasetSharderClient()
        assignment = await client.assign_batches(
            worker_ids=[worker_id], strategy="shard_per_worker"
        )

        assignments = assignment.get("assignments", {})
        worker_assignment = assignments.get(worker_id, {})
        batch_id_strs = worker_assignment.get("assigned_batches", [])
        shard_id = worker_assignment.get("shard_id")

        # Convert batch IDs like "shard_0_batch_3" -> 3 (int)
        batch_ids: List[int] = []
        for bid in batch_id_strs:
            if isinstance(bid, str) and "_batch_" in bid:
                try:
                    batch_ids.append(int(bid.split("_batch_")[-1]))
                    continue
                except ValueError:
                    pass
            try:
                batch_ids.append(int(bid))
            except Exception:
                batch_ids.append(0)

        return batch_ids, shard_id


def create_grpc_services() -> TaskOrchestratorServicer:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_client = Redis(host=redis_host, port=redis_port, decode_responses=False)

    worker_registry = WorkerRegistry()
    job_queue = JobQueue(redis_client)
    worker_discovery = WorkerDiscoveryService(worker_registry, job_queue)
    task_assignment = TaskAssignmentService(worker_discovery, job_queue, worker_registry)
    model_registry = ModelRegistryClient()
    metrics_client = MetricsClient()

    return TaskOrchestratorServicer(
        worker_discovery,
        job_queue,
        task_assignment,
        worker_registry,
        model_registry=model_registry,
        metrics_client=metrics_client,
    )


async def start_grpc_server(app, host: str, port: int) -> None:
    server = grpc.aio.server()
    servicer = create_grpc_services()
    task_orchestrator_pb2_grpc.add_TaskOrchestratorServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    app.state.grpc_server = server
    logger.info(f"gRPC server started on {host}:{port}")
