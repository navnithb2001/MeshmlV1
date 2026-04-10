"""
Main MeshML Worker Implementation

Coordinates training, communication, and checkpoint management.
"""

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil
from meshml_worker.config import WorkerConfig

logger = logging.getLogger(__name__)


class BlobCache:
    """Simple file hash cache used to avoid unnecessary re-downloads."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path

    async def _read_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except Exception:
            return {}

    async def _write_cache(self, cache: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2))

    async def hash_file(self, path: Path) -> str:
        def _compute() -> str:
            sha = hashlib.sha256()
            with open(path, "rb") as file_obj:
                for chunk in iter(lambda: file_obj.read(8192), b""):
                    sha.update(chunk)
            return sha.hexdigest()

        return await asyncio.to_thread(_compute)

    async def should_use_cached(
        self, cache_key: str, file_path: Path, expected_sha256: Optional[str]
    ) -> bool:
        if not expected_sha256 or not file_path.exists():
            return False

        cache = await self._read_cache()
        cache_entry = cache.get(cache_key, {})
        if cache_entry.get("sha256") != expected_sha256:
            return False

        return (await self.hash_file(file_path)) == expected_sha256

    async def record_download(self, cache_key: str, source_url: str, file_path: Path) -> str:
        file_hash = await self.hash_file(file_path)
        cache = await self._read_cache()
        cache[cache_key] = {"url": source_url, "sha256": file_hash}
        await self._write_cache(cache)
        return file_hash


class ResourceMonitor:
    """Monitors local host resources and toggles pause event for training."""

    def __init__(
        self,
        pause_event: asyncio.Event,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
    ):
        self.pause_event = pause_event
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

    async def check_once(self) -> Tuple[float, float]:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        if cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold:
            if not self.pause_event.is_set():
                logger.warning("High system usage detected; pausing training")
            self.pause_event.set()
        else:
            if self.pause_event.is_set():
                logger.info("Resources available; resuming training")
            self.pause_event.clear()
        return cpu_usage, memory_usage

    async def run(self, is_running: Callable[[], bool]) -> None:
        while is_running():
            await self.check_once()
            await asyncio.sleep(10)


class MeshMLWorker:
    """MeshML Worker for Federated Learning

    Handles:
    - Communication with Parameter Server
    - Local training on data shards
    - Gradient computation and upload
    - Checkpoint management
    - Error recovery
    """

    def __init__(self, config: WorkerConfig):
        """Initialize worker

        Args:
            config: Worker configuration
        """
        self.config = config
        self.worker_id = config.worker.id
        self.running = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._pause_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_signal_count = 0

        # Setup logging
        self._setup_logging()

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info(f"Worker initialized: {self.worker_id}")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(level=self.config.logging.level, format=self.config.logging.format)

        if self.config.logging.file:
            file_handler = logging.FileHandler(self.config.logging.file)
            file_handler.setFormatter(logging.Formatter(self.config.logging.format))
            logging.getLogger().addHandler(file_handler)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum: int, frame: Any) -> None:
            self._shutdown_signal_count += 1
            if self._shutdown_signal_count > 1:
                logger.warning(
                    "Received signal %s again during shutdown; forcing immediate exit.", signum
                )
                os._exit(130)
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.running = False
            if self._loop and self._shutdown_event:
                self._loop.call_soon_threadsafe(self._shutdown_event.set)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self, user_id: str, preferred_job_ids: Optional[list] = None) -> None:
        """Run worker with full Task Orchestrator integration

        This is the production-ready entry point that:
        1. Registers with Task Orchestrator
        2. Requests task assignments
        3. Downloads model from Model Registry
        4. Downloads data from Dataset Sharder
        5. Trains and reports progress

        Args:
            user_id: User ID for authentication
            preferred_job_ids: Optional list of preferred job IDs
        """
        from meshml_worker.communication.metrics_client import MetricsClient
        from meshml_worker.communication.parameter_server_client import ParameterServerClient
        from meshml_worker.communication.task_orchestrator_client import TaskOrchestratorClient
        from meshml_worker.training.trainer import Trainer
        from meshml_worker.utils.device import get_device

        self.running = True
        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        logger.info("=" * 60)
        logger.info("Starting MeshML Worker with Task Orchestrator Integration")
        logger.info("=" * 60)
        logger.info(f"Worker ID: {self.worker_id}")
        logger.info(f"User ID: {user_id}")

        try:
            # Initialize Task Orchestrator client
            orchestrator = TaskOrchestratorClient(
                grpc_url=self.config.task_orchestrator.grpc_url,
                user_id=user_id,
                worker_id=self.worker_id,
                max_retries=self.config.task_orchestrator.max_retries,
                retry_delay=self.config.task_orchestrator.retry_delay,
            )

            # Step 1: Register with Task Orchestrator
            logger.info("\n[1/5] Registering with Task Orchestrator...")
            registration_result = await orchestrator.register()
            logger.info(f"✓ Registration successful!")
            logger.info(
                f"  Capabilities: CPU={registration_result.get('cpu_cores', 0)} cores, "
                f"RAM={registration_result.get('ram_gb', 0):.1f} GB, "
                f"GPU={registration_result.get('has_gpu', False)}"
            )

            cache = BlobCache(self.config.storage.models_dir / ".model_cache.json")

            async def _download(
                url: str, dest_path: Path, cache_key: str, expected_sha: Optional[str]
            ) -> None:
                import httpx

                if await cache.should_use_cached(cache_key, dest_path, expected_sha):
                    logger.info(f"Using cached model: {dest_path}")
                    return

                async with httpx.AsyncClient(timeout=60) as http_client:
                    resp = await http_client.get(url)
                    resp.raise_for_status()
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    dest_path.write_bytes(resp.content)
                await cache.record_download(cache_key, url, dest_path)

            disable_throttle = os.getenv("MESHML_DISABLE_RESOURCE_THROTTLE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            auto_exit_on_job_complete = (
                os.getenv("MESHML_EXIT_ON_JOB_COMPLETE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            cpu_threshold = float(os.getenv("MESHML_CPU_PAUSE_THRESHOLD", "80"))
            ram_threshold = float(os.getenv("MESHML_RAM_PAUSE_THRESHOLD", "80"))

            monitor_task = None
            if disable_throttle:
                logger.info("Resource throttling disabled for this run.")
            else:
                monitor = ResourceMonitor(
                    self._pause_event,
                    cpu_threshold=cpu_threshold,
                    memory_threshold=ram_threshold,
                )
                monitor_task = asyncio.create_task(
                    monitor.run(lambda: bool(self.running and self._pause_event))
                )
            job_progress: Dict[str, Dict[str, int]] = {}

            async def _handle_assignment(assignment, hyperparameters):
                try:
                    job_id = assignment.job_id
                    model_id = hyperparameters.get("model_id") or "unknown"
                    model_sha = hyperparameters.get("model_sha256")
                    total_batches = int(hyperparameters.get("total_batches") or 0)
                    logger.info(
                        "Received assignment: job_id=%s batch_id=%s model_id=%s",
                        job_id,
                        assignment.batch_id,
                        model_id,
                    )

                    model_path = self.config.storage.models_dir / f"{model_id}.py"
                    if assignment.model_url:
                        logger.info("Downloading model artifact for job %s", job_id)
                        await _download(
                            assignment.model_url, model_path, f"model:{model_id}", model_sha
                        )

                    data_dir = self.config.storage.data_dir / f"{assignment.batch_id}"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    data_path = data_dir / "batch.data"
                    if assignment.data_url:
                        logger.info("Downloading data batch %s for job %s", assignment.batch_id, job_id)
                        await _download(
                            assignment.data_url, data_path, f"batch:{assignment.batch_id}", None
                        )

                    # Detect device
                    device = get_device(self.config.training.device)

                    # Initialize Parameter Server client
                    parameter_client = ParameterServerClient(self.config.parameter_server.grpc_url)
                    if not parameter_client.connect():
                        raise RuntimeError("Failed to connect to Parameter Server")

                    trainer = Trainer(
                        config=self.config,
                        grpc_client=parameter_client,
                        device=device,
                        orchestrator_client=orchestrator,
                        metrics_client=MetricsClient(self.config.metrics_service.grpc_url),
                        job_id=job_id,
                        model_path=model_path,
                        data_paths=[data_dir],
                        pause_event=self._pause_event,
                    )

                    await trainer.train(
                        model_id=str(model_id),
                        job_id=job_id,
                        batch_ids=[assignment.batch_id],
                        epochs=1,
                    )

                    if total_batches > 0:
                        progress = job_progress.setdefault(
                            job_id, {"completed": 0, "total": total_batches}
                        )
                        progress["completed"] += 1
                        if progress["completed"] >= progress["total"]:
                            logger.info(f"Job {job_id} completed all batches")
                            if auto_exit_on_job_complete:
                                logger.info(
                                    "Auto-exit enabled; closing stream after job completion."
                                )
                                if self._shutdown_event:
                                    self._shutdown_event.set()

                    return {"success": True}
                except Exception as e:
                    logger.exception(
                        "Assignment execution failed: job_id=%s batch_id=%s error=%s",
                        getattr(assignment, "job_id", ""),
                        getattr(assignment, "batch_id", ""),
                        e,
                    )
                    return {"success": False, "error_message": str(e)}

            logger.info("\n[2/5] Waiting for streamed assignments...")
            while self.running and self._shutdown_event and not self._shutdown_event.is_set():
                try:
                    stream_coro = orchestrator.run_assignment_stream(
                        _handle_assignment,
                        self._shutdown_event,
                        preferred_job_ids=preferred_job_ids,
                    )
                except TypeError:
                    # Backward-compatible path for test doubles/older clients
                    # that do not accept preferred_job_ids.
                    stream_coro = orchestrator.run_assignment_stream(
                        _handle_assignment,
                        self._shutdown_event,
                    )

                stream_task = asyncio.create_task(stream_coro)
                shutdown_wait_task = asyncio.create_task(self._shutdown_event.wait())
                done, _ = await asyncio.wait(
                    {stream_task, shutdown_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if shutdown_wait_task in done and not stream_task.done():
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                else:
                    shutdown_wait_task.cancel()
                    try:
                        await shutdown_wait_task
                    except asyncio.CancelledError:
                        pass

                if self._shutdown_event.is_set():
                    break

                if stream_task.done() and stream_task.exception():
                    logger.warning(
                        "Assignment stream ended with error: %s", stream_task.exception()
                    )
                logger.warning("Assignment stream ended unexpectedly; reconnecting in 2s...")
                await asyncio.sleep(2)

            if monitor_task:
                monitor_task.cancel()
            if auto_exit_on_job_complete and self._shutdown_event and self._shutdown_event.is_set():
                logger.info("Training completed successfully!")
                logger.info("=" * 60)
            elif self.running:
                logger.info("Worker stopped.")
            else:
                logger.info("Worker shutdown complete.")

        except KeyboardInterrupt:
            logger.info("\n\nTraining interrupted by user")
            raise
        except asyncio.CancelledError:
            logger.info("Worker stream cancelled, shutting down.")
        except Exception as e:
            logger.error(f"\n\nTraining failed: {e}", exc_info=True)
            raise
        finally:
            self.running = False
            # Cleanup
            if "orchestrator" in locals():
                await orchestrator.close()

    def validate_setup(self) -> Dict[str, Any]:
        """Validate worker setup

        Returns:
            Validation results
        """
        results: Dict[str, Any] = {"valid": True, "checks": {}}

        # Check PyTorch installation
        try:
            import torch

            results["checks"]["pytorch"] = {"installed": True, "version": torch.__version__}
        except ImportError:
            results["checks"]["pytorch"] = {"installed": False, "error": "PyTorch not installed"}
            results["valid"] = False

        # Check storage directories
        try:
            self.config.storage.create_directories()
            results["checks"]["storage"] = {"created": True}
        except Exception as e:
            results["checks"]["storage"] = {"created": False, "error": str(e)}
            results["valid"] = False

        # Check Parameter Server connectivity
        try:
            import requests

            response = requests.get(
                f"{self.config.parameter_server.url}/health",
                timeout=self.config.parameter_server.timeout,
            )
            results["checks"]["parameter_server"] = {"reachable": response.status_code == 200}
        except Exception as e:
            results["checks"]["parameter_server"] = {"reachable": False, "error": str(e)}
            # Don't mark as invalid - server might not be running yet

        return results


if __name__ == "__main__":
    """Entry point for running worker directly."""
    import asyncio
    import os
    import sys

    from meshml_worker.config import ParameterServerConfig, WorkerConfig, WorkerIdentityConfig

    # Create configuration from environment variables
    config = WorkerConfig(
        worker=WorkerIdentityConfig(
            id=os.getenv("WORKER_ID", "python-worker-1"),
            name=os.getenv("WORKER_NAME", "MeshML Python Worker"),
        ),
        parameter_server=ParameterServerConfig(
            url=os.getenv("PARAMETER_SERVER_URL", "http://parameter-server:8003"),
            grpc_url=os.getenv("ORCHESTRATOR_URL", "task-orchestrator:50051"),
        ),
    )

    # Create worker
    worker = MeshMLWorker(config)
    logger.info(f"Worker initialized: {config.worker.id}")

    # Validate setup
    validation = worker.validate_setup()
    logger.info(f"Setup validation: {validation}")

    # Keep worker running
    try:
        logger.info(f"Worker {config.worker.id} is ready and waiting for tasks...")
        # Keep the process alive indefinitely
        while True:
            asyncio.run(asyncio.sleep(3600))  # Sleep for 1 hour at a time
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed: {e}", exc_info=True)
        sys.exit(1)
