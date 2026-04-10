"""
Task Orchestrator gRPC Client

Handles gRPC communication with the Task Orchestrator service for:
- Worker registration
- Task assignment requests
- Heartbeat monitoring
- Progress reporting
- Failure handling
"""

import asyncio
import json
import logging
import platform
from typing import Any, Dict, List, Optional

import grpc
import psutil
import torch
from meshml_worker.proto import task_orchestrator_pb2, task_orchestrator_pb2_grpc

logger = logging.getLogger(__name__)


class TaskOrchestratorClient:
    """Client for Task Orchestrator gRPC communication"""

    def __init__(
        self,
        grpc_url: str,
        user_id: str,
        worker_id: Optional[str] = None,
        worker_name: str = "MeshML Python Worker",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize Task Orchestrator client

        Args:
            grpc_url: gRPC server address (host:port)
            user_id: User ID who owns this worker
            worker_id: Optional pre-configured worker ID (server may override)
            worker_name: Human-readable worker name
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.grpc_url = grpc_url
        self.user_id = user_id
        if worker_name == "MeshML Python Worker" and worker_id:
            self.worker_name = worker_id
        else:
            self.worker_name = worker_name or "MeshML Python Worker"
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[task_orchestrator_pb2_grpc.TaskOrchestratorStub] = None
        self.worker_id: Optional[str] = worker_id
        self.heartbeat_interval: int = 30  # seconds
        self._heartbeat_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized Task Orchestrator client for {grpc_url}")

    async def connect(self) -> None:
        """Establish gRPC connection to Task Orchestrator"""
        try:
            logger.info(f"Connecting to Task Orchestrator at {self.grpc_url}")

            # Create async gRPC channel
            self.channel = grpc.aio.insecure_channel(
                self.grpc_url,
                options=[
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
                    ("grpc.keepalive_time_ms", 30000),
                    ("grpc.keepalive_timeout_ms", 10000),
                ],
            )

            # Create stub
            self.stub = task_orchestrator_pb2_grpc.TaskOrchestratorStub(self.channel)

            logger.info("Successfully connected to Task Orchestrator")

        except Exception as e:
            logger.error(f"Failed to connect to Task Orchestrator: {e}")
            raise RuntimeError(f"gRPC connection failed: {e}")

    async def close(self) -> None:
        """Close gRPC connection"""
        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close channel
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("Task Orchestrator connection closed")

    def _get_worker_capabilities(self) -> task_orchestrator_pb2.WorkerCapabilities:
        """
        Gather worker capabilities for registration

        Returns:
            WorkerCapabilities protobuf message
        """
        # Get system info
        cpu_count = psutil.cpu_count(logical=False) or 1
        ram_bytes = psutil.virtual_memory().total

        # Get GPU info
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu = task_orchestrator_pb2.GPU(
                    name=props.name,
                    memory_bytes=props.total_memory,
                    driver_version=torch.version.cuda or "unknown",
                    cuda_available=True,
                    metal_available=False,
                )
                gpus.append(gpu)
        elif torch.backends.mps.is_available():
            # Apple Metal
            gpu = task_orchestrator_pb2.GPU(
                name="Apple Metal",
                memory_bytes=ram_bytes,  # Unified memory
                driver_version="metal",
                cuda_available=False,
                metal_available=True,
            )
            gpus.append(gpu)

        # Get framework versions
        frameworks = {
            "pytorch": torch.__version__,
            "python": platform.python_version(),
        }

        # Try to get IP address
        try:
            import socket

            ip_address = socket.gethostbyname(socket.gethostname())
        except:
            ip_address = "unknown"

        # Create capabilities message
        capabilities = task_orchestrator_pb2.WorkerCapabilities(
            user_id=self.user_id,
            device_type="python",
            os=platform.system(),
            arch=platform.machine(),
            cpu_cores=cpu_count,
            ram_bytes=ram_bytes,
            gpus=gpus,
            frameworks=frameworks,
            ip_address=ip_address,
            worker_name=self.worker_name,
        )

        return capabilities

    async def register(self) -> Dict[str, Any]:
        """
        Register worker with Task Orchestrator

        Returns:
            Registration response with worker_id, groups, heartbeat_interval

        Raises:
            RuntimeError: If registration fails
        """
        if not self.stub:
            await self.connect()

        logger.info("Registering worker with Task Orchestrator")

        try:
            # Get capabilities
            capabilities = self._get_worker_capabilities()

            # Make registration RPC call
            response: task_orchestrator_pb2.WorkerRegistration = await self.stub.RegisterWorker(
                capabilities
            )

            # Store worker ID and heartbeat interval
            self.worker_id = response.worker_id
            self.heartbeat_interval = response.heartbeat_interval_seconds

            logger.info(
                f"Worker registered successfully: {self.worker_id} "
                f"(heartbeat: {self.heartbeat_interval}s)"
            )

            # Start heartbeat
            await self.start_heartbeat()

            return {
                "worker_id": response.worker_id,
                "groups": list(response.groups),
                "heartbeat_interval": response.heartbeat_interval_seconds,
                "message": response.message,
                "cpu_cores": capabilities.cpu_cores,
                "ram_gb": capabilities.ram_bytes / (1024**3),
                "has_gpu": len(capabilities.gpus) > 0,
            }

        except grpc.RpcError as e:
            logger.error(f"Worker registration failed: {e.code()} - {e.details()}")
            raise RuntimeError(f"Registration failed: {e.details()}")

    async def send_heartbeat(self, status: str = "idle", active_tasks: int = 0) -> bool:
        """
        Send heartbeat to Task Orchestrator

        Args:
            status: Worker status (online, busy, idle)
            active_tasks: Number of currently active tasks

        Returns:
            True if heartbeat was acknowledged, False otherwise
        """
        if not self.stub or not self.worker_id:
            logger.warning("Cannot send heartbeat: worker not registered")
            return False

        try:
            # Get current resource usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            ram_usage = psutil.virtual_memory().percent

            # GPU usage (if available)
            gpu_usage = 0.0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                try:
                    # This requires nvidia-ml-py3
                    import pynvml

                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = float(util.gpu)
                    pynvml.nvmlShutdown()
                except:
                    gpu_usage = 0.0

            # Create heartbeat message
            heartbeat = task_orchestrator_pb2.Heartbeat(
                worker_id=self.worker_id,
                status=status,
                active_tasks=active_tasks,
                cpu_usage_percent=cpu_usage,
                ram_usage_percent=ram_usage,
                gpu_usage_percent=gpu_usage,
            )

            # Send heartbeat
            response: task_orchestrator_pb2.HeartbeatAck = await self.stub.SendHeartbeat(heartbeat)

            if response.success:
                logger.debug(f"Heartbeat acknowledged: {response.message}")
                return True
            else:
                logger.warning(f"Heartbeat not acknowledged: {response.message}")
                return False

        except grpc.RpcError as e:
            logger.error(f"Heartbeat failed: {e.code()} - {e.details()}")
            return False

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats"""
        logger.info(f"Starting heartbeat loop (interval: {self.heartbeat_interval}s)")

        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.send_heartbeat()

            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                # Continue sending heartbeats despite errors
                await asyncio.sleep(self.heartbeat_interval)

    async def start_heartbeat(self) -> None:
        """Start the heartbeat background task"""
        if self._heartbeat_task and not self._heartbeat_task.done():
            logger.warning("Heartbeat already running")
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat task started")

    async def request_task(
        self, preferred_job_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request a task assignment from Task Orchestrator

        Args:
            preferred_job_ids: Optional list of preferred job IDs

        Returns:
            Task assignment dict with job_id, batch_id, paths, etc.
            None if no tasks available
        """
        if not self.stub or not self.worker_id:
            raise RuntimeError("Worker not registered")

        logger.info(f"Requesting task assignment for worker {self.worker_id}")

        try:
            # Create request
            request = task_orchestrator_pb2.TaskRequest(
                worker_id=self.worker_id, preferred_job_ids=preferred_job_ids or []
            )

            # Make RPC call
            assignment: task_orchestrator_pb2.TaskAssignment = await self.stub.RequestTask(request)

            if not assignment.has_task:
                logger.info(f"No tasks available: {assignment.message}")
                return None

            # Parse hyperparameters
            hyperparameters = {}
            if assignment.hyperparameters:
                try:
                    hyperparameters = json.loads(assignment.hyperparameters.decode("utf-8"))
                except:
                    logger.warning("Failed to parse hyperparameters")

            model_id = hyperparameters.get("model_id")
            dataset_id = hyperparameters.get("dataset_id")
            batch_ids = hyperparameters.get("batch_ids", [])
            if not isinstance(batch_ids, list):
                batch_ids = []

            task_info = {
                "job_id": assignment.job_id,
                "batch_id": assignment.batch_id,
                "batch_gcs_path": assignment.batch_gcs_path,
                "model_gcs_path": assignment.model_gcs_path,
                "current_epoch": assignment.current_epoch,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "batch_ids": batch_ids,
                "hyperparameters": hyperparameters,
                "message": assignment.message,
            }

            logger.info(
                f"Task assigned: job={task_info['job_id']}, "
                f"batch={task_info['batch_id']}, "
                f"epoch={task_info['current_epoch']}"
            )

            return task_info

        except grpc.RpcError as e:
            logger.error(f"Task request failed: {e.code()} - {e.details()}")
            raise RuntimeError(f"Task request failed: {e.details()}")

    async def request_task_streaming(
        self, preferred_job_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Request a task using bi-directional streaming RPC.
        """
        if not self.stub or not self.worker_id:
            raise RuntimeError("Worker not registered")

        try:
            call = self.stub.StreamTasks()
            await call.write(
                task_orchestrator_pb2.WorkerStreamRequest(
                    worker_id=self.worker_id,
                    task_request=task_orchestrator_pb2.TaskRequest(
                        worker_id=self.worker_id, preferred_job_ids=preferred_job_ids or []
                    ),
                )
            )
            response = await call.read()
            await call.done_writing()

            if not response or not response.HasField("assignment"):
                return None

            assignment = response.assignment
            if not assignment.has_task:
                return None

            hyperparameters = {}
            if assignment.hyperparameters:
                try:
                    hyperparameters = json.loads(assignment.hyperparameters.decode("utf-8"))
                except:
                    logger.warning("Failed to parse hyperparameters")

            model_id = hyperparameters.get("model_id")
            dataset_id = hyperparameters.get("dataset_id")
            batch_ids = hyperparameters.get("batch_ids", [])
            if not isinstance(batch_ids, list):
                batch_ids = []

            return {
                "job_id": assignment.job_id,
                "batch_id": assignment.batch_id,
                "batch_gcs_path": assignment.batch_gcs_path,
                "model_gcs_path": assignment.model_gcs_path,
                "current_epoch": assignment.current_epoch,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "batch_ids": batch_ids,
                "hyperparameters": hyperparameters,
                "message": assignment.message,
            }
        except grpc.RpcError as e:
            logger.error(f"Streaming task request failed: {e.code()} - {e.details()}")
            raise RuntimeError(f"Streaming task request failed: {e.details()}")

    async def run_assignment_stream(
        self,
        handler,
        stop_event: Optional[asyncio.Event] = None,
        preferred_job_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Maintain bidirectional stream and handle assignments.
        """
        if not self.stub or not self.worker_id:
            raise RuntimeError("Worker not registered")

        call = self.stub.StreamTasks()
        write_lock = asyncio.Lock()

        # Prime stream on open so server can keep it alive and preflight sharding.
        async with write_lock:
            await call.write(
                task_orchestrator_pb2.WorkerStreamRequest(
                    worker_id=self.worker_id,
                    task_request=task_orchestrator_pb2.TaskRequest(
                        worker_id=self.worker_id,
                        preferred_job_ids=preferred_job_ids or [],
                    ),
                )
            )

        async def send_heartbeat():
            first_tick = True
            while True:
                if first_tick:
                    first_tick = False
                else:
                    await asyncio.sleep(self.heartbeat_interval)
                if stop_event and stop_event.is_set():
                    break
                try:
                    async with write_lock:
                        await call.write(
                            task_orchestrator_pb2.WorkerStreamRequest(
                                worker_id=self.worker_id,
                                heartbeat=task_orchestrator_pb2.Heartbeat(
                                    worker_id=self.worker_id,
                                    status="idle",
                                    active_tasks=0,
                                    cpu_usage_percent=0.0,
                                    ram_usage_percent=0.0,
                                    gpu_usage_percent=0.0,
                                ),
                            )
                        )
                        # Re-issue task request so newly submitted jobs can be
                        # discovered even if the stream was opened earlier.
                        await call.write(
                            task_orchestrator_pb2.WorkerStreamRequest(
                                worker_id=self.worker_id,
                                task_request=task_orchestrator_pb2.TaskRequest(
                                    worker_id=self.worker_id,
                                    preferred_job_ids=preferred_job_ids or [],
                                ),
                            )
                        )
                except Exception:
                    # Stream likely closed; stop heartbeat loop.
                    break

        heartbeat_task = asyncio.create_task(send_heartbeat())
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                try:
                    response = await call.read()
                except asyncio.CancelledError:
                    # Stream cancelled by shutdown or transport close.
                    break
                except grpc.aio.AioRpcError as e:
                    logger.warning("Assignment stream closed: %s - %s", e.code(), e.details())
                    break

                if response is grpc.aio.EOF:
                    logger.info("Assignment stream reached EOF")
                    break
                if not response:
                    continue

                if response.HasField("assignment"):
                    assignment = response.assignment
                    if not assignment.has_task:
                        continue

                    hyperparameters = {}
                    if assignment.hyperparameters:
                        try:
                            hyperparameters = json.loads(assignment.hyperparameters.decode("utf-8"))
                        except:
                            pass

                    result = await handler(assignment, hyperparameters)
                    success = bool(result.get("success", False))
                    error_message = result.get("error_message", "")
                    async with write_lock:
                        await call.write(
                            task_orchestrator_pb2.WorkerStreamRequest(
                                worker_id=self.worker_id,
                                task_result=task_orchestrator_pb2.TaskResult(
                                    worker_id=self.worker_id,
                                    job_id=assignment.job_id,
                                    batch_id=assignment.batch_id,
                                    success=success,
                                    error_message=error_message,
                                ),
                            )
                        )
        finally:
            if stop_event and stop_event.is_set():
                try:
                    async with write_lock:
                        await call.write(
                            task_orchestrator_pb2.WorkerStreamRequest(
                                worker_id=self.worker_id,
                                task_result=task_orchestrator_pb2.TaskResult(
                                    worker_id=self.worker_id,
                                    job_id="",
                                    batch_id="",
                                    success=False,
                                    error_message="DISCONNECTING",
                                ),
                            )
                        )
                except (asyncio.CancelledError, Exception):
                    pass
            heartbeat_task.cancel()
            try:
                call.cancel()
            except Exception:
                pass
            try:
                await call.done_writing()
            except (asyncio.CancelledError, Exception):
                pass
