from types import SimpleNamespace
from unittest.mock import patch

import pytest
from app.grpc_server import TaskOrchestratorServicer
from app.proto import task_orchestrator_pb2


class DummyAssignmentEngine:
    def __init__(self, _stream_manager):
        pass

    async def run(self, stop_event):
        await stop_event.wait()


class DummyTask:
    def cancel(self):
        return None


def _fake_create_task(coro):
    coro.close()
    return DummyTask()


class FakeWorkerDiscovery:
    def __init__(self):
        self.config = SimpleNamespace(heartbeat_timeout_seconds=30)
        self.last_registration = None

    def register_worker(self, **kwargs):
        self.last_registration = kwargs


class FakeWorkerRegistry:
    def update_heartbeat(self, worker_id, status):
        return worker_id == "worker-1" and status == "idle"


@pytest.mark.asyncio
async def test_register_worker_returns_generated_worker():
    with (
        patch("app.grpc_server.AssignmentEngine", DummyAssignmentEngine),
        patch("app.grpc_server.asyncio.create_task", side_effect=_fake_create_task),
        patch.object(TaskOrchestratorServicer, "_resolve_worker_groups", return_value=[]),
    ):
        discovery = FakeWorkerDiscovery()
        registry = FakeWorkerRegistry()
        servicer = TaskOrchestratorServicer(
            worker_discovery=discovery,
            job_queue=SimpleNamespace(redis=None),
            task_assignment=SimpleNamespace(),
            worker_registry=registry,
        )

        request = task_orchestrator_pb2.WorkerCapabilities(
            user_id="user-1",
            worker_name="worker-a",
            cpu_cores=4,
            ram_bytes=8 * 1024 * 1024 * 1024,
        )

        response = await servicer.RegisterWorker(request, context=None)

        assert response.worker_id.startswith("worker-")
        assert response.heartbeat_interval_seconds == 30
        assert discovery.last_registration is not None


@pytest.mark.asyncio
async def test_send_heartbeat_acknowledges_registered_worker():
    with (
        patch("app.grpc_server.AssignmentEngine", DummyAssignmentEngine),
        patch("app.grpc_server.asyncio.create_task", side_effect=_fake_create_task),
        patch.object(TaskOrchestratorServicer, "_resolve_worker_groups", return_value=[]),
    ):
        servicer = TaskOrchestratorServicer(
            worker_discovery=FakeWorkerDiscovery(),
            job_queue=SimpleNamespace(redis=None),
            task_assignment=SimpleNamespace(),
            worker_registry=FakeWorkerRegistry(),
        )

        response = await servicer.SendHeartbeat(
            task_orchestrator_pb2.Heartbeat(worker_id="worker-1", status="idle"),
            context=None,
        )

        assert response.success is True
        assert response.message == "ok"
