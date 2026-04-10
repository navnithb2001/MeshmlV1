import types

import pytest
from meshml_worker.config import WorkerConfig
from meshml_worker.main import MeshMLWorker


@pytest.mark.asyncio
async def test_worker_uses_parameter_server_grpc(monkeypatch, tmp_path):
    config = WorkerConfig()
    config.worker.id = "worker-test"
    config.parameter_server.grpc_url = "localhost:50052"
    config.storage.base_dir = tmp_path
    config.storage.models_dir = tmp_path / "models"
    config.storage.data_dir = tmp_path / "data"

    captured = {}

    class FakeParameterServerClient:
        def __init__(self, grpc_url: str):
            captured["grpc_url"] = grpc_url

        def connect(self) -> bool:
            captured["connected"] = True
            return True

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            captured["client_type"] = type(kwargs["grpc_client"]).__name__

        async def train(self, **_kwargs):
            return None

    class FakeTaskOrchestratorClient:
        def __init__(self, *args, **kwargs):
            self.worker_id = kwargs.get("worker_id")

        async def register(self):
            return {"cpu_cores": 4, "ram_gb": 8.0, "has_gpu": False}

        async def run_assignment_stream(self, callback, shutdown_event):
            assignment = types.SimpleNamespace(
                job_id="job-1",
                batch_id="batch-1",
                model_url="",
                data_url="",
            )
            await callback(assignment, {"model_id": "model-1", "total_batches": 1})
            shutdown_event.set()

        async def close(self):
            return None

    class FakeMetricsClient:
        def __init__(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "meshml_worker.communication.parameter_server_client.ParameterServerClient",
        FakeParameterServerClient,
    )
    monkeypatch.setattr(
        "meshml_worker.training.trainer.Trainer",
        FakeTrainer,
    )
    monkeypatch.setattr(
        "meshml_worker.communication.task_orchestrator_client.TaskOrchestratorClient",
        FakeTaskOrchestratorClient,
    )
    monkeypatch.setattr(
        "meshml_worker.communication.metrics_client.MetricsClient",
        FakeMetricsClient,
    )

    # Guard rail: worker path must not instantiate HTTP client for PS.
    def _http_client_forbidden(*_args, **_kwargs):
        raise AssertionError("HTTP client should not be used for Parameter Server")

    monkeypatch.setattr(
        "meshml_worker.communication.http_client.HTTPClient.__init__",
        _http_client_forbidden,
    )

    worker = MeshMLWorker(config)
    await worker.run(user_id="user-1")

    assert captured["grpc_url"] == "localhost:50052"
    assert captured["connected"] is True
    assert captured["client_type"] == "FakeParameterServerClient"
