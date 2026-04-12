import io
import math

import numpy as np
import pytest
import torch
from app.grpc_server import ParameterServerServicer
from app.proto import parameter_server_pb2


class FakeLock:
    def acquire(self, blocking=True):
        return True

    def release(self):
        return None


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def lock(self, *_args, **_kwargs):
        return FakeLock()


class FakeStorage:
    def __init__(self, redis_client):
        self.enable_redis = True
        self.redis_client = redis_client
        self.parameters = {}
        self.current_versions = {}

    def get_current_version(self, model_id):
        return self.current_versions.get(model_id, 0)

    def _persist_to_redis(self, model_id, version, params):
        buffer = io.BytesIO()
        arrays = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            for k, v in params.items()
        }
        np.savez(buffer, **arrays)
        self.redis_client.set(f"params:{model_id}:v{version}", buffer.getvalue())
        self.redis_client.set(f"params:{model_id}:current_version", str(version))


@pytest.mark.asyncio
async def test_push_gradients_applies_staleness_and_momentum() -> None:
    servicer = ParameterServerServicer()
    servicer.momentum = 0.9
    servicer.staleness_lambda = 0.3

    fake_redis = FakeRedis()
    storage = FakeStorage(fake_redis)
    servicer.storage = storage

    model_id = "job-1"
    initial_weights = {"w": torch.tensor([10.0])}
    storage._persist_to_redis(model_id, 1, initial_weights)
    storage.parameters[model_id] = initial_weights
    storage.current_versions[model_id] = 1

    gradients = {"w": torch.tensor([2.0])}
    payload = servicer._dump_tensor_dict(gradients)

    request = parameter_server_pb2.GradientsUpdate(
        job_id=model_id,
        worker_id="worker-1",
        version=0,
        gradients=payload,
        compression_type="none",
        learning_rate=0.1,
    )

    response = await servicer.PushGradients(request, context=None)
    assert response.success is True
    assert response.new_version == 2

    updated = storage.parameters[model_id]["w"].item()
    expected_weight = math.exp(-0.3 * 1)
    expected = 10.0 - 0.1 * expected_weight * 2.0
    assert updated == pytest.approx(expected, rel=1e-6)

    request_2 = parameter_server_pb2.GradientsUpdate(
        job_id=model_id,
        worker_id="worker-1",
        version=1,
        gradients=payload,
        compression_type="none",
        learning_rate=0.1,
    )
    response_2 = await servicer.PushGradients(request_2, context=None)

    assert response_2.success is True
    assert response_2.new_version == 3

    updated_2 = storage.parameters[model_id]["w"].item()
    momentum_buffer = 0.9 * 2.0 + 2.0
    expected_2 = expected - 0.1 * expected_weight * momentum_buffer
    assert updated_2 == pytest.approx(expected_2, rel=1e-6)
