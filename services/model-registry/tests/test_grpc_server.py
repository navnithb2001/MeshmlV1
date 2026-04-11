from types import SimpleNamespace
from unittest.mock import patch

import pytest
from app.grpc_server import ModelRegistryServicer
from app.proto import model_registry_pb2


class FakeContext:
    async def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")


class FakeResult:
    def __init__(self, model):
        self._model = model

    def scalar_one_or_none(self):
        return self._model


class FakeSession:
    def __init__(self, model):
        self.model = model
        self.committed = 0

    async def execute(self, *_args, **_kwargs):
        return FakeResult(self.model)

    async def commit(self):
        self.committed += 1


class FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_upload_checkpoint_increments_checkpoint_version():
    servicer = ModelRegistryServicer(gcs_client=None)
    model = SimpleNamespace(file_hash=None, checkpoint_version=0)
    fake_session = FakeSession(model)

    with (
        patch("app.grpc_server.async_session_maker", return_value=FakeSessionContext(fake_session)),
        patch.object(
            servicer, "_upload_bytes", return_value="s3://meshml-models/checkpoints/1/v1.pt"
        ),
    ):
        response = await servicer.UploadCheckpoint(
            model_registry_pb2.CheckpointUploadRequest(
                model_id=1,
                state_dict=b"weights",
                checkpoint_type="v1",
            ),
            FakeContext(),
        )

    assert response.success is True
    assert model.checkpoint_version == 1
    assert model.file_hash is not None


@pytest.mark.asyncio
async def test_get_final_model_download_url_returns_not_found_on_generation_error():
    servicer = ModelRegistryServicer(gcs_client=None)
    with patch.object(servicer, "_generate_download_url", side_effect=RuntimeError("storage down")):
        response = await servicer.GetFinalModelDownloadUrl(
            model_registry_pb2.GetFinalModelDownloadUrlRequest(model_id=1),
            FakeContext(),
        )

        assert response.found is False
