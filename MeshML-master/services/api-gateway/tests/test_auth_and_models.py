import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from app.routers import models
from app.routers.auth import get_current_user
from app.utils.security import create_access_token, decode_access_token
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(models.router, prefix="/api/models")

    async def _current_user_override() -> SimpleNamespace:
        return SimpleNamespace(id=uuid.uuid4(), email="tester@example.com")

    app.dependency_overrides[get_current_user] = _current_user_override
    return app


@pytest.mark.asyncio
async def test_upload_model_validates_required_fields(app: FastAPI) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/models/upload",
            data={"group_id": str(uuid.uuid4())},
            files={"file": ("model.py", b"print('x')", "text/x-python")},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_model_maps_grpc_registration_error_to_502(app: FastAPI) -> None:
    with patch("app.routers.models.ModelRegistryClient") as client_ctor:
        client_instance = client_ctor.return_value
        client_instance.register_new_model = AsyncMock(side_effect=RuntimeError("grpc unavailable"))

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/models/upload",
                data={"name": "resnet", "group_id": str(uuid.uuid4())},
                files={"file": ("model.py", b"print('x')", "text/x-python")},
            )

    assert response.status_code == 502
    assert response.json()["detail"] == "Model registry registration failed"


@pytest.mark.asyncio
async def test_upload_model_maps_signed_upload_error_to_502(app: FastAPI) -> None:
    with patch("app.routers.models.ModelRegistryClient") as client_ctor:
        client_instance = client_ctor.return_value
        client_instance.register_new_model = AsyncMock(
            return_value=SimpleNamespace(
                model_id=1, upload_url="http://signed-upload", gcs_path="gs://bucket/path"
            )
        )
        client_instance.finalize_model_upload = AsyncMock()

        put_response = SimpleNamespace(status_code=500)
        http_client = AsyncMock()
        http_client.put.return_value = put_response
        http_client.__aenter__.return_value = http_client
        http_client.__aexit__.return_value = False
        with patch("app.routers.models.httpx.AsyncClient", return_value=http_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/models/upload",
                    data={"name": "resnet", "group_id": str(uuid.uuid4())},
                    files={"file": ("model.py", b"print('x')", "text/x-python")},
                )

    assert response.status_code == 502
    assert response.json()["detail"].startswith("Signed upload failed")


def test_jwt_create_and_decode_round_trip() -> None:
    token = create_access_token({"sub": "user-1", "email": "user@example.com"})
    payload = decode_access_token(token)

    assert payload["sub"] == "user-1"
    assert payload["email"] == "user@example.com"
    assert "exp" in payload


@pytest.mark.parametrize("token", ["invalid-token", "", "abc.def.ghi"])
def test_jwt_decode_invalid_token_raises(token: str) -> None:
    with pytest.raises(Exception):
        decode_access_token(token)
