"""gRPC server for Model Registry."""

import asyncio
import hashlib
import logging
import os
from typing import Optional
from urllib.parse import urlparse
from uuid import UUID

import boto3
import grpc
from botocore.config import Config
from sqlalchemy import select, text

from .config import settings
from .database import async_session_maker
from .models import Model, ModelState
from .proto import model_registry_pb2, model_registry_pb2_grpc
from .storage import GCSClient

logger = logging.getLogger(__name__)


class ModelRegistryServicer(model_registry_pb2_grpc.ModelRegistryServicer):
    """gRPC servicer implementing Model Registry APIs."""

    def __init__(self, gcs_client: Optional[GCSClient]):
        self.gcs_client = gcs_client
        self.emulator_url = os.getenv("STORAGE_EMULATOR_URL")
        self.public_emulator_url = os.getenv("STORAGE_PUBLIC_URL", self.emulator_url)
        self.emulator_bucket = settings.GCS_BUCKET_NAME

    def _get_emulator_client(self, endpoint_url: Optional[str] = None):
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or "meshml"
        secret_key = (
            os.getenv("AWS_SECRET_ACCESS_KEY")
            or os.getenv("MINIO_ROOT_PASSWORD")
            or "meshml_minio_password"
        )
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url or self.emulator_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )

    def _ensure_emulator_bucket(self, client) -> None:
        try:
            client.head_bucket(Bucket=self.emulator_bucket)
        except Exception:
            client.create_bucket(Bucket=self.emulator_bucket)

    def _rewrite_presigned_url_for_public(self, url: str) -> str:
        """
        Replace docker-internal storage host with host-accessible endpoint for
        URLs consumed by host workers.
        """
        if not self.emulator_url or not self.public_emulator_url:
            return url
        internal = urlparse(self.emulator_url)
        public = urlparse(self.public_emulator_url)
        if not internal.netloc or not public.netloc or internal.netloc == public.netloc:
            return url
        return url.replace(
            f"{internal.scheme}://{internal.netloc}",
            f"{public.scheme}://{public.netloc}",
            1,
        )

    def _upload_bytes(self, key: str, data: bytes) -> str:
        if self.emulator_url:
            client = self._get_emulator_client()
            self._ensure_emulator_bucket(client)
            client.put_object(Bucket=self.emulator_bucket, Key=key, Body=data)
            return f"s3://{self.emulator_bucket}/{key}"

        if not self.gcs_client:
            raise RuntimeError("GCS not available")

        blob = self.gcs_client.bucket.blob(key)
        blob.upload_from_string(data, content_type="application/octet-stream")
        return self.gcs_client.get_gcs_uri(key)

    def _default_model_key(self, model_id: int, filename: str = "model.py") -> str:
        return f"{settings.MODEL_STORAGE_PREFIX}/{model_id}/{filename}"

    def _resolve_storage_key(self, model_id: int, existing_path: Optional[str]) -> str:
        default_key = self._default_model_key(model_id)
        if not existing_path:
            return default_key

        if existing_path.startswith("gs://"):
            parts = existing_path[5:].split("/", 1)
            return parts[1] if len(parts) > 1 else default_key
        if existing_path.startswith("s3://"):
            parts = existing_path[5:].split("/", 1)
            return parts[1] if len(parts) > 1 else default_key
        return existing_path

    def _generate_upload_url(self, key: str, expires_in: int = 3600) -> str:
        if self.emulator_url:
            # Upload URL is consumed by in-cluster services (API Gateway), so keep
            # internal docker endpoint here.
            client = self._get_emulator_client(endpoint_url=self.emulator_url)
            self._ensure_emulator_bucket(client)
            return client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.emulator_bucket, "Key": key, "ContentType": "text/x-python"},
                ExpiresIn=expires_in,
            )

        if not self.gcs_client:
            raise RuntimeError("GCS not available")

        blob = self.gcs_client.bucket.blob(key)
        return blob.generate_signed_url(
            version="v4",
            expiration=expires_in,
            method="PUT",
            content_type="text/x-python",
        )

    def _generate_download_url(self, key: str, expires_in: int = 3600) -> str:
        if self.emulator_url:
            # Bucket checks must use docker-internal endpoint; URL signing for host workers
            # must use the public endpoint so the Host header matches the signature.
            internal_client = self._get_emulator_client(endpoint_url=self.emulator_url)
            self._ensure_emulator_bucket(internal_client)
            signing_client = self._get_emulator_client(endpoint_url=self.public_emulator_url)
            download_url = signing_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.emulator_bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return download_url

        if not self.gcs_client:
            raise RuntimeError("GCS not available")

        blob = self.gcs_client.bucket.blob(key)
        return blob.generate_signed_url(version="v4", expiration=expires_in, method="GET")

    async def RegisterNewModel(self, request, context):
        try:
            if not self.gcs_client and not self.emulator_url:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "GCS not available")

            async with async_session_maker() as session:
                try:
                    created_by = UUID(request.created_by_user_id)
                except Exception:
                    created_by = UUID("00000000-0000-0000-0000-000000000001")

                model = Model(
                    name=request.name,
                    description=request.description or None,
                    group_id=UUID(request.group_id),
                    created_by_user_id=created_by,
                    architecture_type=request.architecture_type or None,
                    dataset_type=request.dataset_type or None,
                    model_metadata=dict(request.metadata),
                    version=request.version or "1.0.0",
                    state=ModelState.UPLOADING.value,
                )
                session.add(model)
                await session.commit()
                await session.refresh(model)

                key = self._default_model_key(model.id)
                upload_url = self._generate_upload_url(key)
                if self.emulator_url:
                    gcs_path = f"s3://{self.emulator_bucket}/{key}"
                else:
                    gcs_path = self.gcs_client.get_gcs_uri(key)

                return model_registry_pb2.RegisterModelResponse(
                    model_id=model.id,
                    upload_url=upload_url,
                    gcs_path=gcs_path,
                    expires_in_seconds=3600,
                )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def FinalizeModelUpload(self, request, context):
        try:
            async with async_session_maker() as session:
                result = await session.execute(select(Model).where(Model.id == request.model_id))
                model = result.scalar_one_or_none()
                if not model:
                    await context.abort(grpc.StatusCode.NOT_FOUND, "Model not found")

                model.gcs_path = request.gcs_path or model.gcs_path
                model.file_size_bytes = request.file_size_bytes or model.file_size_bytes
                model.file_hash = request.file_hash or model.file_hash
                model.state = ModelState.READY.value
                await session.commit()

                return model_registry_pb2.FinalizeModelUploadResponse(
                    success=True, message="finalized"
                )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetModelArtifact(self, request, context):
        try:
            if not self.gcs_client and not self.emulator_url:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "GCS not available")

            async with async_session_maker() as session:
                result = await session.execute(select(Model).where(Model.id == request.model_id))
                model = result.scalar_one_or_none()
                if not model:
                    return model_registry_pb2.GetModelArtifactResponse(found=False)

                filename = request.filename or "model.py"
                key = self._resolve_storage_key(model.id, model.gcs_path)
                # Preserve requested filename only when model path is not already explicit.
                if model.gcs_path and (
                    model.gcs_path.endswith(".py") or model.gcs_path.endswith(".pt")
                ):
                    pass
                elif filename:
                    key = self._default_model_key(model.id, filename=filename)

                if self.emulator_url:
                    download_url = self._generate_download_url(key, expires_in=3600)
                    gcs_path = f"s3://{self.emulator_bucket}/{key}"
                else:
                    blob = self.gcs_client.bucket.blob(key)
                    download_url = blob.generate_signed_url(
                        version="v4", expiration=3600, method="GET"
                    )
                    gcs_path = self.gcs_client.get_gcs_uri(key)

                return model_registry_pb2.GetModelArtifactResponse(
                    found=True,
                    download_url=download_url,
                    gcs_path=gcs_path,
                    expires_in_seconds=3600,
                    sha256=model.file_hash or "",
                )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def UploadCheckpoint(self, request, context):
        try:
            blob_path = f"checkpoints/{request.model_id}/{request.checkpoint_type}.pt"
            file_hash = hashlib.sha256(request.state_dict).hexdigest()
            gcs_path = self._upload_bytes(blob_path, request.state_dict)

            async with async_session_maker() as session:
                result = await session.execute(select(Model).where(Model.id == request.model_id))
                model = result.scalar_one_or_none()
                if model:
                    model.file_hash = file_hash
                    parsed_version = None
                    if request.checkpoint_type and request.checkpoint_type.startswith("v"):
                        try:
                            parsed_version = int(request.checkpoint_type[1:])
                        except Exception:
                            parsed_version = None
                    if parsed_version is not None:
                        model.checkpoint_version = max(
                            model.checkpoint_version or 0, parsed_version
                        )
                    else:
                        model.checkpoint_version = (model.checkpoint_version or 0) + 1
                    await session.commit()

            return model_registry_pb2.CheckpointUploadResponse(
                success=True, message="uploaded", gcs_path=gcs_path
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def UploadFinalModel(self, request, context):
        try:
            blob_path = f"final/{request.model_id}/model.pt"
            file_hash = hashlib.sha256(request.state_dict).hexdigest()
            gcs_path = self._upload_bytes(blob_path, request.state_dict)

            async with async_session_maker() as session:
                result = await session.execute(select(Model).where(Model.id == request.model_id))
                model = result.scalar_one_or_none()
                if model:
                    model.gcs_path = gcs_path
                    model.file_hash = file_hash
                    model.state = "COMPLETED"
                    await session.commit()
                    try:
                        await session.execute(
                            text("UPDATE jobs SET status='COMPLETED' WHERE model_id = :model_id"),
                            {"model_id": str(request.model_id)},
                        )
                        await session.commit()
                    except Exception:
                        await session.rollback()

            return model_registry_pb2.FinalModelUploadResponse(
                success=True, message="uploaded", gcs_path=gcs_path
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetFinalModelDownloadUrl(self, request, context):
        try:
            blob_path = f"final/{request.model_id}/model.pt"
            download_url = self._generate_download_url(blob_path, expires_in=3600)
            storage_path = (
                f"s3://{self.emulator_bucket}/{blob_path}"
                if self.emulator_url
                else self.gcs_client.get_gcs_uri(blob_path)
            )
            return model_registry_pb2.GetFinalModelDownloadUrlResponse(
                found=True,
                download_url=download_url,
                storage_path=storage_path,
                expires_in_seconds=3600,
            )
        except Exception:
            return model_registry_pb2.GetFinalModelDownloadUrlResponse(found=False)


def create_grpc_services(gcs_client: Optional[GCSClient]) -> ModelRegistryServicer:
    return ModelRegistryServicer(gcs_client=gcs_client)


async def start_grpc_server(app, host: str, port: int) -> None:
    server = grpc.aio.server()
    servicer = create_grpc_services(getattr(app.state, "gcs_client", None))
    model_registry_pb2_grpc.add_ModelRegistryServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    app.state.grpc_server = server
    logger.info(f"Model Registry gRPC server started on {host}:{port}")
