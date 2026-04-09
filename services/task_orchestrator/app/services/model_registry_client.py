"""Model Registry gRPC client for Task Orchestrator."""

import os

import grpc
from app.proto import model_registry_pb2, model_registry_pb2_grpc


class ModelRegistryClient:
    def __init__(self, grpc_url: str | None = None):
        self.grpc_url = grpc_url or os.getenv(
            "MODEL_REGISTRY_GRPC_URL", "model-registry-service:50052"
        )

    async def get_model_artifact(
        self, model_id: int
    ) -> model_registry_pb2.GetModelArtifactResponse:
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = model_registry_pb2_grpc.ModelRegistryStub(channel)
            return await stub.GetModelArtifact(
                model_registry_pb2.GetModelArtifactRequest(model_id=model_id)
            )

    async def get_final_model_download_url(
        self, model_id: int
    ) -> model_registry_pb2.GetFinalModelDownloadUrlResponse:
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = model_registry_pb2_grpc.ModelRegistryStub(channel)
            return await stub.GetFinalModelDownloadUrl(
                model_registry_pb2.GetFinalModelDownloadUrlRequest(model_id=model_id)
            )
