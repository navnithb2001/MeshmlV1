"""Model Registry gRPC client for Parameter Server."""

import os

import grpc
from app.proto import model_registry_pb2, model_registry_pb2_grpc


class ModelRegistryClient:
    def __init__(self, grpc_url: str | None = None):
        self.grpc_url = grpc_url or os.getenv(
            "MODEL_REGISTRY_GRPC_URL", "model-registry-service:50052"
        )

    async def upload_checkpoint(self, model_id: int, checkpoint_type: str, state_dict: bytes):
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = model_registry_pb2_grpc.ModelRegistryStub(channel)
            return await stub.UploadCheckpoint(
                model_registry_pb2.CheckpointUploadRequest(
                    model_id=model_id, checkpoint_type=checkpoint_type, state_dict=state_dict
                )
            )

    async def upload_final_model(self, model_id: int, state_dict: bytes):
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = model_registry_pb2_grpc.ModelRegistryStub(channel)
            return await stub.UploadFinalModel(
                model_registry_pb2.FinalModelUploadRequest(model_id=model_id, state_dict=state_dict)
            )
