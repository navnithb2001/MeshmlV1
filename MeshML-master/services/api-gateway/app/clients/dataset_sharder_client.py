"""Dataset Sharder gRPC client for API Gateway."""

import os

import grpc
from app.proto import dataset_sharder_pb2, dataset_sharder_pb2_grpc


class DatasetSharderClient:
    def __init__(self, grpc_url: str | None = None):
        self.grpc_url = grpc_url or os.getenv(
            "DATASET_SHARDER_GRPC_URL", "dataset-sharder-service:50053"
        )

    async def shard_dataset(
        self, request: dataset_sharder_pb2.ShardDatasetRequest
    ) -> dataset_sharder_pb2.ShardDatasetResponse:
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = dataset_sharder_pb2_grpc.DatasetSharderStub(channel)
            return await stub.StartSharding(request)
