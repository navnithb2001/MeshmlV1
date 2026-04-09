"""Dataset Sharder gRPC client for Task Orchestrator integration."""

import logging
import os
from typing import Any, Dict, List, Optional

import grpc
from app.proto import dataset_sharder_pb2, dataset_sharder_pb2_grpc

logger = logging.getLogger(__name__)


class DatasetSharderClient:
    """Minimal async gRPC client for Dataset Sharder service."""

    def __init__(self, grpc_url: Optional[str] = None):
        self.grpc_url = grpc_url or os.getenv(
            "DATASET_SHARDER_GRPC_URL", "dataset-sharder-service:50053"
        )

    async def shard_dataset(
        self,
        dataset_id: str,
        job_id: Optional[str],
        model_id: Optional[str],
        dataset_path: str,
        format: Optional[str],
        num_shards: int,
        strategy: str,
        batch_size: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Trigger dataset sharding in Dataset Sharder."""
        payload = {
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
            "format": format,
            "num_shards": num_shards,
            "strategy": strategy,
            "batch_size": batch_size,
            "seed": seed,
        }

        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = dataset_sharder_pb2_grpc.DatasetSharderStub(channel)
            response = await stub.ShardDataset(
                dataset_sharder_pb2.ShardDatasetRequest(
                    dataset_id=dataset_id,
                    job_id=job_id or "",
                    model_id=model_id or "",
                    dataset_path=dataset_path,
                    format=format or "",
                    num_shards=num_shards,
                    strategy=strategy,
                    batch_size=batch_size,
                    seed=seed,
                )
            )
            if not response.success:
                raise RuntimeError(response.message or "Dataset sharding failed")
            return {
                "success": response.success,
                "dataset_id": response.dataset_id,
                "num_shards": response.num_shards,
                "total_batches": response.total_batches,
            }

    async def assign_batches(
        self,
        worker_ids: List[str],
        shard_id: Optional[int] = None,
        strategy: str = "shard_per_worker",
    ) -> Dict[str, Any]:
        """Assign batches to workers in Dataset Sharder."""
        payload: Dict[str, Any] = {"worker_ids": worker_ids, "strategy": strategy}
        if shard_id is not None:
            payload["shard_id"] = shard_id

        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = dataset_sharder_pb2_grpc.DatasetSharderStub(channel)
            response = await stub.AssignBatches(
                dataset_sharder_pb2.AssignBatchesRequest(
                    worker_ids=worker_ids, shard_id=shard_id or 0, strategy=strategy
                )
            )
            if not response.success:
                raise RuntimeError(response.message or "Batch assignment failed")

            assignments: Dict[str, Any] = {}
            for assignment in response.assignments:
                assignments[assignment.worker_id] = {
                    "worker_id": assignment.worker_id,
                    "shard_id": assignment.shard_id,
                    "assigned_batches": list(assignment.assigned_batches),
                    "total_samples": assignment.total_samples,
                    "progress": assignment.progress,
                    "is_complete": assignment.is_complete,
                }
            return {"assignments": assignments}

    async def get_batch_download_url(self, batch_id: str) -> Dict[str, Any]:
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = dataset_sharder_pb2_grpc.DatasetSharderStub(channel)
            response = await stub.GetBatchDownloadUrl(
                dataset_sharder_pb2.GetBatchDownloadUrlRequest(batch_id=batch_id)
            )
            if not response.found:
                return {}
            return {
                "download_url": response.download_url,
                "storage_path": response.storage_path,
                "expires_in_seconds": response.expires_in_seconds,
            }
