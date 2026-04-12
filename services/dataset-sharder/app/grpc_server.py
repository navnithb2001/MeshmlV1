"""gRPC server for Dataset Sharder."""

import logging
import os
from typing import List, Optional

import boto3
import grpc
from app.config import settings
from app.core.storage import get_dataset_storage
from app.proto import dataset_sharder_pb2, dataset_sharder_pb2_grpc
from app.services.batch_persistence import persist_batches
from app.services.batch_storage import BatchManager, create_storage_backend
from app.services.data_distribution import DataDistributor, DistributionStrategy
from app.services.dataset_loader import DatasetFormat, create_loader
from app.services.dataset_sharder import DatasetSharder, ShardingConfig, ShardingStrategy
from botocore.config import Config

logger = logging.getLogger(__name__)


def _get_batch_manager() -> BatchManager:
    storage_type = os.getenv("BATCH_STORAGE_TYPE")
    if not storage_type:
        # Respect USE_GCS in docker-compose when explicit batch storage type is not set.
        storage_type = "gcs" if settings.USE_GCS else "local"
    if storage_type == "gcs":
        storage = create_storage_backend(
            storage_type="gcs", bucket_name=settings.GCS_BUCKET_DATASETS, base_prefix="batches"
        )
    else:
        storage = create_storage_backend(storage_type="local")
    return BatchManager(storage_backend=storage)


class DatasetSharderServicer(dataset_sharder_pb2_grpc.DatasetSharderServicer):
    def __init__(self):
        self.batch_manager = _get_batch_manager()
        self.distributor = DataDistributor(
            batch_manager=self.batch_manager, strategy=DistributionStrategy.SHARD_PER_WORKER
        )

    async def ShardDataset(self, request, context):
        try:
            dataset_format = None
            if request.format:
                try:
                    dataset_format = DatasetFormat(request.format)
                except ValueError:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, "Unsupported dataset format"
                    )

            import asyncio

            loader = create_loader(request.dataset_path, format=dataset_format)
            metadata = await asyncio.to_thread(loader.load_metadata)

            try:
                strategy = ShardingStrategy(request.strategy or "stratified")
            except ValueError:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "Unsupported sharding strategy"
                )

            batch_size = request.batch_size or settings.DEFAULT_BATCH_SIZE
            config = ShardingConfig(
                num_shards=request.num_shards or 10,
                strategy=strategy,
                batch_size=batch_size,
                seed=request.seed or 42,
            )

            sharder = DatasetSharder(loader, config)
            shards = await asyncio.to_thread(sharder.create_shards)

            total_batches = 0
            shard_summaries: List[dataset_sharder_pb2.ShardSummary] = []
            all_batches = []
            for shard in shards:
                batches = await asyncio.to_thread(
                    self.batch_manager.create_batches_from_shard,
                    shard=shard,
                    loader=loader,
                    batch_size=batch_size,
                )
                total_batches += len(batches)
                all_batches.extend(batches)
                shard_summaries.append(
                    dataset_sharder_pb2.ShardSummary(
                        shard_id=shard.shard_id,
                        num_samples=shard.num_samples,
                        num_batches=len(batches),
                    )
                )
            if request.job_id and request.model_id:
                inserted = await persist_batches(
                    job_id=request.job_id, model_id=request.model_id, batches=all_batches
                )
                logger.info(
                    "Persisted %s batches to data_batches for job_id=%s model_id=%s",
                    inserted,
                    request.job_id,
                    request.model_id,
                )

            return dataset_sharder_pb2.ShardDatasetResponse(
                success=True,
                message="sharded",
                dataset_id=request.dataset_id,
                format=metadata.format.value,
                total_samples=metadata.total_samples,
                num_classes=metadata.num_classes,
                num_shards=len(shards),
                batch_size=batch_size,
                total_batches=total_batches,
                strategy=strategy.value,
                shards=shard_summaries,
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def StartSharding(self, request, context):
        return await self.ShardDataset(request, context)

    async def AssignBatches(self, request, context):
        try:
            if request.strategy:
                try:
                    self.distributor.strategy = DistributionStrategy(request.strategy)
                except ValueError:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid strategy")

            assignments = self.distributor.assign_batches_to_workers(
                worker_ids=list(request.worker_ids),
                shard_id=request.shard_id if request.shard_id > 0 else None,
            )

            assignment_list = []
            for worker_id, assignment in assignments.items():
                assignment_list.append(
                    dataset_sharder_pb2.WorkerAssignment(
                        worker_id=assignment.worker_id,
                        shard_id=assignment.shard_id,
                        assigned_batches=assignment.assigned_batches,
                        total_samples=assignment.total_samples,
                        progress=assignment.get_progress(),
                        is_complete=assignment.is_complete(),
                    )
                )

            return dataset_sharder_pb2.AssignBatchesResponse(
                success=True,
                message=f"Assigned batches to {len(assignments)} workers",
                assignments=assignment_list,
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetWorkerAssignment(self, request, context):
        try:
            assignment = self.distributor.get_worker_assignment(request.worker_id)
            if not assignment:
                return dataset_sharder_pb2.WorkerAssignmentResponse(found=False)

            return dataset_sharder_pb2.WorkerAssignmentResponse(
                found=True,
                assignment=dataset_sharder_pb2.WorkerAssignment(
                    worker_id=assignment.worker_id,
                    shard_id=assignment.shard_id,
                    assigned_batches=assignment.assigned_batches,
                    total_samples=assignment.total_samples,
                    progress=assignment.get_progress(),
                    is_complete=assignment.is_complete(),
                ),
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetBatchDownloadUrl(self, request, context):
        try:
            _, metadata = self.batch_manager.load_batch(request.batch_id)
            storage_path = metadata.storage_path
            if storage_path.startswith("gs://"):
                blob_path = storage_path.replace(f"gs://{settings.GCS_BUCKET_DATASETS}/", "")
                emulator_endpoint = os.getenv("STORAGE_EMULATOR_URL")
                if emulator_endpoint:
                    public_endpoint = os.getenv("STORAGE_PUBLIC_URL", emulator_endpoint)
                    s3 = boto3.client(
                        "s3",
                        endpoint_url=public_endpoint,
                        aws_access_key_id=(
                            os.getenv("AWS_ACCESS_KEY_ID")
                            or os.getenv("MINIO_ROOT_USER")
                            or "meshml"
                        ),
                        aws_secret_access_key=(
                            os.getenv("AWS_SECRET_ACCESS_KEY")
                            or os.getenv("MINIO_ROOT_PASSWORD")
                            or "meshml_minio_password"
                        ),
                        region_name="us-east-1",
                        config=Config(signature_version="s3v4"),
                    )
                    download_url = s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": settings.GCS_BUCKET_DATASETS, "Key": blob_path},
                        ExpiresIn=3600,
                    )
                else:
                    storage_client = get_dataset_storage()
                    download_url = storage_client.generate_presigned_download_url(blob_path)
                return dataset_sharder_pb2.GetBatchDownloadUrlResponse(
                    found=True,
                    download_url=download_url,
                    storage_path=storage_path,
                    expires_in_seconds=3600,
                )

            return dataset_sharder_pb2.GetBatchDownloadUrlResponse(
                found=False, storage_path=storage_path
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def create_grpc_services() -> DatasetSharderServicer:
    return DatasetSharderServicer()


async def start_grpc_server(app, host: str, port: int) -> None:
    server = grpc.aio.server()
    servicer = create_grpc_services()
    dataset_sharder_pb2_grpc.add_DatasetSharderServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    app.state.grpc_server = server
    logger.info(f"Dataset Sharder gRPC server started on {host}:{port}")
