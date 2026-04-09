"""API router for dataset sharding endpoints.

Provides HTTP endpoints to create shards and batches from a dataset path.
"""

import logging
from typing import Any, Dict, List, Optional

from app.config import settings
from app.routers.distribution import get_batch_manager
from app.services.batch_persistence import persist_batches
from app.services.batch_storage import BatchManager
from app.services.dataset_loader import DatasetFormat, create_loader
from app.services.dataset_sharder import DatasetSharder, ShardingConfig, ShardingStrategy
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shard", tags=["sharding"])


class ShardDatasetRequest(BaseModel):
    """Request to shard a dataset and create batches."""

    dataset_id: str = Field(..., description="Dataset ID")
    job_id: Optional[str] = Field(default=None, description="Job ID for batch tracking")
    model_id: Optional[str] = Field(default=None, description="Model ID for batch tracking")
    dataset_path: str = Field(..., description="Dataset path (local or gs://)")
    format: Optional[str] = Field(
        default=None, description="Dataset format: imagefolder, coco, csv (auto-detect if omitted)"
    )
    num_shards: int = Field(10, gt=0, le=1000, description="Number of shards to create")
    strategy: str = Field(
        default="stratified",
        description="Sharding strategy: random, stratified, non_iid, sequential",
    )
    batch_size: Optional[int] = Field(
        default=None, description="Batch size (defaults to service setting)"
    )
    seed: int = Field(default=42, description="Random seed for sharding")


@router.post("", response_model=dict)
async def shard_dataset(
    request: ShardDatasetRequest, batch_manager: BatchManager = Depends(get_batch_manager)
):
    """
    Create dataset shards and store batches.

    Returns summary statistics and batch metadata counts.
    """
    try:
        # Parse dataset format if provided
        dataset_format = None
        if request.format:
            try:
                dataset_format = DatasetFormat(request.format)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported dataset format: {request.format}"
                )

        # Create loader and read metadata
        loader = create_loader(request.dataset_path, format=dataset_format)
        metadata = loader.load_metadata()

        # Parse sharding strategy
        try:
            strategy = ShardingStrategy(request.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unsupported sharding strategy: {request.strategy}"
            )

        # Build sharding config
        batch_size = request.batch_size or settings.DEFAULT_BATCH_SIZE
        config = ShardingConfig(
            num_shards=request.num_shards,
            strategy=strategy,
            batch_size=batch_size,
            seed=request.seed,
        )

        # Create shards
        sharder = DatasetSharder(loader, config)
        shards = sharder.create_shards()

        # Create batches from shards
        total_batches = 0
        shard_summaries: List[Dict[str, Any]] = []
        all_batches: List[Any] = []
        for shard in shards:
            batches = batch_manager.create_batches_from_shard(
                shard=shard, loader=loader, batch_size=batch_size
            )
            total_batches += len(batches)
            all_batches.extend(batches)
            shard_summaries.append(
                {
                    "shard_id": shard.shard_id,
                    "num_samples": shard.num_samples,
                    "num_batches": len(batches),
                    "class_distribution": shard.class_distribution,
                }
            )

        if request.job_id and request.model_id:
            await persist_batches(
                job_id=request.job_id, model_id=request.model_id, batches=all_batches
            )

        return {
            "success": True,
            "dataset_id": request.dataset_id,
            "dataset_path": request.dataset_path,
            "format": metadata.format.value,
            "total_samples": metadata.total_samples,
            "num_classes": metadata.num_classes,
            "num_shards": len(shards),
            "batch_size": batch_size,
            "total_batches": total_batches,
            "strategy": strategy.value,
            "shards": shard_summaries,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sharding failed for dataset {request.dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
