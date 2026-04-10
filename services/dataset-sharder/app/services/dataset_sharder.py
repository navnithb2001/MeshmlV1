"""Dataset sharding algorithms for distributed training.

This module provides various sharding strategies to partition datasets across
multiple workers while maintaining data distribution properties.
"""

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from app.services.dataset_loader import DataSample, DatasetLoader, DatasetMetadata

logger = logging.getLogger(__name__)


class ShardingStrategy(str, Enum):
    """Available sharding strategies."""

    RANDOM = "random"  # Random split (IID)
    STRATIFIED = "stratified"  # Maintain class distribution (IID)
    NON_IID = "non_iid"  # Non-IID partitioning (different distributions per worker)
    SEQUENTIAL = "sequential"  # Sequential chunks (for debugging)


class DataDistribution(str, Enum):
    """Data distribution type."""

    IID = "iid"  # Independent and Identically Distributed
    NON_IID = "non_iid"  # Non-IID (skewed distributions)


@dataclass
class ShardMetadata:
    """Metadata for a single shard."""

    shard_id: int
    total_shards: int
    num_samples: int
    sample_indices: List[int]
    class_distribution: Dict[str, int]
    size_bytes: int
    checksum: Optional[str] = None

    def get_balance_ratio(self) -> float:
        """
        Calculate class balance ratio (max_count / min_count).
        Lower is better (1.0 = perfectly balanced).
        """
        if not self.class_distribution:
            return 1.0

        counts = list(self.class_distribution.values())
        if min(counts) == 0:
            return float("inf")

        return max(counts) / min(counts)


@dataclass
class ShardingConfig:
    """Configuration for dataset sharding."""

    num_shards: int
    strategy: ShardingStrategy = ShardingStrategy.STRATIFIED
    batch_size: Optional[int] = None  # If None, auto-calculate
    min_samples_per_shard: int = 10
    max_samples_per_shard: Optional[int] = None
    seed: int = 42

    # Non-IID specific parameters
    non_iid_alpha: float = 0.5  # Dirichlet distribution parameter (lower = more skewed)
    non_iid_classes_per_shard: Optional[int] = None  # Limit classes per shard


class DatasetSharder:
    """Dataset sharding service."""

    def __init__(self, dataset_loader: DatasetLoader, config: ShardingConfig):
        """
        Initialize dataset sharder.

        Args:
            dataset_loader: DatasetLoader instance
            config: Sharding configuration
        """
        self.loader = dataset_loader
        self.config = config
        self.metadata: Optional[DatasetMetadata] = None
        self.shards: List[ShardMetadata] = []

        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)

    def create_shards(self) -> List[ShardMetadata]:
        """
        Create shards based on the configured strategy.

        Returns:
            List of ShardMetadata instances
        """
        # Load dataset metadata
        if not self.metadata:
            self.metadata = self.loader.load_metadata()

        logger.info(
            f"Creating {self.config.num_shards} shards using {self.config.strategy} strategy"
        )
        logger.info(
            f"Dataset: {self.metadata.total_samples} samples, {self.metadata.num_classes} classes"
        )

        # Validate configuration
        self._validate_config()

        # Create shards based on strategy
        if self.config.strategy == ShardingStrategy.RANDOM:
            self.shards = self._create_random_shards()
        elif self.config.strategy == ShardingStrategy.STRATIFIED:
            self.shards = self._create_stratified_shards()
        elif self.config.strategy == ShardingStrategy.NON_IID:
            self.shards = self._create_non_iid_shards()
        elif self.config.strategy == ShardingStrategy.SEQUENTIAL:
            self.shards = self._create_sequential_shards()
        else:
            raise ValueError(f"Unsupported sharding strategy: {self.config.strategy}")

        # Log shard statistics
        self._log_shard_stats()

        return self.shards

    def _validate_config(self):
        """Validate sharding configuration."""
        if self.config.num_shards <= 0:
            raise ValueError("num_shards must be positive")

        if self.config.num_shards > self.metadata.total_samples:
            raise ValueError(
                f"num_shards ({self.config.num_shards}) exceeds total samples "
                f"({self.metadata.total_samples})"
            )

        min_samples = self.metadata.total_samples // self.config.num_shards
        if min_samples < self.config.min_samples_per_shard:
            logger.warning(
                f"Average samples per shard ({min_samples}) is less than "
                f"min_samples_per_shard ({self.config.min_samples_per_shard})"
            )

    def _create_random_shards(self) -> List[ShardMetadata]:
        """
        Create random IID shards.

        Randomly shuffles all samples and divides into equal-sized shards.
        """
        # Create list of all sample indices
        all_indices = list(range(self.metadata.total_samples))
        random.shuffle(all_indices)

        # Calculate samples per shard
        samples_per_shard = self.metadata.total_samples // self.config.num_shards

        shards = []
        for shard_id in range(self.config.num_shards):
            start_idx = shard_id * samples_per_shard

            # Last shard gets remaining samples
            if shard_id == self.config.num_shards - 1:
                end_idx = self.metadata.total_samples
            else:
                end_idx = start_idx + samples_per_shard

            shard_indices = all_indices[start_idx:end_idx]

            # Calculate class distribution for this shard
            class_dist = self._calculate_class_distribution(shard_indices)

            shard = ShardMetadata(
                shard_id=shard_id,
                total_shards=self.config.num_shards,
                num_samples=len(shard_indices),
                sample_indices=shard_indices,
                class_distribution=class_dist,
                size_bytes=self._estimate_shard_size(len(shard_indices)),
            )
            shards.append(shard)

        return shards

    def _create_stratified_shards(self) -> List[ShardMetadata]:
        """
        Create stratified shards maintaining class distribution.

        Each shard gets approximately the same proportion of each class.
        """
        # Group samples by class or bucket
        class_to_indices = self._group_samples_by_class()

        # Shuffle samples within each class
        for class_name in class_to_indices:
            random.shuffle(class_to_indices[class_name])

        # Initialize shards
        shards = [[] for _ in range(self.config.num_shards)]

        # Distribute samples from each class across shards
        for class_name, indices in class_to_indices.items():
            samples_per_shard = len(indices) // self.config.num_shards

            for shard_id in range(self.config.num_shards):
                start_idx = shard_id * samples_per_shard

                # Last shard gets remaining samples
                if shard_id == self.config.num_shards - 1:
                    end_idx = len(indices)
                else:
                    end_idx = start_idx + samples_per_shard

                shard_samples = indices[start_idx:end_idx]
                shards[shard_id].extend(shard_samples)

        # Shuffle samples within each shard
        for shard_samples in shards:
            random.shuffle(shard_samples)

        # Create ShardMetadata objects
        shard_metadata_list = []
        for shard_id, shard_indices in enumerate(shards):
            class_dist = self._calculate_class_distribution(shard_indices)

            shard = ShardMetadata(
                shard_id=shard_id,
                total_shards=self.config.num_shards,
                num_samples=len(shard_indices),
                sample_indices=shard_indices,
                class_distribution=class_dist,
                size_bytes=self._estimate_shard_size(len(shard_indices)),
            )
            shard_metadata_list.append(shard)

        return shard_metadata_list

    def _create_non_iid_shards(self) -> List[ShardMetadata]:
        """
        Create non-IID shards with skewed class distributions.

        Uses Dirichlet distribution to create realistic non-IID partitions.
        """
        # Group samples by class
        class_to_indices = self._group_samples_by_class()

        num_classes = len(class_to_indices)

        # Use Dirichlet distribution to generate class proportions for each shard
        # Lower alpha = more skewed (non-IID)
        alpha = [self.config.non_iid_alpha] * num_classes
        proportions = np.random.dirichlet(alpha, self.config.num_shards)

        # Initialize shards
        shards = [[] for _ in range(self.config.num_shards)]

        # Distribute samples based on Dirichlet proportions
        for class_idx, (class_name, indices) in enumerate(class_to_indices.items()):
            random.shuffle(indices)

            # Calculate how many samples each shard should get from this class
            class_proportions = proportions[:, class_idx]
            class_proportions = class_proportions / class_proportions.sum()  # Normalize

            samples_per_shard = (class_proportions * len(indices)).astype(int)

            # Adjust to ensure all samples are distributed
            diff = len(indices) - samples_per_shard.sum()
            if diff > 0:
                # Add remaining samples to shards with largest proportions
                top_shards = np.argsort(class_proportions)[-diff:]
                for shard_id in top_shards:
                    samples_per_shard[shard_id] += 1

            # Distribute samples
            start_idx = 0
            for shard_id in range(self.config.num_shards):
                num_samples = samples_per_shard[shard_id]
                end_idx = start_idx + num_samples

                shard_samples = indices[start_idx:end_idx]
                shards[shard_id].extend(shard_samples)

                start_idx = end_idx

        # Shuffle samples within each shard
        for shard_samples in shards:
            random.shuffle(shard_samples)

        # Create ShardMetadata objects
        shard_metadata_list = []
        for shard_id, shard_indices in enumerate(shards):
            class_dist = self._calculate_class_distribution(shard_indices)

            shard = ShardMetadata(
                shard_id=shard_id,
                total_shards=self.config.num_shards,
                num_samples=len(shard_indices),
                sample_indices=shard_indices,
                class_distribution=class_dist,
                size_bytes=self._estimate_shard_size(len(shard_indices)),
            )
            shard_metadata_list.append(shard)

        return shard_metadata_list

    def _create_sequential_shards(self) -> List[ShardMetadata]:
        """
        Create sequential shards (for debugging/testing).

        Divides dataset into contiguous chunks without shuffling.
        """
        samples_per_shard = self.metadata.total_samples // self.config.num_shards

        shards = []
        for shard_id in range(self.config.num_shards):
            start_idx = shard_id * samples_per_shard

            if shard_id == self.config.num_shards - 1:
                end_idx = self.metadata.total_samples
            else:
                end_idx = start_idx + samples_per_shard

            shard_indices = list(range(start_idx, end_idx))
            class_dist = self._calculate_class_distribution(shard_indices)

            shard = ShardMetadata(
                shard_id=shard_id,
                total_shards=self.config.num_shards,
                num_samples=len(shard_indices),
                sample_indices=shard_indices,
                class_distribution=class_dist,
                size_bytes=self._estimate_shard_size(len(shard_indices)),
            )
            shards.append(shard)

        return shards

    def _group_samples_by_class(self) -> Dict[str, List[int]]:
        """
        Group sample indices by their class label or bucket ID (for regression).
        Automatically detects continuous targets and uses quantile binning.
        """
        if hasattr(self, '_cached_groups') and self._cached_groups:
            return self._cached_groups

        labels_by_idx = []
        for batch in self.loader.stream_samples(batch_size=1000):
            for sample in batch:
                labels_by_idx.append(sample.label)
                
        labels_array = np.array(labels_by_idx)
        is_continuous = False
        
        # Check if numeric
        if np.issubdtype(labels_array.dtype, np.number):
            if np.issubdtype(labels_array.dtype, np.floating):
                is_continuous = True
            elif len(np.unique(labels_array)) / max(1, len(labels_array)) > 0.5:
                is_continuous = True

        class_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._sample_to_class = {}
        
        if is_continuous:
            logger.info("Detected continuous target labels. Applying quantile binning.")
            num_bins = min(10, len(np.unique(labels_array)))
            if num_bins < 2:
                for idx in range(len(labels_array)):
                    class_name = "Bucket_0"
                    class_to_indices[class_name].append(idx)
                    self._sample_to_class[idx] = class_name
            else:
                quantiles = np.linspace(0, 1, num_bins + 1)
                bin_edges = np.quantile(labels_array, quantiles)
                bin_edges[0] -= 1e-5
                bin_edges[-1] += 1e-5
                bucket_ids = np.digitize(labels_array, bin_edges) - 1
                for idx, bucket_id in enumerate(bucket_ids):
                    class_name = f"Bucket_{bucket_id}"
                    class_to_indices[class_name].append(idx)
                    self._sample_to_class[idx] = class_name
        else:
            for idx, label in enumerate(labels_by_idx):
                if isinstance(label, int) and self.metadata and hasattr(self.metadata, 'class_names') and self.metadata.class_names and label < len(self.metadata.class_names):
                    class_name = self.metadata.class_names[label]
                else:
                    class_name = str(label)
                class_to_indices[class_name].append(idx)
                self._sample_to_class[idx] = class_name
                
        self._cached_groups = class_to_indices
        return class_to_indices

    def _calculate_class_distribution(self, indices: List[int]) -> Dict[str, int]:
        """Calculate class distribution for a list of sample indices."""
        self._group_samples_by_class() # Ensure buckets are initialized
        
        class_counts = defaultdict(int)
        for idx in indices:
            class_name = self._sample_to_class[idx]
            class_counts[class_name] += 1

        return dict(class_counts)

    def _estimate_shard_size(self, num_samples: int) -> int:
        """Estimate shard size in bytes."""
        avg_sample_size = self.metadata.total_size_bytes / self.metadata.total_samples
        return int(num_samples * avg_sample_size)

    def _log_shard_stats(self):
        """Log statistics about created shards."""
        logger.info("=" * 70)
        logger.info("Shard Statistics")
        logger.info("=" * 70)

        for shard in self.shards:
            logger.info(f"\nShard {shard.shard_id}:")
            logger.info(f"  Samples: {shard.num_samples}")
            logger.info(f"  Size: {shard.size_bytes / (1024**2):.2f} MB")
            logger.info(f"  Balance ratio: {shard.get_balance_ratio():.2f}")
            logger.info(f"  Class distribution: {shard.class_distribution}")

        # Overall statistics
        total_samples = sum(s.num_samples for s in self.shards)
        avg_samples = total_samples / len(self.shards)
        std_samples = np.std([s.num_samples for s in self.shards])
        avg_balance = np.mean([s.get_balance_ratio() for s in self.shards])

        logger.info("\nOverall Statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Avg samples per shard: {avg_samples:.1f} ± {std_samples:.1f}")
        logger.info(f"  Avg balance ratio: {avg_balance:.2f}")
        logger.info("=" * 70)

    def get_shard(self, shard_id: int) -> ShardMetadata:
        """Get metadata for a specific shard."""
        if shard_id >= len(self.shards):
            raise ValueError(f"Shard {shard_id} does not exist")
        return self.shards[shard_id]

    def calculate_batch_size(
        self,
        target_batches_per_epoch: int = 100,
        min_batch_size: int = 8,
        max_batch_size: int = 512,
    ) -> int:
        """
        Calculate optimal batch size for training.

        Args:
            target_batches_per_epoch: Desired number of batches per epoch
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size

        Returns:
            Optimal batch size
        """
        if not self.shards:
            raise ValueError("No shards created yet. Call create_shards() first.")

        # Use the smallest shard for calculation
        min_shard_size = min(s.num_samples for s in self.shards)

        # Calculate batch size to get target number of batches
        batch_size = max(min_batch_size, min_shard_size // target_batches_per_epoch)
        batch_size = min(batch_size, max_batch_size)

        # Round to nearest power of 2 for efficiency
        batch_size = 2 ** int(math.log2(batch_size))

        logger.info(f"Calculated batch size: {batch_size}")
        logger.info(f"  Min shard size: {min_shard_size}")
        logger.info(f"  Batches per epoch: {min_shard_size // batch_size}")

        return batch_size


def analyze_distribution_quality(shards: List[ShardMetadata]) -> Dict[str, Any]:
    """
    Analyze the quality of shard distribution.

    Args:
        shards: List of ShardMetadata

    Returns:
        Dictionary with distribution metrics
    """
    # Calculate statistics
    shard_sizes = [s.num_samples for s in shards]
    balance_ratios = [s.get_balance_ratio() for s in shards]

    # Size distribution
    size_stats = {
        "mean": np.mean(shard_sizes),
        "std": np.std(shard_sizes),
        "min": np.min(shard_sizes),
        "max": np.max(shard_sizes),
        "cv": np.std(shard_sizes) / np.mean(shard_sizes) if np.mean(shard_sizes) > 0 else 0,
    }

    # Balance quality
    balance_stats = {
        "mean": np.mean(balance_ratios),
        "std": np.std(balance_ratios),
        "min": np.min(balance_ratios),
        "max": np.max(balance_ratios),
    }

    # Per-class distribution variance (measure of IID vs non-IID)
    all_classes = set()
    for shard in shards:
        all_classes.update(shard.class_distribution.keys())

    class_variances = {}
    for class_name in all_classes:
        class_counts = [shard.class_distribution.get(class_name, 0) for shard in shards]
        class_variances[class_name] = np.var(class_counts)

    avg_class_variance = np.mean(list(class_variances.values()))

    return {
        "num_shards": len(shards),
        "total_samples": sum(shard_sizes),
        "size_distribution": size_stats,
        "balance_distribution": balance_stats,
        "class_variance": {"per_class": class_variances, "average": avg_class_variance},
        "quality_score": _calculate_quality_score(size_stats, balance_stats, avg_class_variance),
    }


def _calculate_quality_score(
    size_stats: Dict[str, float], balance_stats: Dict[str, float], avg_class_variance: float
) -> float:
    """
    Calculate overall shard quality score (0-100).

    Higher is better.
    """
    # Size uniformity (lower CV is better)
    size_score = max(0, 100 - size_stats["cv"] * 100)

    # Balance quality (closer to 1.0 is better)
    balance_score = max(0, 100 - (balance_stats["mean"] - 1.0) * 20)

    # Class distribution (lower variance is better for IID)
    # Normalize by dividing by typical variance
    class_score = max(0, 100 - min(avg_class_variance / 100, 1.0) * 100)

    # Weighted average
    overall_score = size_score * 0.3 + balance_score * 0.4 + class_score * 0.3

    return round(overall_score, 2)
