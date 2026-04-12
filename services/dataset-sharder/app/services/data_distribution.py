"""Data distribution service for worker batch downloads.

This module provides HTTP endpoints and management for distributing dataset
batches to distributed workers.
"""

import asyncio
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from app.services.batch_storage import BatchManager, BatchMetadata
from app.services.dataset_sharder import ShardMetadata

logger = logging.getLogger(__name__)


class AssignmentStatus(Enum):
    """Status of batch assignment to worker."""

    PENDING = "pending"  # Assigned but not yet downloaded
    DOWNLOADING = "downloading"  # Worker is downloading
    COMPLETED = "completed"  # Successfully downloaded
    FAILED = "failed"  # Download failed
    REASSIGNED = "reassigned"  # Reassigned to different worker


@dataclass
class BatchAssignment:
    """Assignment of a batch to a worker."""

    assignment_id: str
    batch_id: str
    worker_id: str
    shard_id: int
    batch_index: int
    status: AssignmentStatus
    assigned_at: str
    downloaded_at: Optional[str] = None
    failed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchAssignment":
        """Create from dictionary."""
        if isinstance(data.get("status"), str):
            data["status"] = AssignmentStatus(data["status"])
        return cls(**data)

    def can_retry(self) -> bool:
        """Check if assignment can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class WorkerAssignment:
    """Track all batches assigned to a worker."""

    worker_id: str
    shard_id: int
    assigned_batches: List[str] = field(default_factory=list)
    completed_batches: List[str] = field(default_factory=list)
    failed_batches: List[str] = field(default_factory=list)
    total_samples: int = 0
    assigned_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_progress(self) -> float:
        """Calculate download progress (0.0 to 1.0)."""
        if not self.assigned_batches:
            return 0.0
        return len(self.completed_batches) / len(self.assigned_batches)

    def is_complete(self) -> bool:
        """Check if all batches are downloaded."""
        return len(self.completed_batches) == len(self.assigned_batches)


class DistributionStrategy(Enum):
    """Strategy for distributing batches to workers."""

    ROUND_ROBIN = "round_robin"  # Distribute evenly in rotation
    LOAD_BALANCED = "load_balanced"  # Balance by worker capacity
    SHARD_PER_WORKER = "shard_per_worker"  # Each worker gets complete shard
    LOCALITY_AWARE = "locality_aware"  # Minimize data transfer


class DataDistributor:
    """Manages distribution of dataset batches to workers."""

    def __init__(
        self,
        batch_manager: BatchManager,
        strategy: DistributionStrategy = DistributionStrategy.SHARD_PER_WORKER,
    ):
        """
        Initialize data distributor.

        Args:
            batch_manager: BatchManager instance
            strategy: Distribution strategy
        """
        self.batch_manager = batch_manager
        self.strategy = strategy

        # Assignment tracking
        self.assignments: Dict[str, BatchAssignment] = {}  # assignment_id -> assignment
        self.worker_assignments: Dict[str, WorkerAssignment] = {}  # worker_id -> assignment
        self.batch_to_worker: Dict[str, str] = {}  # batch_id -> worker_id

        # Synchronization
        self._lock = threading.Lock()

        logger.info(f"Initialized DataDistributor with strategy: {strategy.value}")

    def assign_batches_to_workers(
        self, worker_ids: List[str], shard_id: Optional[int] = None
    ) -> Dict[str, WorkerAssignment]:
        """
        Assign batches to workers based on strategy.

        Args:
            worker_ids: List of worker IDs
            shard_id: Optional shard ID to filter batches

        Returns:
            Dictionary mapping worker_id to WorkerAssignment
        """
        with self._lock:
            # Get available batches
            batches = self.batch_manager.list_batches(shard_id=shard_id)

            if not batches:
                logger.warning("No batches available for assignment")
                return {}

            if not worker_ids:
                raise ValueError("No worker IDs provided")

            # Distribute based on strategy
            if self.strategy == DistributionStrategy.SHARD_PER_WORKER:
                assignments = self._assign_shard_per_worker(worker_ids, batches)
            elif self.strategy == DistributionStrategy.ROUND_ROBIN:
                assignments = self._assign_round_robin(worker_ids, batches)
            elif self.strategy == DistributionStrategy.LOAD_BALANCED:
                assignments = self._assign_load_balanced(worker_ids, batches)
            else:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

            # Create batch assignments
            for worker_id, worker_assignment in assignments.items():
                for batch_id in worker_assignment.assigned_batches:
                    batch_meta = next(b for b in batches if b.batch_id == batch_id)

                    assignment_id = f"{worker_id}_{batch_id}"

                    assignment = BatchAssignment(
                        assignment_id=assignment_id,
                        batch_id=batch_id,
                        worker_id=worker_id,
                        shard_id=batch_meta.shard_id,
                        batch_index=batch_meta.batch_index,
                        status=AssignmentStatus.PENDING,
                        assigned_at=datetime.utcnow().isoformat(),
                    )

                    self.assignments[assignment_id] = assignment
                    self.batch_to_worker[batch_id] = worker_id

                self.worker_assignments[worker_id] = worker_assignment

            logger.info(
                f"Assigned {len(batches)} batches to {len(worker_ids)} workers "
                f"using {self.strategy.value} strategy"
            )

            return assignments

    def _assign_shard_per_worker(
        self, worker_ids: List[str], batches: List[BatchMetadata]
    ) -> Dict[str, WorkerAssignment]:
        """Assign complete shards to workers (default federated learning approach)."""
        assignments = {}

        # Group batches by shard
        shard_batches: Dict[int, List[BatchMetadata]] = {}
        for batch in batches:
            shard_id = batch.shard_id
            if shard_id not in shard_batches:
                shard_batches[shard_id] = []
            shard_batches[shard_id].append(batch)

        # Sort batches within each shard
        for shard_id in shard_batches:
            shard_batches[shard_id].sort(key=lambda b: b.batch_index)

        # Assign one shard per worker
        shard_ids = sorted(shard_batches.keys())

        for i, worker_id in enumerate(worker_ids):
            if i >= len(shard_ids):
                logger.warning(
                    f"More workers ({len(worker_ids)}) than shards ({len(shard_ids)}). "
                    f"Worker {worker_id} will not receive data."
                )
                break

            shard_id = shard_ids[i]
            shard_batch_list = shard_batches[shard_id]

            batch_ids = [b.batch_id for b in shard_batch_list]
            total_samples = sum(b.num_samples for b in shard_batch_list)

            assignments[worker_id] = WorkerAssignment(
                worker_id=worker_id,
                shard_id=shard_id,
                assigned_batches=batch_ids,
                total_samples=total_samples,
            )

        return assignments

    def _assign_round_robin(
        self, worker_ids: List[str], batches: List[BatchMetadata]
    ) -> Dict[str, WorkerAssignment]:
        """Distribute batches evenly using round-robin."""
        assignments = {
            worker_id: WorkerAssignment(worker_id=worker_id, shard_id=-1)  # Mixed shards
            for worker_id in worker_ids
        }

        # Sort batches for consistent assignment
        sorted_batches = sorted(batches, key=lambda b: (b.shard_id, b.batch_index))

        # Round-robin assignment
        for i, batch in enumerate(sorted_batches):
            worker_id = worker_ids[i % len(worker_ids)]
            assignments[worker_id].assigned_batches.append(batch.batch_id)
            assignments[worker_id].total_samples += batch.num_samples

        return assignments

    def _assign_load_balanced(
        self, worker_ids: List[str], batches: List[BatchMetadata]
    ) -> Dict[str, WorkerAssignment]:
        """Distribute batches to balance total samples per worker."""
        assignments = {
            worker_id: WorkerAssignment(worker_id=worker_id, shard_id=-1)  # Mixed shards
            for worker_id in worker_ids
        }

        # Sort batches by size (descending) for better balancing
        sorted_batches = sorted(batches, key=lambda b: b.num_samples, reverse=True)

        # Greedy assignment to worker with least samples
        for batch in sorted_batches:
            # Find worker with minimum samples
            min_worker = min(worker_ids, key=lambda w: assignments[w].total_samples)

            assignments[min_worker].assigned_batches.append(batch.batch_id)
            assignments[min_worker].total_samples += batch.num_samples

        return assignments

    def get_worker_assignment(self, worker_id: str) -> Optional[WorkerAssignment]:
        """Get assignment for a specific worker."""
        return self.worker_assignments.get(worker_id)

    def get_batch_assignment(self, batch_id: str) -> Optional[BatchAssignment]:
        """Get assignment for a specific batch."""
        worker_id = self.batch_to_worker.get(batch_id)
        if not worker_id:
            return None

        assignment_id = f"{worker_id}_{batch_id}"
        return self.assignments.get(assignment_id)

    def mark_download_started(self, worker_id: str, batch_id: str) -> bool:
        """
        Mark batch download as started.

        Args:
            worker_id: Worker ID
            batch_id: Batch ID

        Returns:
            True if marked successfully
        """
        with self._lock:
            assignment_id = f"{worker_id}_{batch_id}"
            assignment = self.assignments.get(assignment_id)

            if not assignment:
                logger.warning(f"Assignment not found: {assignment_id}")
                return False

            if assignment.status != AssignmentStatus.PENDING:
                logger.warning(
                    f"Assignment {assignment_id} not in PENDING state: {assignment.status}"
                )
                return False

            assignment.status = AssignmentStatus.DOWNLOADING

            logger.info(f"Download started: {assignment_id}")
            return True

    def mark_download_completed(self, worker_id: str, batch_id: str) -> bool:
        """
        Mark batch download as completed.

        Args:
            worker_id: Worker ID
            batch_id: Batch ID

        Returns:
            True if marked successfully
        """
        with self._lock:
            assignment_id = f"{worker_id}_{batch_id}"
            assignment = self.assignments.get(assignment_id)

            if not assignment:
                logger.warning(f"Assignment not found: {assignment_id}")
                return False

            assignment.status = AssignmentStatus.COMPLETED
            assignment.downloaded_at = datetime.utcnow().isoformat()

            # Update worker assignment
            worker_assignment = self.worker_assignments.get(worker_id)
            if worker_assignment and batch_id not in worker_assignment.completed_batches:
                worker_assignment.completed_batches.append(batch_id)

            logger.info(f"Download completed: {assignment_id}")
            return True

    def mark_download_failed(self, worker_id: str, batch_id: str, reason: str) -> bool:
        """
        Mark batch download as failed.

        Args:
            worker_id: Worker ID
            batch_id: Batch ID
            reason: Failure reason

        Returns:
            True if marked successfully
        """
        with self._lock:
            assignment_id = f"{worker_id}_{batch_id}"
            assignment = self.assignments.get(assignment_id)

            if not assignment:
                logger.warning(f"Assignment not found: {assignment_id}")
                return False

            assignment.status = AssignmentStatus.FAILED
            assignment.failed_at = datetime.utcnow().isoformat()
            assignment.failure_reason = reason
            assignment.retry_count += 1

            # Update worker assignment
            worker_assignment = self.worker_assignments.get(worker_id)
            if worker_assignment and batch_id not in worker_assignment.failed_batches:
                worker_assignment.failed_batches.append(batch_id)

            logger.warning(
                f"Download failed: {assignment_id} (retry {assignment.retry_count}/"
                f"{assignment.max_retries}): {reason}"
            )

            return True

    def reassign_failed_batch(self, batch_id: str, new_worker_id: str) -> Optional[BatchAssignment]:
        """
        Reassign a failed batch to a different worker.

        Args:
            batch_id: Batch ID to reassign
            new_worker_id: New worker ID

        Returns:
            New BatchAssignment if successful
        """
        with self._lock:
            # Find original assignment
            old_worker_id = self.batch_to_worker.get(batch_id)
            if not old_worker_id:
                logger.warning(f"No assignment found for batch {batch_id}")
                return None

            old_assignment_id = f"{old_worker_id}_{batch_id}"
            old_assignment = self.assignments.get(old_assignment_id)

            if not old_assignment or not old_assignment.can_retry():
                logger.warning(f"Cannot reassign batch {batch_id}: max retries exceeded")
                return None

            # Mark old assignment as reassigned
            old_assignment.status = AssignmentStatus.REASSIGNED

            # Create new assignment
            new_assignment_id = f"{new_worker_id}_{batch_id}"
            new_assignment = BatchAssignment(
                assignment_id=new_assignment_id,
                batch_id=batch_id,
                worker_id=new_worker_id,
                shard_id=old_assignment.shard_id,
                batch_index=old_assignment.batch_index,
                status=AssignmentStatus.PENDING,
                assigned_at=datetime.utcnow().isoformat(),
                retry_count=old_assignment.retry_count,
            )

            self.assignments[new_assignment_id] = new_assignment
            self.batch_to_worker[batch_id] = new_worker_id

            # Update worker assignments
            if new_worker_id not in self.worker_assignments:
                self.worker_assignments[new_worker_id] = WorkerAssignment(
                    worker_id=new_worker_id, shard_id=old_assignment.shard_id
                )

            self.worker_assignments[new_worker_id].assigned_batches.append(batch_id)

            logger.info(f"Reassigned batch {batch_id} from {old_worker_id} to {new_worker_id}")

            return new_assignment

    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get distribution statistics."""
        with self._lock:
            total_assignments = len(self.assignments)

            status_counts = {status: 0 for status in AssignmentStatus}
            for assignment in self.assignments.values():
                status_counts[assignment.status] += 1

            worker_stats = {}
            for worker_id, worker_assignment in self.worker_assignments.items():
                worker_stats[worker_id] = {
                    "shard_id": worker_assignment.shard_id,
                    "total_batches": len(worker_assignment.assigned_batches),
                    "completed_batches": len(worker_assignment.completed_batches),
                    "failed_batches": len(worker_assignment.failed_batches),
                    "progress": worker_assignment.get_progress(),
                    "is_complete": worker_assignment.is_complete(),
                    "total_samples": worker_assignment.total_samples,
                }

            return {
                "total_assignments": total_assignments,
                "total_workers": len(self.worker_assignments),
                "status_counts": {k.value: v for k, v in status_counts.items()},
                "worker_stats": worker_stats,
                "strategy": self.strategy.value,
            }

    def get_failed_batches(self) -> List[BatchAssignment]:
        """Get all failed batch assignments."""
        with self._lock:
            return [
                a
                for a in self.assignments.values()
                if a.status == AssignmentStatus.FAILED and a.can_retry()
            ]

    def auto_reassign_failed_batches(self, available_workers: List[str]) -> List[BatchAssignment]:
        """
        Automatically reassign failed batches to available workers.

        Args:
            available_workers: List of available worker IDs

        Returns:
            List of new assignments
        """
        failed_batches = self.get_failed_batches()
        new_assignments = []

        if not failed_batches:
            return new_assignments

        logger.info(f"Auto-reassigning {len(failed_batches)} failed batches")

        # Round-robin reassignment
        for i, failed_assignment in enumerate(failed_batches):
            new_worker = available_workers[i % len(available_workers)]

            # Skip if reassigning to same worker
            if new_worker == failed_assignment.worker_id:
                continue

            new_assignment = self.reassign_failed_batch(
                batch_id=failed_assignment.batch_id, new_worker_id=new_worker
            )

            if new_assignment:
                new_assignments.append(new_assignment)

        logger.info(f"Reassigned {len(new_assignments)} batches")

        return new_assignments
