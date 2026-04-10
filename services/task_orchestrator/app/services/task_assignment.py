"""
Task Assignment Logic Service

High-level orchestration service that implements intelligent task assignment
strategies including batch assignment, load balancing, and resource optimization.

Integrates:
- TASK-6.1: Worker Health Monitoring (WorkerRegistry)
- TASK-6.2: Job Queue Management (JobQueue)
- TASK-6.3: Worker Discovery & Registration (WorkerDiscoveryService)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.dataset_sharder_client import DatasetSharderClient

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class AssignmentStrategy(str, Enum):
    """Task assignment strategies"""

    GREEDY = "greedy"  # Assign to first available worker
    BALANCED = "balanced"  # Distribute evenly across workers
    BEST_FIT = "best_fit"  # Match worker capabilities to job requirements
    COMPUTE_OPTIMIZED = "compute_optimized"  # Prefer highest compute score workers
    COST_OPTIMIZED = "cost_optimized"  # Prefer lower-cost workers
    AFFINITY = "affinity"  # Co-locate related jobs
    ANTI_AFFINITY = "anti_affinity"  # Distribute for fault tolerance


class LoadBalancingPolicy(str, Enum):
    """Load balancing policies"""

    ROUND_ROBIN = "round_robin"  # Rotate through available workers
    LEAST_LOADED = "least_loaded"  # Assign to worker with fewest jobs
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weight by compute score
    RANDOM = "random"  # Random assignment
    PRIORITY_BASED = "priority_based"  # High-priority jobs to best workers


class AssignmentStatus(str, Enum):
    """Assignment operation status"""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    NO_WORKERS_AVAILABLE = "no_workers_available"
    INSUFFICIENT_RESOURCES = "insufficient_resources"


# ==================== Data Classes ====================


@dataclass
class AssignmentConstraints:
    """Constraints for task assignment"""

    require_group: Optional[str] = None
    exclude_workers: Set[str] = field(default_factory=set)
    require_gpu: bool = False
    min_gpu_count: int = 0
    min_ram_gb: float = 0.0
    require_cuda: bool = False
    require_mps: bool = False
    max_jobs_per_worker: int = 10
    affinity_jobs: List[str] = field(default_factory=list)  # Jobs to co-locate
    anti_affinity_jobs: List[str] = field(default_factory=list)  # Jobs to separate
    preferred_workers: List[str] = field(default_factory=list)  # Prefer these workers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "require_group": self.require_group,
            "exclude_workers": list(self.exclude_workers),
            "require_gpu": self.require_gpu,
            "min_gpu_count": self.min_gpu_count,
            "min_ram_gb": self.min_ram_gb,
            "require_cuda": self.require_cuda,
            "require_mps": self.require_mps,
            "max_jobs_per_worker": self.max_jobs_per_worker,
            "affinity_jobs": self.affinity_jobs,
            "anti_affinity_jobs": self.anti_affinity_jobs,
            "preferred_workers": self.preferred_workers,
        }


@dataclass
class AssignmentResult:
    """Result of a task assignment operation"""

    job_id: str
    worker_id: Optional[str]
    status: AssignmentStatus
    assigned_at: Optional[datetime] = None
    shard_ids: List[int] = field(default_factory=list)
    compute_score: float = 0.0
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "status": self.status.value,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "shard_ids": self.shard_ids,
            "compute_score": self.compute_score,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class BatchAssignmentResult:
    """Result of a batch assignment operation"""

    total_jobs: int
    successful: int
    failed: int
    assignments: List[AssignmentResult]
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_jobs == 0:
            return 0.0
        return self.successful / self.total_jobs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs": self.total_jobs,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "assignments": [a.to_dict() for a in self.assignments],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class WorkerLoad:
    """Current load information for a worker"""

    worker_id: str
    assigned_jobs: int
    total_capacity: int
    utilization: float  # 0.0 to 1.0
    compute_score: float
    available_capacity: int

    @property
    def is_available(self) -> bool:
        return self.available_capacity > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "assigned_jobs": self.assigned_jobs,
            "total_capacity": self.total_capacity,
            "utilization": self.utilization,
            "compute_score": self.compute_score,
            "available_capacity": self.available_capacity,
            "is_available": self.is_available,
        }


@dataclass
class AssignmentConfig:
    """Configuration for task assignment service"""

    default_strategy: AssignmentStrategy = AssignmentStrategy.BEST_FIT
    default_load_balancing: LoadBalancingPolicy = LoadBalancingPolicy.LEAST_LOADED
    max_retries: int = 3
    retry_delay_seconds: int = 5
    batch_size: int = 100
    enable_affinity: bool = True
    enable_anti_affinity: bool = True
    max_concurrent_assignments: int = 10
    worker_capacity_threshold: float = 0.8  # 80% utilization threshold
    rebalance_interval_seconds: int = 300  # 5 minutes


# ==================== Task Assignment Service ====================


class TaskAssignmentService:
    """
    High-level orchestration service for intelligent task assignment.

    Provides multiple assignment strategies, load balancing policies,
    and batch assignment capabilities.
    """

    def __init__(
        self,
        worker_discovery,  # WorkerDiscoveryService from TASK-6.3
        job_queue,  # JobQueue from TASK-6.2
        worker_registry,  # WorkerRegistry from TASK-6.1
        config: Optional[AssignmentConfig] = None,
    ):
        self.worker_discovery = worker_discovery
        self.job_queue = job_queue
        self.worker_registry = worker_registry
        self.config = config or AssignmentConfig()

        # Internal state
        self.round_robin_index: Dict[str, int] = {}  # group_id -> index
        self.assignment_history: List[AssignmentResult] = []
        self.rebalance_task: Optional[asyncio.Task] = None
        self._sharded_datasets: Set[str] = set()
        self._shard_lock = asyncio.Lock()

        logger.info(
            f"TaskAssignmentService initialized with strategy={self.config.default_strategy.value}"
        )

    # ==================== Single Job Assignment ====================

    async def assign_job(
        self,
        job_id: str,
        strategy: Optional[AssignmentStrategy] = None,
        constraints: Optional[AssignmentConstraints] = None,
        shard_ids: Optional[List[int]] = None,
    ) -> AssignmentResult:
        """
        Assign a single job to a worker using the specified strategy.

        Args:
            job_id: Job to assign
            strategy: Assignment strategy (defaults to config.default_strategy)
            constraints: Assignment constraints
            shard_ids: Optional shard IDs to assign

        Returns:
            AssignmentResult with assignment details
        """
        strategy = strategy or self.config.default_strategy
        constraints = constraints or AssignmentConstraints()

        logger.info(f"Assigning job {job_id} with strategy {strategy.value}")

        try:
            # Get job from queue
            job = self.job_queue.get_job(job_id)
            if not job:
                return AssignmentResult(
                    job_id=job_id,
                    worker_id=None,
                    status=AssignmentStatus.FAILED,
                    error=f"Job {job_id} not found",
                )

            # Apply constraints from job metadata if not explicitly provided
            if not constraints.require_group and hasattr(job.metadata, "group_id"):
                constraints.require_group = job.metadata.group_id

            # Select worker based on strategy
            worker = await self._select_worker(job, strategy, constraints)

            if not worker:
                return AssignmentResult(
                    job_id=job_id,
                    worker_id=None,
                    status=AssignmentStatus.NO_WORKERS_AVAILABLE,
                    message="No suitable worker found",
                )

            # Ensure dataset is sharded and batches assigned when shard_ids not provided
            if shard_ids is None:
                shard_ids = await self._ensure_shards_and_assign_batches(job, worker.worker_id)

            # Perform assignment
            success = self.worker_discovery.assign_job_to_worker(
                job_id=job_id, worker_id=worker.worker_id, shard_ids=shard_ids
            )

            if success:
                result = AssignmentResult(
                    job_id=job_id,
                    worker_id=worker.worker_id,
                    status=AssignmentStatus.SUCCESS,
                    assigned_at=datetime.utcnow(),
                    shard_ids=shard_ids or [],
                    compute_score=worker.capabilities.get_compute_score(),
                    message=f"Job assigned to {worker.worker_id}",
                )
            else:
                result = AssignmentResult(
                    job_id=job_id,
                    worker_id=None,
                    status=AssignmentStatus.FAILED,
                    error="Assignment failed in worker discovery service",
                )

            self.assignment_history.append(result)
            return result

        except Exception as e:
            logger.error(f"Error assigning job {job_id}: {e}")
            return AssignmentResult(
                job_id=job_id, worker_id=None, status=AssignmentStatus.FAILED, error=str(e)
            )

    async def _ensure_shards_and_assign_batches(self, job, worker_id: str) -> List[int]:
        """Ensure dataset shards exist and assign batches to the worker.

        Returns list of shard IDs assigned to the worker.
        """
        dataset_id = job.metadata.dataset_id

        async with self._shard_lock:
            if dataset_id not in self._sharded_datasets:
                bucket = os.getenv("DATASET_GCS_BUCKET", "meshml-datasets")
                template = os.getenv("DATASET_PATH_TEMPLATE", "gs://{bucket}/{dataset_id}/")
                dataset_path = template.format(bucket=bucket, dataset_id=dataset_id)

                dataset_format = None
                if hasattr(job.metadata, "tags") and isinstance(job.metadata.tags, dict):
                    dataset_format = job.metadata.tags.get("dataset_format")

                num_shards = int(os.getenv("DATASET_DEFAULT_SHARDS", "10"))
                if hasattr(job.metadata, "tags") and isinstance(job.metadata.tags, dict):
                    num_shards = int(job.metadata.tags.get("num_shards", num_shards))

                shard_strategy = "stratified"
                if hasattr(job.metadata, "tags") and isinstance(job.metadata.tags, dict):
                    shard_strategy = job.metadata.tags.get("shard_strategy", shard_strategy)

                batch_size = job.metadata.batch_size

                client = DatasetSharderClient()
                await client.shard_dataset(
                    dataset_id=dataset_id,
                    dataset_path=dataset_path,
                    format=dataset_format,
                    num_shards=num_shards,
                    strategy=shard_strategy,
                    batch_size=batch_size,
                    seed=42,
                )

                self._sharded_datasets.add(dataset_id)

        # Assign batches for this worker
        client = DatasetSharderClient()
        assignment = await client.assign_batches(
            worker_ids=[worker_id], strategy="shard_per_worker"
        )

        shard_ids: List[int] = []
        assignments = assignment.get("assignments", {})
        worker_assignment = assignments.get(worker_id)
        if worker_assignment and "shard_id" in worker_assignment:
            shard_ids = [worker_assignment["shard_id"]]

        return shard_ids

    async def _select_worker(
        self, job, strategy: AssignmentStrategy, constraints: AssignmentConstraints
    ):
        """Select worker based on assignment strategy"""

        # Get available workers
        workers = self.worker_discovery.get_available_workers(
            group_id=constraints.require_group, min_gpu_count=constraints.min_gpu_count
        )

        if not workers:
            return None

        # Apply constraints
        workers = self._apply_constraints(workers, constraints)

        if not workers:
            return None

        # Apply strategy
        if strategy == AssignmentStrategy.GREEDY:
            return workers[0]  # First available

        elif strategy == AssignmentStrategy.BEST_FIT:
            return self.worker_discovery.match_worker_to_job(job.job_id, constraints.require_group)

        elif strategy == AssignmentStrategy.COMPUTE_OPTIMIZED:
            # Sort by compute score descending
            workers_sorted = sorted(
                workers, key=lambda w: w.capabilities.get_compute_score(), reverse=True
            )
            return workers_sorted[0]

        elif strategy == AssignmentStrategy.BALANCED:
            # Use least loaded worker
            return await self._select_least_loaded_worker(workers)

        elif strategy == AssignmentStrategy.AFFINITY:
            return await self._select_affinity_worker(workers, constraints)

        elif strategy == AssignmentStrategy.ANTI_AFFINITY:
            return await self._select_anti_affinity_worker(workers, constraints)

        else:
            # Default to first available
            return workers[0]

    def _apply_constraints(self, workers, constraints: AssignmentConstraints):
        """Apply assignment constraints to filter workers"""
        filtered = workers

        # Exclude specific workers
        if constraints.exclude_workers:
            filtered = [w for w in filtered if w.worker_id not in constraints.exclude_workers]

        # Prefer specific workers (move to front)
        if constraints.preferred_workers:
            preferred = [w for w in filtered if w.worker_id in constraints.preferred_workers]
            others = [w for w in filtered if w.worker_id not in constraints.preferred_workers]
            filtered = preferred + others

        # Check worker capacity
        filtered = [
            w
            for w in filtered
            if self._get_worker_assigned_jobs(w.worker_id) < constraints.max_jobs_per_worker
        ]

        return filtered

    async def _select_least_loaded_worker(self, workers):
        """Select worker with least load"""
        worker_loads = []

        for worker in workers:
            load = await self.get_worker_load(worker.worker_id)
            worker_loads.append((worker, load))

        # Sort by utilization ascending
        worker_loads.sort(key=lambda x: x[1].utilization)

        if worker_loads:
            return worker_loads[0][0]
        return None

    async def _select_affinity_worker(self, workers, constraints: AssignmentConstraints):
        """Select worker that already runs affinity jobs (co-location)"""
        if not constraints.affinity_jobs:
            return workers[0]

        # Find workers running affinity jobs
        for worker in workers:
            assigned_jobs = self._get_worker_jobs(worker.worker_id)
            if any(job_id in constraints.affinity_jobs for job_id in assigned_jobs):
                return worker

        # No affinity match, use first available
        return workers[0]

    async def _select_anti_affinity_worker(self, workers, constraints: AssignmentConstraints):
        """Select worker that doesn't run anti-affinity jobs (fault tolerance)"""
        if not constraints.anti_affinity_jobs:
            return workers[0]

        # Filter out workers running anti-affinity jobs
        available = []
        for worker in workers:
            assigned_jobs = self._get_worker_jobs(worker.worker_id)
            if not any(job_id in constraints.anti_affinity_jobs for job_id in assigned_jobs):
                available.append(worker)

        return available[0] if available else workers[0]

    # ==================== Batch Assignment ====================

    async def assign_batch(
        self,
        job_ids: List[str],
        strategy: Optional[AssignmentStrategy] = None,
        load_balancing: Optional[LoadBalancingPolicy] = None,
        constraints: Optional[AssignmentConstraints] = None,
    ) -> BatchAssignmentResult:
        """
        Assign multiple jobs to workers using batch optimization.

        Args:
            job_ids: List of job IDs to assign
            strategy: Assignment strategy
            load_balancing: Load balancing policy
            constraints: Assignment constraints

        Returns:
            BatchAssignmentResult with details for all assignments
        """
        started_at = datetime.utcnow()
        strategy = strategy or self.config.default_strategy
        load_balancing = load_balancing or self.config.default_load_balancing
        constraints = constraints or AssignmentConstraints()

        logger.info(
            f"Batch assigning {len(job_ids)} jobs with strategy={strategy.value}, load_balancing={load_balancing.value}"
        )

        assignments = []
        successful = 0
        failed = 0

        # Apply load balancing policy
        if load_balancing == LoadBalancingPolicy.ROUND_ROBIN:
            assignments = await self._assign_batch_round_robin(job_ids, strategy, constraints)

        elif load_balancing == LoadBalancingPolicy.LEAST_LOADED:
            assignments = await self._assign_batch_least_loaded(job_ids, strategy, constraints)

        elif load_balancing == LoadBalancingPolicy.WEIGHTED_ROUND_ROBIN:
            assignments = await self._assign_batch_weighted_round_robin(
                job_ids, strategy, constraints
            )

        elif load_balancing == LoadBalancingPolicy.PRIORITY_BASED:
            assignments = await self._assign_batch_priority(job_ids, strategy, constraints)

        else:
            # Default: assign sequentially
            for job_id in job_ids:
                result = await self.assign_job(job_id, strategy, constraints)
                assignments.append(result)

        # Count results
        for assignment in assignments:
            if assignment.status == AssignmentStatus.SUCCESS:
                successful += 1
            else:
                failed += 1

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        return BatchAssignmentResult(
            total_jobs=len(job_ids),
            successful=successful,
            failed=failed,
            assignments=assignments,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )

    async def _assign_batch_round_robin(
        self, job_ids: List[str], strategy: AssignmentStrategy, constraints: AssignmentConstraints
    ) -> List[AssignmentResult]:
        """Assign jobs in round-robin fashion across workers"""
        group_id = constraints.require_group or "default"

        # Get available workers
        workers = self.worker_discovery.get_available_workers(
            group_id=constraints.require_group, min_gpu_count=constraints.min_gpu_count
        )

        if not workers:
            return [
                AssignmentResult(
                    job_id=job_id,
                    worker_id=None,
                    status=AssignmentStatus.NO_WORKERS_AVAILABLE,
                    message="No workers available",
                )
                for job_id in job_ids
            ]

        workers = self._apply_constraints(workers, constraints)

        # Initialize round-robin index
        if group_id not in self.round_robin_index:
            self.round_robin_index[group_id] = 0

        assignments = []
        for job_id in job_ids:
            # Get next worker in round-robin
            worker_idx = self.round_robin_index[group_id] % len(workers)
            worker = workers[worker_idx]
            self.round_robin_index[group_id] += 1

            # Ensure dataset is sharded and batches assigned
            job = self.job_queue.get_job(job_id)
            shard_ids: Optional[List[int]] = None
            if job:
                shard_ids = await self._ensure_shards_and_assign_batches(job, worker.worker_id)

            # Assign job
            success = self.worker_discovery.assign_job_to_worker(
                job_id=job_id, worker_id=worker.worker_id, shard_ids=shard_ids
            )

            if success:
                result = AssignmentResult(
                    job_id=job_id,
                    worker_id=worker.worker_id,
                    status=AssignmentStatus.SUCCESS,
                    assigned_at=datetime.utcnow(),
                    shard_ids=shard_ids or [],
                    compute_score=worker.capabilities.get_compute_score(),
                )
            else:
                result = AssignmentResult(
                    job_id=job_id,
                    worker_id=None,
                    status=AssignmentStatus.FAILED,
                    error="Assignment failed",
                )

            assignments.append(result)

        return assignments

    async def _assign_batch_least_loaded(
        self, job_ids: List[str], strategy: AssignmentStrategy, constraints: AssignmentConstraints
    ) -> List[AssignmentResult]:
        """Assign jobs to least loaded workers"""
        assignments = []

        for job_id in job_ids:
            # Dynamically select least loaded worker for each job
            result = await self.assign_job(job_id, AssignmentStrategy.BALANCED, constraints)
            assignments.append(result)

        return assignments

    async def _assign_batch_weighted_round_robin(
        self, job_ids: List[str], strategy: AssignmentStrategy, constraints: AssignmentConstraints
    ) -> List[AssignmentResult]:
        """Assign jobs using weighted round-robin based on compute score"""
        workers = self.worker_discovery.get_available_workers(
            group_id=constraints.require_group, min_gpu_count=constraints.min_gpu_count
        )

        if not workers:
            return [
                AssignmentResult(
                    job_id=job_id, worker_id=None, status=AssignmentStatus.NO_WORKERS_AVAILABLE
                )
                for job_id in job_ids
            ]

        workers = self._apply_constraints(workers, constraints)

        # Calculate weights based on compute score
        total_score = sum(w.capabilities.get_compute_score() for w in workers)
        worker_weights = [(w, w.capabilities.get_compute_score() / total_score) for w in workers]

        assignments = []
        weight_accumulator = 0.0
        worker_idx = 0

        for i, job_id in enumerate(job_ids):
            # Select worker based on weight
            weight_accumulator += worker_weights[worker_idx][1]

            if weight_accumulator >= 1.0 or i == len(job_ids) - 1:
                weight_accumulator = 0.0
                worker_idx = (worker_idx + 1) % len(worker_weights)

            worker = worker_weights[worker_idx][0]

            job = self.job_queue.get_job(job_id)
            shard_ids: Optional[List[int]] = None
            if job:
                shard_ids = await self._ensure_shards_and_assign_batches(job, worker.worker_id)

            # Assign job
            success = self.worker_discovery.assign_job_to_worker(
                job_id=job_id, worker_id=worker.worker_id, shard_ids=shard_ids
            )

            result = AssignmentResult(
                job_id=job_id,
                worker_id=worker.worker_id if success else None,
                status=AssignmentStatus.SUCCESS if success else AssignmentStatus.FAILED,
                assigned_at=datetime.utcnow() if success else None,
                shard_ids=shard_ids or [],
                compute_score=worker.capabilities.get_compute_score(),
            )
            assignments.append(result)

        return assignments

    async def _assign_batch_priority(
        self, job_ids: List[str], strategy: AssignmentStrategy, constraints: AssignmentConstraints
    ) -> List[AssignmentResult]:
        """Assign high-priority jobs to best workers"""
        # Get jobs with priorities
        jobs_with_priority = []
        for job_id in job_ids:
            job = self.job_queue.get_job(job_id)
            if job:
                priority = getattr(job, "priority", 0)
                jobs_with_priority.append((job_id, priority))

        # Sort by priority descending
        jobs_with_priority.sort(key=lambda x: x[1], reverse=True)

        # Get workers sorted by compute score
        workers = self.worker_discovery.get_available_workers(
            group_id=constraints.require_group, min_gpu_count=constraints.min_gpu_count
        )

        if not workers:
            return [
                AssignmentResult(
                    job_id=job_id, worker_id=None, status=AssignmentStatus.NO_WORKERS_AVAILABLE
                )
                for job_id, _ in jobs_with_priority
            ]

        workers = self._apply_constraints(workers, constraints)
        workers_sorted = sorted(
            workers, key=lambda w: w.capabilities.get_compute_score(), reverse=True
        )

        assignments = []
        worker_idx = 0

        for job_id, priority in jobs_with_priority:
            # High priority jobs get best workers
            worker = workers_sorted[worker_idx % len(workers_sorted)]
            worker_idx += 1

            job = self.job_queue.get_job(job_id)
            shard_ids: Optional[List[int]] = None
            if job:
                shard_ids = await self._ensure_shards_and_assign_batches(job, worker.worker_id)

            success = self.worker_discovery.assign_job_to_worker(
                job_id=job_id, worker_id=worker.worker_id, shard_ids=shard_ids
            )

            result = AssignmentResult(
                job_id=job_id,
                worker_id=worker.worker_id if success else None,
                status=AssignmentStatus.SUCCESS if success else AssignmentStatus.FAILED,
                assigned_at=datetime.utcnow() if success else None,
                shard_ids=shard_ids or [],
                compute_score=worker.capabilities.get_compute_score(),
            )
            assignments.append(result)

        return assignments

    # ==================== Load Monitoring ====================

    async def get_worker_load(self, worker_id: str) -> WorkerLoad:
        """Get current load for a worker"""
        assigned_jobs = self._get_worker_assigned_jobs(worker_id)
        total_capacity = self.config.batch_size  # Simplified capacity model
        utilization = assigned_jobs / total_capacity if total_capacity > 0 else 0.0

        # Get worker info for compute score
        workers = self.worker_discovery.discover_workers()
        worker_info = next((w for w in workers if w.worker_id == worker_id), None)
        compute_score = worker_info.capabilities.get_compute_score() if worker_info else 0.0

        return WorkerLoad(
            worker_id=worker_id,
            assigned_jobs=assigned_jobs,
            total_capacity=total_capacity,
            utilization=utilization,
            compute_score=compute_score,
            available_capacity=max(0, total_capacity - assigned_jobs),
        )

    async def get_cluster_load(self, group_id: Optional[str] = None) -> Dict[str, Any]:
        """Get load statistics for entire cluster or group"""
        workers = self.worker_discovery.discover_workers(group_id=group_id)

        total_workers = len(workers)
        total_jobs = 0
        worker_loads = []

        for worker in workers:
            load = await self.get_worker_load(worker.worker_id)
            worker_loads.append(load)
            total_jobs += load.assigned_jobs

        avg_utilization = (
            sum(load.utilization for load in worker_loads) / len(worker_loads)
            if worker_loads
            else 0.0
        )

        return {
            "group_id": group_id,
            "total_workers": total_workers,
            "total_jobs": total_jobs,
            "avg_utilization": avg_utilization,
            "worker_loads": [load.to_dict() for load in worker_loads],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_worker_assigned_jobs(self, worker_id: str) -> int:
        """Get number of jobs assigned to worker"""
        # Query job queue for jobs assigned to this worker
        jobs = self.job_queue.list_jobs(worker_id=worker_id, status="running")
        return len(jobs) if jobs else 0

    def _get_worker_jobs(self, worker_id: str) -> List[str]:
        """Get list of job IDs assigned to worker"""
        jobs = self.job_queue.list_jobs(worker_id=worker_id, status="running")
        return [job.job_id for job in jobs] if jobs else []

    # ==================== Load Rebalancing ====================

    async def rebalance_load(
        self, group_id: Optional[str] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rebalance load across workers by reassigning jobs.

        Args:
            group_id: Optional group to rebalance
            threshold: Utilization threshold (default: config.worker_capacity_threshold)

        Returns:
            Dict with rebalancing statistics
        """
        threshold = threshold or self.config.worker_capacity_threshold

        logger.info(f"Rebalancing load for group={group_id}, threshold={threshold}")

        # Get worker loads
        workers = self.worker_discovery.discover_workers(group_id=group_id)
        worker_loads = []

        for worker in workers:
            load = await self.get_worker_load(worker.worker_id)
            worker_loads.append(load)

        # Separate overloaded and underutilized workers
        overloaded = [load for load in worker_loads if load.utilization > threshold]
        underutilized = [load for load in worker_loads if load.utilization < 0.5]

        reassigned_jobs = 0

        # Reassign jobs from overloaded to underutilized workers
        for overloaded_worker in overloaded:
            if not underutilized:
                break

            # Get jobs to move
            jobs_to_move = int(
                (overloaded_worker.utilization - threshold) * overloaded_worker.total_capacity
            )
            jobs = self._get_worker_jobs(overloaded_worker.worker_id)[:jobs_to_move]

            for job_id in jobs:
                if not underutilized:
                    break

                # Select underutilized worker
                target_worker = underutilized[0]

                # Reassign job
                # 1. Release from current worker
                self.job_queue.release_job_from_worker(
                    job_id, overloaded_worker.worker_id, "Load rebalancing"
                )

                # 2. Assign to new worker
                success = self.worker_discovery.assign_job_to_worker(
                    job_id=job_id, worker_id=target_worker.worker_id
                )

                if success:
                    reassigned_jobs += 1

                    # Update target worker load
                    target_worker.assigned_jobs += 1
                    target_worker.utilization = (
                        target_worker.assigned_jobs / target_worker.total_capacity
                    )

                    # Remove from underutilized if now balanced
                    if target_worker.utilization >= 0.5:
                        underutilized.pop(0)

        return {
            "group_id": group_id,
            "reassigned_jobs": reassigned_jobs,
            "overloaded_workers": len(overloaded),
            "underutilized_workers": len(underutilized),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def start_auto_rebalancing(self):
        """Start automatic load rebalancing task"""
        if self.rebalance_task:
            logger.warning("Auto-rebalancing already running")
            return

        async def rebalance_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.rebalance_interval_seconds)
                    await self.rebalance_load()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-rebalancing: {e}")

        self.rebalance_task = asyncio.create_task(rebalance_loop())
        logger.info("Auto-rebalancing started")

    async def stop_auto_rebalancing(self):
        """Stop automatic load rebalancing"""
        if self.rebalance_task:
            self.rebalance_task.cancel()
            try:
                await self.rebalance_task
            except asyncio.CancelledError:
                pass
            self.rebalance_task = None
            logger.info("Auto-rebalancing stopped")

    # ==================== Statistics ====================

    def get_assignment_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get assignment statistics for the last N hours"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)

        recent_assignments = [
            a
            for a in self.assignment_history
            if a.assigned_at and a.assigned_at.timestamp() > cutoff
        ]

        total = len(recent_assignments)
        successful = sum(1 for a in recent_assignments if a.status == AssignmentStatus.SUCCESS)
        failed = sum(1 for a in recent_assignments if a.status == AssignmentStatus.FAILED)

        return {
            "hours": hours,
            "total_assignments": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "timestamp": datetime.utcnow().isoformat(),
        }
