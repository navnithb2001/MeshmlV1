"""
Worker Discovery & Registration Orchestration Service

Integrates worker registry (TASK-6.1) with job queue (TASK-6.2) to provide:
- Automatic worker discovery and registration
- Worker pool management with group-based access control
- Intelligent job-to-worker matching
- Worker capability tracking and reporting
- Dynamic worker scaling and load balancing

This service acts as the orchestration layer between workers, jobs, and resources.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status (mirrors TASK-6.1)"""

    ONLINE = "online"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class WorkerPoolStatus(Enum):
    """Worker pool health status"""

    HEALTHY = "healthy"  # Sufficient workers available
    DEGRADED = "degraded"  # Some workers offline but operational
    CRITICAL = "critical"  # Insufficient workers for demand
    OFFLINE = "offline"  # No workers available


@dataclass
class WorkerCapabilities:
    """
    Worker hardware and software capabilities.
    Matches TASK-6.1 WorkerCapabilities for consistency.
    """

    gpu_count: int
    gpu_memory_gb: float
    gpu_type: str
    cpu_count: int
    ram_gb: float
    network_speed_mbps: float
    storage_gb: float
    supports_cuda: bool
    supports_mps: bool
    pytorch_version: str
    python_version: str

    def get_compute_score(self) -> float:
        """Calculate compute capability score (same as TASK-6.1)"""
        gpu_score = self.gpu_count * self.gpu_memory_gb * 10
        cpu_score = self.cpu_count * 0.5
        ram_score = self.ram_gb * 0.2
        return gpu_score + cpu_score + ram_score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerCapabilities":
        return cls(**data)


@dataclass
class WorkerInfo:
    """
    Worker information summary.
    Simplified version of TASK-6.1 WorkerInfo for orchestration.
    """

    worker_id: str
    hostname: str
    ip_address: str
    port: int
    status: WorkerStatus
    capabilities: WorkerCapabilities
    group_id: Optional[str] = None
    assigned_job_id: Optional[str] = None
    assigned_shard_ids: List[int] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = self.capabilities.to_dict()
        data["status"] = self.status.value
        return data


@dataclass
class WorkerPool:
    """
    Worker pool for a specific group.
    Manages workers belonging to a group with access control.
    """

    group_id: str
    name: str
    description: str = ""

    # Worker tracking
    worker_ids: Set[str] = field(default_factory=set)

    # Pool configuration
    min_workers: int = 1
    max_workers: int = 100
    auto_scale: bool = False

    # Pool metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)

    def get_worker_count(self) -> int:
        """Get current number of workers in pool"""
        return len(self.worker_ids)

    def is_at_capacity(self) -> bool:
        """Check if pool is at max capacity"""
        return len(self.worker_ids) >= self.max_workers

    def needs_scaling(self) -> bool:
        """Check if pool needs more workers"""
        return self.auto_scale and len(self.worker_ids) < self.min_workers

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["worker_ids"] = list(self.worker_ids)
        return data


@dataclass
class DiscoveryConfig:
    """Worker discovery configuration"""

    heartbeat_timeout_seconds: int = 30
    discovery_interval_seconds: int = 60
    auto_register_workers: bool = True
    require_group_assignment: bool = True
    enable_auto_scaling: bool = False
    max_workers_per_group: int = 100


class WorkerDiscoveryService:
    """
    Worker Discovery & Registration Orchestration Service.

    Coordinates between:
    - Worker Registry (TASK-6.1): Worker health and lifecycle
    - Job Queue (TASK-6.2): Job scheduling and assignment
    - Worker Pools: Group-based access control

    Responsibilities:
    1. Worker registration and discovery
    2. Worker pool management
    3. Group-based access control
    4. Capability-based job matching
    5. Worker availability tracking
    """

    def __init__(
        self,
        worker_registry,  # TASK-6.1 WorkerRegistry instance
        job_queue,  # TASK-6.2 JobQueue instance
        config: Optional[DiscoveryConfig] = None,
    ):
        self.worker_registry = worker_registry
        self.job_queue = job_queue
        self.config = config or DiscoveryConfig()

        # Worker pools (group_id → WorkerPool)
        self.pools: Dict[str, WorkerPool] = {}

        # Worker to pool mapping (worker_id → group_id)
        self.worker_to_pool: Dict[str, str] = {}

        # Discovery state
        self._running = False
        self._discovery_task: Optional[asyncio.Task] = None

    # ==================== Worker Pool Management ====================

    def create_pool(
        self,
        group_id: str,
        name: str,
        description: str = "",
        min_workers: int = 1,
        max_workers: int = 100,
        auto_scale: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> WorkerPool:
        """
        Create new worker pool for a group.

        Args:
            group_id: Unique group identifier
            name: Pool name
            description: Pool description
            min_workers: Minimum workers to maintain
            max_workers: Maximum workers allowed
            auto_scale: Enable auto-scaling
            tags: Custom tags

        Returns:
            WorkerPool instance
        """
        if group_id in self.pools:
            logger.warning(f"Pool for group {group_id} already exists")
            return self.pools[group_id]

        pool = WorkerPool(
            group_id=group_id,
            name=name,
            description=description,
            min_workers=min_workers,
            max_workers=max_workers,
            auto_scale=auto_scale,
            tags=tags or {},
        )

        self.pools[group_id] = pool
        logger.info(f"Created worker pool for group {group_id}: {name}")
        return pool

    def get_pool(self, group_id: str) -> Optional[WorkerPool]:
        """Get worker pool by group ID"""
        return self.pools.get(group_id)

    def list_pools(self) -> List[WorkerPool]:
        """List all worker pools"""
        return list(self.pools.values())

    def delete_pool(self, group_id: str, force: bool = False) -> bool:
        """
        Delete worker pool.

        Args:
            group_id: Group ID
            force: Force delete even if workers are assigned

        Returns:
            Success boolean
        """
        pool = self.pools.get(group_id)
        if not pool:
            return False

        if len(pool.worker_ids) > 0 and not force:
            logger.warning(f"Cannot delete pool {group_id} with active workers. Use force=True")
            return False

        # Remove worker mappings
        for worker_id in list(pool.worker_ids):
            self.worker_to_pool.pop(worker_id, None)

        del self.pools[group_id]
        logger.info(f"Deleted worker pool for group {group_id}")
        return True

    # ==================== Worker Registration ====================

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        ip_address: str,
        port: int,
        capabilities: WorkerCapabilities,
        group_id: Optional[str] = None,
        version: str = "1.0.0",
        tags: Optional[Dict[str, str]] = None,
    ) -> WorkerInfo:
        """
        Register worker with discovery service.

        Workflow:
        1. Validate group assignment (if required)
        2. Register with worker registry (TASK-6.1)
        3. Add to worker pool
        4. Update worker-to-pool mapping
        5. Return WorkerInfo

        Args:
            worker_id: Unique worker identifier
            hostname: Worker hostname
            ip_address: Worker IP address
            port: Worker port
            capabilities: Worker capabilities
            group_id: Group assignment (optional)
            version: Worker software version
            tags: Custom tags

        Returns:
            WorkerInfo instance
        """
        # Validate group requirement
        if self.config.require_group_assignment and not group_id:
            raise ValueError("Group assignment required but not provided")

        # Create pool if doesn't exist
        if group_id and group_id not in self.pools:
            self.create_pool(
                group_id=group_id, name=f"Pool for {group_id}", description="Auto-created pool"
            )

        # Check pool capacity
        if group_id:
            pool = self.pools[group_id]
            if pool.is_at_capacity():
                raise ValueError(
                    f"Pool {group_id} is at maximum capacity ({pool.max_workers} workers)"
                )

        # Register with worker registry (TASK-6.1)
        # Note: This assumes worker_registry has register_worker method from TASK-6.1
        registry_worker = self.worker_registry.register_worker(
            worker_id=worker_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            capabilities=capabilities.__dict__,  # Convert to dict for registry
            group_id=group_id,
            version=version,
            tags=tags or {},
        )

        # Add to pool
        if group_id:
            self.pools[group_id].worker_ids.add(worker_id)
            self.worker_to_pool[worker_id] = group_id

        # Convert registry worker to WorkerInfo
        worker_info = WorkerInfo(
            worker_id=worker_id,
            hostname=hostname,
            ip_address=ip_address,
            port=port,
            status=WorkerStatus.IDLE,
            capabilities=capabilities,
            group_id=group_id,
            registered_at=registry_worker.registered_at,
            last_heartbeat=registry_worker.last_heartbeat,
        )

        logger.info(f"Registered worker {worker_id} in group {group_id}")
        return worker_info

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister worker from discovery service.

        Workflow:
        1. Remove from worker pool
        2. Remove from worker-to-pool mapping
        3. Remove from worker registry (TASK-6.1)

        Returns:
            Success boolean
        """
        # Remove from pool
        group_id = self.worker_to_pool.get(worker_id)
        if group_id and group_id in self.pools:
            self.pools[group_id].worker_ids.discard(worker_id)

        # Remove mapping
        self.worker_to_pool.pop(worker_id, None)

        # Remove from registry
        success = self.worker_registry.remove_worker(worker_id)

        if success:
            logger.info(f"Unregistered worker {worker_id}")
        return success

    # ==================== Worker Discovery ====================

    def discover_workers(
        self,
        group_id: Optional[str] = None,
        min_gpu_count: int = 0,
        status_filter: Optional[List[WorkerStatus]] = None,
    ) -> List[WorkerInfo]:
        """
        Discover available workers.

        Args:
            group_id: Filter by group
            min_gpu_count: Minimum GPU count
            status_filter: Filter by worker status

        Returns:
            List of WorkerInfo matching criteria
        """
        # Get workers from registry
        registry_workers = self.worker_registry.list_workers(
            group_id=group_id, min_gpu_count=min_gpu_count
        )

        # Convert to WorkerInfo and apply status filter
        workers = []
        for rw in registry_workers:
            # Parse status
            try:
                status = WorkerStatus(rw.status.value if hasattr(rw.status, "value") else rw.status)
            except ValueError:
                status = WorkerStatus.UNKNOWN

            # Apply status filter
            if status_filter and status not in status_filter:
                continue

            # Convert capabilities
            caps = WorkerCapabilities.from_dict(rw.capabilities.to_dict())

            worker_info = WorkerInfo(
                worker_id=rw.worker_id,
                hostname=rw.hostname,
                ip_address=rw.ip_address,
                port=rw.port,
                status=status,
                capabilities=caps,
                group_id=rw.group_id,
                assigned_job_id=rw.assigned_job_id,
                assigned_shard_ids=rw.assigned_shard_ids,
                registered_at=rw.registered_at,
                last_heartbeat=rw.last_heartbeat,
            )
            workers.append(worker_info)

        return workers

    def get_available_workers(
        self, group_id: Optional[str] = None, min_gpu_count: int = 0
    ) -> List[WorkerInfo]:
        """
        Get workers available for job assignment.

        Returns workers with IDLE or ONLINE status, sorted by compute capability.
        """
        return self.discover_workers(
            group_id=group_id,
            min_gpu_count=min_gpu_count,
            status_filter=[WorkerStatus.IDLE, WorkerStatus.ONLINE],
        )

    # ==================== Worker-Job Matching ====================

    def match_worker_to_job(
        self, job_id: str, group_id: Optional[str] = None
    ) -> Optional[WorkerInfo]:
        """
        Find best worker for a job.

        Algorithm:
        1. Get job from queue (TASK-6.2)
        2. Get available workers (optionally filtered by group)
        3. Filter workers by job requirements
        4. Sort by compute score (highest first)
        5. Return best match

        Args:
            job_id: Job identifier
            group_id: Restrict to workers in this group

        Returns:
            Best matching WorkerInfo or None
        """
        # Get job details
        job = self.job_queue.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return None

        # Use job's group if not specified
        if not group_id:
            group_id = job.metadata.group_id

        # Get available workers
        available_workers = self.get_available_workers(group_id=group_id)

        if not available_workers:
            logger.warning(f"No available workers for group {group_id}")
            return None

        # Filter by job requirements
        job_req = job.metadata.requirements
        matching_workers = []

        for worker in available_workers:
            caps = worker.capabilities

            # Check if worker meets requirements
            if (
                caps.gpu_count >= job_req.min_gpu_count
                and caps.gpu_memory_gb >= job_req.min_gpu_memory_gb
                and caps.cpu_count >= job_req.min_cpu_count
                and caps.ram_gb >= job_req.min_ram_gb
                and (not job_req.requires_cuda or caps.supports_cuda)
                and (not job_req.requires_mps or caps.supports_mps)
            ):

                matching_workers.append(worker)

        if not matching_workers:
            logger.warning(f"No workers match requirements for job {job_id}")
            return None

        # Sort by compute score (best first)
        matching_workers.sort(key=lambda w: w.capabilities.get_compute_score(), reverse=True)

        best_worker = matching_workers[0]
        logger.info(
            f"Matched job {job_id} to worker {best_worker.worker_id} "
            f"(score: {best_worker.capabilities.get_compute_score():.2f})"
        )

        return best_worker

    def assign_job_to_worker(
        self, job_id: str, worker_id: Optional[str] = None, shard_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Assign job to worker.

        Workflow:
        1. If worker_id not provided, find best match
        2. Assign job in job queue (TASK-6.2)
        3. Assign job in worker registry (TASK-6.1)
        4. Return success

        Args:
            job_id: Job identifier
            worker_id: Worker identifier (auto-match if None)
            shard_ids: Data shard IDs to assign

        Returns:
            Success boolean
        """
        # Auto-match worker if not specified
        if not worker_id:
            worker = self.match_worker_to_job(job_id)
            if not worker:
                logger.error(f"Failed to find matching worker for job {job_id}")
                return False
            worker_id = worker.worker_id

        # Assign in job queue (TASK-6.2)
        job_success = self.job_queue.assign_job_to_worker(
            job_id=job_id, worker_id=worker_id, shard_ids=shard_ids
        )

        if not job_success:
            logger.error(f"Failed to assign job {job_id} in job queue")
            return False

        # Assign in worker registry (TASK-6.1)
        worker_success = self.worker_registry.assign_job(
            worker_id=worker_id, job_id=job_id, shard_id=shard_ids[0] if shard_ids else None
        )

        if not worker_success:
            logger.error(f"Failed to assign job {job_id} to worker {worker_id} in registry")
            # Rollback job queue assignment
            self.job_queue.release_job_from_worker(job_id, worker_id, "registry_assignment_failed")
            return False

        logger.info(f"Successfully assigned job {job_id} to worker {worker_id}")
        return True

    # ==================== Pool Health Monitoring ====================

    def get_pool_status(self, group_id: str) -> WorkerPoolStatus:
        """
        Get pool health status.

        Algorithm:
        1. Count workers by status
        2. Determine pool health:
           - HEALTHY: >= 80% workers IDLE/ONLINE
           - DEGRADED: 50-79% workers IDLE/ONLINE
           - CRITICAL: 20-49% workers IDLE/ONLINE
           - OFFLINE: < 20% workers IDLE/ONLINE

        Returns:
            WorkerPoolStatus
        """
        pool = self.pools.get(group_id)
        if not pool or len(pool.worker_ids) == 0:
            return WorkerPoolStatus.OFFLINE

        # Count workers by status
        workers = self.discover_workers(group_id=group_id)
        total = len(workers)

        if total == 0:
            return WorkerPoolStatus.OFFLINE

        healthy_count = sum(
            1 for w in workers if w.status in [WorkerStatus.IDLE, WorkerStatus.ONLINE]
        )

        healthy_ratio = healthy_count / total

        if healthy_ratio >= 0.8:
            return WorkerPoolStatus.HEALTHY
        elif healthy_ratio >= 0.5:
            return WorkerPoolStatus.DEGRADED
        elif healthy_ratio >= 0.2:
            return WorkerPoolStatus.CRITICAL
        else:
            return WorkerPoolStatus.OFFLINE

    def get_pool_stats(self, group_id: str) -> Dict[str, Any]:
        """
        Get detailed pool statistics.

        Returns:
            Dictionary with pool stats
        """
        pool = self.pools.get(group_id)
        if not pool:
            return {}

        workers = self.discover_workers(group_id=group_id)

        # Count by status
        status_counts = {}
        for status in WorkerStatus:
            status_counts[status.value] = sum(1 for w in workers if w.status == status)

        # Calculate total resources
        total_gpus = sum(w.capabilities.gpu_count for w in workers)
        total_ram_gb = sum(w.capabilities.ram_gb for w in workers)
        total_storage_gb = sum(w.capabilities.storage_gb for w in workers)

        # Calculate average compute score
        avg_compute_score = (
            sum(w.capabilities.get_compute_score() for w in workers) / len(workers)
            if workers
            else 0
        )

        return {
            "group_id": group_id,
            "pool_name": pool.name,
            "total_workers": len(workers),
            "status_counts": status_counts,
            "available_workers": status_counts.get("idle", 0) + status_counts.get("online", 0),
            "busy_workers": status_counts.get("busy", 0),
            "offline_workers": status_counts.get("offline", 0),
            "total_gpus": total_gpus,
            "total_ram_gb": total_ram_gb,
            "total_storage_gb": total_storage_gb,
            "avg_compute_score": avg_compute_score,
            "pool_status": self.get_pool_status(group_id).value,
            "min_workers": pool.min_workers,
            "max_workers": pool.max_workers,
            "auto_scale": pool.auto_scale,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ==================== Auto-Scaling ====================

    def check_scaling_needs(self) -> Dict[str, str]:
        """
        Check which pools need scaling.

        Returns:
            Dictionary mapping group_id to scaling action ("scale_up" or "scale_down")
        """
        scaling_needs = {}

        for group_id, pool in self.pools.items():
            if not pool.auto_scale:
                continue

            workers = self.discover_workers(group_id=group_id)
            active_workers = [
                w
                for w in workers
                if w.status in [WorkerStatus.IDLE, WorkerStatus.ONLINE, WorkerStatus.BUSY]
            ]

            active_count = len(active_workers)

            # Need to scale up
            if active_count < pool.min_workers:
                scaling_needs[group_id] = "scale_up"
                logger.info(
                    f"Pool {group_id} needs scaling up: {active_count}/{pool.min_workers} workers"
                )

            # Could scale down (if > 2x min_workers and < 50% utilization)
            elif active_count > pool.min_workers * 2:
                busy_workers = [w for w in active_workers if w.status == WorkerStatus.BUSY]
                utilization = len(busy_workers) / active_count if active_count > 0 else 0

                if utilization < 0.5:
                    scaling_needs[group_id] = "scale_down"
                    logger.info(f"Pool {group_id} could scale down: {utilization:.1%} utilization")

        return scaling_needs

    # ==================== Background Discovery ====================

    async def start_discovery(self):
        """
        Start background worker discovery.

        Periodically:
        1. Check worker health (via TASK-6.1 registry)
        2. Update pool statistics
        3. Check scaling needs
        4. Log pool health
        """
        self._running = True
        logger.info("Started worker discovery service")

        while self._running:
            try:
                # Check each pool
                for group_id in list(self.pools.keys()):
                    status = self.get_pool_status(group_id)
                    stats = self.get_pool_stats(group_id)

                    if status in [WorkerPoolStatus.CRITICAL, WorkerPoolStatus.OFFLINE]:
                        logger.warning(
                            f"Pool {group_id} is {status.value}: "
                            f"{stats['available_workers']}/{stats['total_workers']} workers available"
                        )

                # Check scaling needs
                if self.config.enable_auto_scaling:
                    scaling_needs = self.check_scaling_needs()
                    if scaling_needs:
                        logger.info(f"Scaling needs detected: {scaling_needs}")
                        # Note: Actual scaling would be implemented in TASK-6.5

                await asyncio.sleep(self.config.discovery_interval_seconds)

            except Exception as e:
                logger.error(f"Error in worker discovery: {e}", exc_info=True)
                await asyncio.sleep(10)  # Brief delay before retry

    async def stop_discovery(self):
        """Stop background worker discovery"""
        self._running = False
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped worker discovery service")

    # ==================== Utility Methods ====================

    def get_worker_distribution(self) -> Dict[str, int]:
        """
        Get worker distribution across pools.

        Returns:
            Dictionary mapping group_id to worker count
        """
        distribution = {}
        for group_id, pool in self.pools.items():
            distribution[group_id] = len(pool.worker_ids)
        return distribution

    def get_total_capacity(self) -> Dict[str, Any]:
        """
        Get total system capacity.

        Returns:
            Dictionary with total GPUs, CPUs, RAM, storage
        """
        all_workers = self.discover_workers()

        return {
            "total_workers": len(all_workers),
            "total_gpus": sum(w.capabilities.gpu_count for w in all_workers),
            "total_cpus": sum(w.capabilities.cpu_count for w in all_workers),
            "total_ram_gb": sum(w.capabilities.ram_gb for w in all_workers),
            "total_storage_gb": sum(w.capabilities.storage_gb for w in all_workers),
            "avg_compute_score": (
                sum(w.capabilities.get_compute_score() for w in all_workers) / len(all_workers)
                if all_workers
                else 0
            ),
        }
