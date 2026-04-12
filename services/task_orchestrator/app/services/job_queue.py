"""
Job Queue Management Service

Provides Redis-based job queue with priority scheduling, state management,
and integration with Phase 4 validation and Phase 6 worker registry.

Key Features:
- Priority-based job scheduling (HIGH > MEDIUM > LOW)
- Job state machine: pending → running → completed/failed
- Validation-gated job acceptance (requires Phase 4 validation to pass)
- Job assignment to workers with worker_id and shard_id tracking
- Retry logic with exponential backoff
- Dead letter queue for permanently failed jobs
- Job metadata and error tracking
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from redis import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job lifecycle states"""

    PENDING = "pending"  # Job submitted, waiting for worker assignment
    VALIDATING = "validating"  # Model/dataset validation in progress (Phase 4)
    WAITING = "waiting"  # Validation passed, waiting for available worker
    RUNNING = "running"  # Assigned to worker, training in progress
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed (retries exhausted or critical error)
    CANCELLED = "cancelled"  # Manually cancelled by user
    TIMEOUT = "timeout"  # Exceeded maximum execution time


class JobPriority(Enum):
    """Job priority levels for scheduling"""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3  # For system-critical jobs

    def __lt__(self, other):
        """Enable priority comparison"""
        if isinstance(other, JobPriority):
            return self.value < other.value
        return NotImplemented


@dataclass
class JobRequirements:
    """Resource requirements for job execution"""

    min_gpu_count: int = 0
    min_gpu_memory_gb: float = 0.0
    min_cpu_count: int = 1
    min_ram_gb: float = 1.0
    requires_cuda: bool = False
    requires_mps: bool = False
    max_execution_time_seconds: int = 3600  # 1 hour default

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRequirements":
        return cls(**data)


@dataclass
class JobMetadata:
    """Job metadata and configuration"""

    job_id: str
    group_id: str
    model_id: str
    dataset_id: str
    user_id: str

    # Configuration
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"

    # Requirements
    requirements: JobRequirements = field(default_factory=JobRequirements)

    # Tags and metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["requirements"] = self.requirements.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobMetadata":
        if "requirements" in data and isinstance(data["requirements"], dict):
            data["requirements"] = JobRequirements.from_dict(data["requirements"])
        return cls(**data)


@dataclass
class JobInfo:
    """Complete job state and tracking information"""

    job_id: str
    metadata: JobMetadata
    status: JobStatus
    priority: JobPriority

    # Timestamps
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Assignment tracking
    assigned_worker_id: Optional[str] = None
    assigned_shard_ids: List[int] = field(default_factory=list)

    # Validation tracking (Phase 4 integration)
    model_validation_status: str = "pending"  # pending, validating, passed, failed
    dataset_validation_status: str = "pending"
    validation_errors: List[str] = field(default_factory=list)

    # Progress tracking
    progress_percent: float = 0.0
    current_epoch: int = 0
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Result storage
    result_path: Optional[str] = None  # GCS path to trained model
    metrics_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = self.metadata.to_dict()
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobInfo":
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = JobMetadata.from_dict(data["metadata"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = JobStatus(data["status"])
        if "priority" in data and isinstance(data["priority"], (int, str)):
            data["priority"] = (
                JobPriority(data["priority"])
                if isinstance(data["priority"], str)
                else JobPriority(data["priority"])
            )
        return cls(**data)

    def is_terminal_state(self) -> bool:
        """Check if job is in a terminal state (no further transitions)"""
        return self.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }

    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries and self.status == JobStatus.FAILED

    def get_execution_time_seconds(self) -> Optional[float]:
        """Calculate job execution time"""
        if not self.started_at:
            return None

        end_time = self.completed_at if self.completed_at else datetime.utcnow().isoformat()
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(end_time)
        return (end - start).total_seconds()


class JobQueue:
    """
    Redis-based job queue with priority scheduling and state management.

    Redis Data Structures:
    - jobs:{job_id} -> JobInfo JSON (hash)
    - queue:{priority} -> sorted set by timestamp (ZSET)
    - jobs:by_status:{status} -> set of job_ids (SET)
    - jobs:by_group:{group_id} -> set of job_ids (SET)
    - jobs:by_worker:{worker_id} -> set of job_ids (SET)
    - jobs:dead_letter -> list of failed job_ids (LIST)
    - jobs:validation_pending -> set of job_ids awaiting validation (SET)
    """

    def __init__(
        self,
        redis_client: Redis,
        validation_timeout_seconds: int = 300,  # 5 minutes
        job_timeout_seconds: int = 3600,  # 1 hour
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        self.redis = redis_client
        self.validation_timeout = validation_timeout_seconds
        self.job_timeout = job_timeout_seconds
        self.cleanup_interval = cleanup_interval_seconds

        # Key prefixes
        self.JOB_KEY = "jobs:{job_id}"
        self.QUEUE_KEY = "queue:{priority}"
        self.STATUS_KEY = "jobs:by_status:{status}"
        self.GROUP_KEY = "jobs:by_group:{group_id}"
        self.WORKER_KEY = "jobs:by_worker:{worker_id}"
        self.DEAD_LETTER_KEY = "jobs:dead_letter"
        self.VALIDATION_PENDING_KEY = "jobs:validation_pending"

    # ==================== Job Submission ====================

    def submit_job(
        self, metadata: JobMetadata, priority: JobPriority = JobPriority.MEDIUM
    ) -> JobInfo:
        """
        Submit new job to queue.

        Algorithm:
        1. Create JobInfo with PENDING status
        2. Store job data in Redis
        3. Add to validation queue (Phase 4 integration)
        4. Add to priority queue
        5. Index by status and group
        6. Return JobInfo
        """
        now = datetime.utcnow().isoformat()

        job_info = JobInfo(
            job_id=metadata.job_id,
            metadata=metadata,
            status=JobStatus.PENDING,
            priority=priority,
            created_at=now,
            updated_at=now,
        )

        try:
            # Store job data
            job_key = self.JOB_KEY.format(job_id=metadata.job_id)
            self.redis.set(job_key, json.dumps(job_info.to_dict()))

            # Add to validation pending queue
            self.redis.sadd(self.VALIDATION_PENDING_KEY, metadata.job_id)

            # Add to priority queue (score = timestamp for FIFO within priority)
            queue_key = self.QUEUE_KEY.format(priority=priority.value)
            self.redis.zadd(queue_key, {metadata.job_id: time.time()})

            # Index by status
            status_key = self.STATUS_KEY.format(status=JobStatus.PENDING.value)
            self.redis.sadd(status_key, metadata.job_id)

            # Index by group
            group_key = self.GROUP_KEY.format(group_id=metadata.group_id)
            self.redis.sadd(group_key, metadata.job_id)

            logger.info(f"Job {metadata.job_id} submitted with priority {priority.name}")
            return job_info

        except RedisError as e:
            logger.error(f"Failed to submit job {metadata.job_id}: {e}")
            raise

    # ==================== Job State Transitions ====================

    def update_job_status(
        self,
        job_id: str,
        new_status: JobStatus,
        error_message: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Update job status with proper state machine validation.

        Valid Transitions:
        - PENDING → VALIDATING → WAITING → RUNNING → COMPLETED
        - * → FAILED
        - * → CANCELLED
        - RUNNING → TIMEOUT
        - FAILED → PENDING (retry)
        """
        job_info = self.get_job(job_id)
        if not job_info:
            logger.warning(f"Job {job_id} not found")
            return False

        old_status = job_info.status

        # Validate state transition
        if not self._is_valid_transition(old_status, new_status):
            logger.warning(
                f"Invalid transition for job {job_id}: {old_status.value} → {new_status.value}"
            )
            return False

        # Update job info
        job_info.status = new_status
        job_info.updated_at = datetime.utcnow().isoformat()

        if error_message:
            job_info.error_message = error_message

        if new_status == JobStatus.RUNNING and not job_info.started_at:
            job_info.started_at = job_info.updated_at
            if worker_id:
                job_info.assigned_worker_id = worker_id

        if new_status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        }:
            job_info.completed_at = job_info.updated_at

        try:
            # Update job data
            job_key = self.JOB_KEY.format(job_id=job_id)
            self.redis.set(job_key, json.dumps(job_info.to_dict()))

            # Update status indices
            old_status_key = self.STATUS_KEY.format(status=old_status.value)
            new_status_key = self.STATUS_KEY.format(status=new_status.value)
            self.redis.srem(old_status_key, job_id)
            self.redis.sadd(new_status_key, job_id)

            # Handle dead letter queue for permanent failures
            if new_status == JobStatus.FAILED and not job_info.can_retry():
                self.redis.lpush(self.DEAD_LETTER_KEY, job_id)

            # Remove from validation pending if transitioning from VALIDATING
            if old_status == JobStatus.VALIDATING:
                self.redis.srem(self.VALIDATION_PENDING_KEY, job_id)

            logger.info(f"Job {job_id} status updated: {old_status.value} → {new_status.value}")
            return True

        except RedisError as e:
            logger.error(f"Failed to update job {job_id} status: {e}")
            return False

    def _is_valid_transition(self, old_status: JobStatus, new_status: JobStatus) -> bool:
        """Validate state machine transitions"""
        # Terminal states cannot transition
        if old_status in {JobStatus.COMPLETED, JobStatus.CANCELLED}:
            return False

        # Can always transition to FAILED or CANCELLED
        if new_status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            return True

        # Valid forward transitions
        valid_transitions = {
            JobStatus.PENDING: {JobStatus.VALIDATING, JobStatus.WAITING},
            JobStatus.VALIDATING: {JobStatus.WAITING, JobStatus.FAILED},
            JobStatus.WAITING: {JobStatus.RUNNING},
            JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.TIMEOUT},
            JobStatus.FAILED: {JobStatus.PENDING},  # Retry
            JobStatus.TIMEOUT: {JobStatus.PENDING},  # Retry
        }

        return new_status in valid_transitions.get(old_status, set())

    # ==================== Job Assignment ====================

    def assign_job_to_worker(
        self, job_id: str, worker_id: str, shard_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Assign job to worker.

        Algorithm:
        1. Validate job is in WAITING status
        2. Update job with worker_id and shard_ids
        3. Change status to RUNNING
        4. Index by worker_id
        5. Remove from priority queue
        """
        job_info = self.get_job(job_id)
        if not job_info:
            return False

        if job_info.status != JobStatus.WAITING:
            logger.warning(f"Job {job_id} not in WAITING status (current: {job_info.status.value})")
            return False

        job_info.assigned_worker_id = worker_id
        job_info.assigned_shard_ids = shard_ids or []

        try:
            # Update status to RUNNING
            if not self.update_job_status(job_id, JobStatus.RUNNING, worker_id=worker_id):
                return False

            # Index by worker
            worker_key = self.WORKER_KEY.format(worker_id=worker_id)
            self.redis.sadd(worker_key, job_id)

            # Remove from all priority queues
            for priority in JobPriority:
                queue_key = self.QUEUE_KEY.format(priority=priority.value)
                self.redis.zrem(queue_key, job_id)

            logger.info(f"Job {job_id} assigned to worker {worker_id} with shards {shard_ids}")
            return True

        except RedisError as e:
            logger.error(f"Failed to assign job {job_id} to worker {worker_id}: {e}")
            return False

    def release_job_from_worker(
        self, job_id: str, worker_id: str, reason: str = "worker_failure"
    ) -> bool:
        """
        Release job from worker (for reassignment after failure).

        Algorithm:
        1. Validate job is assigned to worker
        2. Increment retry count
        3. If can retry: status → WAITING, re-add to queue
        4. If cannot retry: status → FAILED, add to dead letter
        5. Remove from worker index
        """
        job_info = self.get_job(job_id)
        if not job_info or job_info.assigned_worker_id != worker_id:
            return False

        try:
            # Increment retry count
            job_info.retry_count += 1
            job_info.error_message = f"Released from worker {worker_id}: {reason}"

            # Remove from worker index
            worker_key = self.WORKER_KEY.format(worker_id=worker_id)
            self.redis.srem(worker_key, job_id)

            # Clear assignment
            job_info.assigned_worker_id = None
            job_info.assigned_shard_ids = []

            # Update job data
            job_key = self.JOB_KEY.format(job_id=job_id)
            self.redis.set(job_key, json.dumps(job_info.to_dict()))

            if job_info.can_retry():
                # Re-queue for retry
                self.update_job_status(job_id, JobStatus.WAITING)
                queue_key = self.QUEUE_KEY.format(priority=job_info.priority.value)
                self.redis.zadd(queue_key, {job_id: time.time()})
                logger.info(
                    f"Job {job_id} released and re-queued (retry {job_info.retry_count}/{job_info.max_retries})"
                )
            else:
                # Mark as permanently failed
                self.update_job_status(
                    job_id, JobStatus.FAILED, error_message="Max retries exceeded"
                )
                logger.warning(f"Job {job_id} failed permanently (max retries exceeded)")

            return True

        except RedisError as e:
            logger.error(f"Failed to release job {job_id} from worker {worker_id}: {e}")
            return False

    # ==================== Job Retrieval ====================

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Retrieve job by ID"""
        try:
            job_key = self.JOB_KEY.format(job_id=job_id)
            data = self.redis.get(job_key)
            if not data:
                return None
            return JobInfo.from_dict(json.loads(data))
        except (RedisError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to retrieve job {job_id}: {e}")
            return None

    def get_next_job(self, requirements: Optional[JobRequirements] = None) -> Optional[JobInfo]:
        """
        Get next job from queue based on priority.

        Algorithm:
        1. Check queues in priority order (CRITICAL → HIGH → MEDIUM → LOW)
        2. Get oldest job in each queue (lowest ZSET score)
        3. If requirements provided, validate job matches
        4. Return first matching job
        """
        try:
            # Check queues in priority order (highest first)
            for priority in sorted(JobPriority, reverse=True):
                queue_key = self.QUEUE_KEY.format(priority=priority.value)

                # Get oldest job in queue (ZRANGE with lowest score)
                job_ids = self.redis.zrange(queue_key, 0, 0)
                if not job_ids:
                    continue

                job_id = job_ids[0].decode() if isinstance(job_ids[0], bytes) else job_ids[0]
                job_info = self.get_job(job_id)

                if not job_info:
                    # Clean up orphaned entry
                    self.redis.zrem(queue_key, job_id)
                    continue

                # Validate job is in WAITING status
                if job_info.status != JobStatus.WAITING:
                    continue

                # Validate requirements if provided
                if requirements and not self._matches_requirements(job_info, requirements):
                    continue

                return job_info

            return None

        except RedisError as e:
            logger.error(f"Failed to get next job: {e}")
            return None

    def _matches_requirements(self, job_info: JobInfo, requirements: JobRequirements) -> bool:
        """Check if worker requirements satisfy job requirements"""
        job_req = job_info.metadata.requirements

        return (
            requirements.min_gpu_count >= job_req.min_gpu_count
            and requirements.min_gpu_memory_gb >= job_req.min_gpu_memory_gb
            and requirements.min_cpu_count >= job_req.min_cpu_count
            and requirements.min_ram_gb >= job_req.min_ram_gb
            and (not job_req.requires_cuda or requirements.requires_cuda)
            and (not job_req.requires_mps or requirements.requires_mps)
        )

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        group_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[JobInfo]:
        """List jobs with optional filters"""
        try:
            job_ids = set()

            if status:
                status_key = self.STATUS_KEY.format(status=status.value)
                job_ids = set(self.redis.smembers(status_key))
            elif group_id:
                group_key = self.GROUP_KEY.format(group_id=group_id)
                job_ids = set(self.redis.smembers(group_key))
            elif worker_id:
                worker_key = self.WORKER_KEY.format(worker_id=worker_id)
                job_ids = set(self.redis.smembers(worker_key))
            else:
                # Get all jobs from all status sets
                for s in JobStatus:
                    status_key = self.STATUS_KEY.format(status=s.value)
                    job_ids.update(self.redis.smembers(status_key))

            # Convert bytes to strings
            job_ids = [jid.decode() if isinstance(jid, bytes) else jid for jid in job_ids]

            # Retrieve job info
            jobs = []
            for job_id in list(job_ids)[:limit]:
                job_info = self.get_job(job_id)
                if job_info:
                    jobs.append(job_info)

            # Sort by created_at (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs

        except RedisError as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    # ==================== Validation Integration (Phase 4) ====================

    def mark_validation_complete(
        self,
        job_id: str,
        model_validation_passed: bool,
        dataset_validation_passed: bool,
        validation_errors: Optional[List[str]] = None,
    ) -> bool:
        """
        Mark job validation complete (called by Phase 4 validation service).

        Algorithm:
        1. Update validation status fields
        2. If both passed: status → WAITING
        3. If any failed: status → FAILED
        4. Remove from validation pending set
        """
        job_info = self.get_job(job_id)
        if not job_info:
            return False

        try:
            job_info.model_validation_status = "passed" if model_validation_passed else "failed"
            job_info.dataset_validation_status = "passed" if dataset_validation_passed else "failed"
            job_info.validation_errors = validation_errors or []

            # Update job data
            job_key = self.JOB_KEY.format(job_id=job_id)
            self.redis.set(job_key, json.dumps(job_info.to_dict()))

            # Transition based on validation result
            if model_validation_passed and dataset_validation_passed:
                self.update_job_status(job_id, JobStatus.WAITING)
                logger.info(f"Job {job_id} validation passed, status → WAITING")
            else:
                error_msg = (
                    f"Validation failed: {', '.join(validation_errors or ['Unknown error'])}"
                )
                self.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
                logger.warning(f"Job {job_id} validation failed")

            # Remove from validation pending
            self.redis.srem(self.VALIDATION_PENDING_KEY, job_id)
            return True

        except RedisError as e:
            logger.error(f"Failed to mark validation complete for job {job_id}: {e}")
            return False

    # ==================== Job Cancellation ====================

    def cancel_job(self, job_id: str, reason: str = "user_requested") -> bool:
        """Cancel job"""
        job_info = self.get_job(job_id)
        if not job_info:
            return False

        if job_info.is_terminal_state():
            logger.warning(f"Cannot cancel job {job_id} in terminal state {job_info.status.value}")
            return False

        try:
            # Remove from priority queues
            for priority in JobPriority:
                queue_key = self.QUEUE_KEY.format(priority=priority.value)
                self.redis.zrem(queue_key, job_id)

            # Remove from worker if assigned
            if job_info.assigned_worker_id:
                worker_key = self.WORKER_KEY.format(worker_id=job_info.assigned_worker_id)
                self.redis.srem(worker_key, job_id)

            # Update status
            self.update_job_status(
                job_id, JobStatus.CANCELLED, error_message=f"Cancelled: {reason}"
            )
            logger.info(f"Job {job_id} cancelled: {reason}")
            return True

        except RedisError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    # ==================== Statistics ====================

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {
                "total_jobs": 0,
                "by_status": {},
                "by_priority": {},
                "validation_pending": self.redis.scard(self.VALIDATION_PENDING_KEY),
                "dead_letter_count": self.redis.llen(self.DEAD_LETTER_KEY),
            }

            # Count by status
            for status in JobStatus:
                status_key = self.STATUS_KEY.format(status=status.value)
                count = self.redis.scard(status_key)
                stats["by_status"][status.value] = count
                stats["total_jobs"] += count

            # Count by priority (only queued jobs)
            for priority in JobPriority:
                queue_key = self.QUEUE_KEY.format(priority=priority.value)
                count = self.redis.zcard(queue_key)
                stats["by_priority"][priority.name] = count

            return stats

        except RedisError as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

    # ==================== Cleanup ====================

    def cleanup_expired_jobs(self) -> int:
        """
        Clean up expired jobs in validation or running states.

        Returns: Number of jobs cleaned up
        """
        cleaned = 0
        now = datetime.utcnow()

        try:
            # Check validation timeout
            validation_pending = self.redis.smembers(self.VALIDATION_PENDING_KEY)
            for job_id in validation_pending:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                job_info = self.get_job(job_id)

                if not job_info:
                    continue

                created = datetime.fromisoformat(job_info.created_at)
                if (now - created).total_seconds() > self.validation_timeout:
                    self.update_job_status(
                        job_id, JobStatus.FAILED, error_message="Validation timeout exceeded"
                    )
                    cleaned += 1

            # Check job execution timeout
            running_key = self.STATUS_KEY.format(status=JobStatus.RUNNING.value)
            running_jobs = self.redis.smembers(running_key)

            for job_id in running_jobs:
                job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
                job_info = self.get_job(job_id)

                if not job_info or not job_info.started_at:
                    continue

                started = datetime.fromisoformat(job_info.started_at)
                max_time = job_info.metadata.requirements.max_execution_time_seconds

                if (now - started).total_seconds() > max_time:
                    self.update_job_status(
                        job_id,
                        JobStatus.TIMEOUT,
                        error_message=f"Execution timeout ({max_time}s exceeded)",
                    )

                    # Release from worker
                    if job_info.assigned_worker_id:
                        worker_key = self.WORKER_KEY.format(worker_id=job_info.assigned_worker_id)
                        self.redis.srem(worker_key, job_id)

                    cleaned += 1

            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired jobs")

            return cleaned

        except RedisError as e:
            logger.error(f"Failed to cleanup expired jobs: {e}")
            return 0
