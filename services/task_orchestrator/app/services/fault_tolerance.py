"""
Fault Tolerance Service

Implements fault tolerance mechanisms for distributed training coordination:
- Automatic task reassignment on worker failure
- Exponential backoff retry logic
- Dead letter queue for permanently failed tasks
- Checkpoint recovery
- Graceful degradation strategies
- Circuit breaker pattern
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class FailureType(str, Enum):
    """Types of failures that can occur"""

    WORKER_OFFLINE = "worker_offline"
    WORKER_TIMEOUT = "worker_timeout"
    WORKER_DEGRADED = "worker_degraded"
    JOB_TIMEOUT = "job_timeout"
    JOB_ERROR = "job_error"
    VALIDATION_FAILED = "validation_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    NETWORK_ERROR = "network_error"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different failure types"""

    IMMEDIATE_REASSIGN = "immediate_reassign"  # Reassign immediately
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Retry with backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Temporarily disable worker
    CHECKPOINT_RECOVERY = "checkpoint_recovery"  # Restore from checkpoint
    DEGRADED_MODE = "degraded_mode"  # Continue with reduced resources
    DEAD_LETTER = "dead_letter"  # Move to dead letter queue


class CircuitState(str, Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, block requests
    HALF_OPEN = "half_open"  # Testing if service recovered


# ==================== Data Classes ====================


@dataclass
class RetryPolicy:
    """Retry policy configuration"""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt using exponential backoff.

        Formula: delay = min(initial_delay * (multiplier ^ attempt), max_delay)
        With jitter: delay * (1 ± jitter_factor)
        """
        delay = min(
            self.initial_delay_seconds * (self.backoff_multiplier**attempt), self.max_delay_seconds
        )

        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "initial_delay_seconds": self.initial_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter": self.jitter,
            "jitter_factor": self.jitter_factor,
        }


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close circuit
    timeout_seconds: int = 60  # Time to wait before half-open
    half_open_max_requests: int = 3  # Max requests in half-open state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "timeout_seconds": self.timeout_seconds,
            "half_open_max_requests": self.half_open_max_requests,
        }


@dataclass
class CircuitBreaker:
    """Circuit breaker for a specific resource (worker/service)"""

    resource_id: str
    config: CircuitBreakerConfig
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    half_open_requests: int = 0

    def record_success(self) -> None:
        """Record successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed operation"""
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._open()
        elif self.state == CircuitState.HALF_OPEN:
            self._open()

    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._half_open()
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_requests < self.config.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return False

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.opened_at:
            return True

        elapsed = (datetime.utcnow() - self.opened_at).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def _open(self) -> None:
        """Open circuit (block requests)"""
        logger.warning(f"Circuit breaker OPENED for {self.resource_id}")
        self.state = CircuitState.OPEN
        self.opened_at = datetime.utcnow()
        self.success_count = 0
        self.half_open_requests = 0

    def _half_open(self) -> None:
        """Half-open circuit (test if recovered)"""
        logger.info(f"Circuit breaker HALF-OPEN for {self.resource_id}")
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        self.success_count = 0

    def _close(self) -> None:
        """Close circuit (resume normal operation)"""
        logger.info(f"Circuit breaker CLOSED for {self.resource_id}")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.opened_at = None
        self.half_open_requests = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }


@dataclass
class FailureRecord:
    """Record of a failure event"""

    failure_id: str
    job_id: str
    worker_id: Optional[str]
    failure_type: FailureType
    error_message: str
    occurred_at: datetime
    retry_count: int = 0
    max_retries: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF
    next_retry_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check if retry is allowed"""
        return self.retry_count < self.max_retries and not self.resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "job_id": self.job_id,
            "worker_id": self.worker_id,
            "failure_type": self.failure_type.value,
            "error_message": self.error_message,
            "occurred_at": self.occurred_at.isoformat(),
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_strategy": self.recovery_strategy.value,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


@dataclass
class CheckpointInfo:
    """Checkpoint information for recovery"""

    checkpoint_id: str
    job_id: str
    epoch: int
    step: int
    gcs_path: str
    created_at: datetime
    model_state_size_mb: float
    optimizer_state_size_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "epoch": self.epoch,
            "step": self.step,
            "gcs_path": self.gcs_path,
            "created_at": self.created_at.isoformat(),
            "model_state_size_mb": self.model_state_size_mb,
            "optimizer_state_size_mb": self.optimizer_state_size_mb,
            "metadata": self.metadata,
        }


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance service"""

    enable_auto_reassignment: bool = True
    enable_circuit_breaker: bool = True
    enable_checkpoint_recovery: bool = True
    default_retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    dead_letter_max_age_hours: int = 168  # 1 week
    checkpoint_interval_minutes: int = 30
    health_check_interval_seconds: int = 60
    worker_failure_detection_threshold: int = 3
    max_concurrent_recoveries: int = 10


# ==================== Fault Tolerance Service ====================


class FaultToleranceService:
    """
    Fault tolerance service for distributed training coordination.

    Provides:
    - Automatic failure detection
    - Task reassignment with retry logic
    - Circuit breaker pattern
    - Checkpoint-based recovery
    - Dead letter queue management
    """

    def __init__(
        self,
        task_assignment_service,  # TASK-6.4
        worker_discovery_service,  # TASK-6.3
        job_queue,  # TASK-6.2
        worker_registry,  # TASK-6.1
        config: Optional[FaultToleranceConfig] = None,
    ):
        self.task_assignment = task_assignment_service
        self.worker_discovery = worker_discovery_service
        self.job_queue = job_queue
        self.worker_registry = worker_registry
        self.config = config or FaultToleranceConfig()

        # Internal state
        self.failure_records: Dict[str, FailureRecord] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.checkpoints: Dict[str, List[CheckpointInfo]] = {}  # job_id -> checkpoints
        self.dead_letter_queue: List[FailureRecord] = []

        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.retry_task: Optional[asyncio.Task] = None
        self.recovery_semaphore = asyncio.Semaphore(config.max_concurrent_recoveries)

        logger.info("FaultToleranceService initialized")

    # ==================== Failure Detection ====================

    async def detect_failures(self) -> List[FailureRecord]:
        """
        Detect failures across workers and jobs.

        Returns:
            List of newly detected failures
        """
        failures = []

        # Check worker health
        worker_failures = await self._detect_worker_failures()
        failures.extend(worker_failures)

        # Check job timeouts
        job_failures = await self._detect_job_failures()
        failures.extend(job_failures)

        # Store failure records
        for failure in failures:
            self.failure_records[failure.failure_id] = failure
            logger.warning(
                f"Failure detected: {failure.failure_type.value} "
                f"for job {failure.job_id} on worker {failure.worker_id}"
            )

        return failures

    async def _detect_worker_failures(self) -> List[FailureRecord]:
        """Detect worker failures"""
        failures = []

        # Get all workers
        workers = self.worker_discovery.discover_workers()

        for worker in workers:
            # Check if worker is offline or degraded
            if hasattr(worker, "status"):
                if worker.status == "offline":
                    # Get jobs assigned to this worker
                    jobs = self.job_queue.list_jobs(worker_id=worker.worker_id, status="running")

                    for job in jobs:
                        failure_id = f"failure_{job.job_id}_{datetime.utcnow().timestamp()}"
                        failure = FailureRecord(
                            failure_id=failure_id,
                            job_id=job.job_id,
                            worker_id=worker.worker_id,
                            failure_type=FailureType.WORKER_OFFLINE,
                            error_message=f"Worker {worker.worker_id} is offline",
                            occurred_at=datetime.utcnow(),
                            max_retries=self.config.default_retry_policy.max_retries,
                            recovery_strategy=RecoveryStrategy.IMMEDIATE_REASSIGN,
                        )
                        failures.append(failure)

                elif worker.status == "degraded":
                    # Record degraded worker for circuit breaker
                    self._record_worker_degradation(worker.worker_id)

        return failures

    async def _detect_job_failures(self) -> List[FailureRecord]:
        """Detect job failures (timeouts, errors)"""
        failures = []

        # Check for timeout jobs (would be detected by JobQueue)
        # This is a placeholder - actual timeout detection happens in TASK-6.2

        return failures

    def _record_worker_degradation(self, worker_id: str) -> None:
        """Record worker degradation for circuit breaker"""
        breaker = self._get_circuit_breaker(worker_id)
        breaker.record_failure()

    # ==================== Failure Recovery ====================

    async def recover_from_failure(
        self, failure: FailureRecord, checkpoint_id: Optional[str] = None
    ) -> bool:
        """
        Recover from a failure using appropriate strategy.

        Args:
            failure: Failure record to recover from
            checkpoint_id: Optional checkpoint to restore from

        Returns:
            True if recovery successful, False otherwise
        """
        async with self.recovery_semaphore:
            logger.info(
                f"Recovering from failure {failure.failure_id} "
                f"using strategy {failure.recovery_strategy.value}"
            )

            try:
                if failure.recovery_strategy == RecoveryStrategy.IMMEDIATE_REASSIGN:
                    return await self._immediate_reassign(failure)

                elif failure.recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                    return await self._retry_with_backoff(failure)

                elif failure.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    return await self._circuit_breaker_recovery(failure)

                elif failure.recovery_strategy == RecoveryStrategy.CHECKPOINT_RECOVERY:
                    return await self._checkpoint_recovery(failure, checkpoint_id)

                elif failure.recovery_strategy == RecoveryStrategy.DEGRADED_MODE:
                    return await self._degraded_mode_recovery(failure)

                elif failure.recovery_strategy == RecoveryStrategy.DEAD_LETTER:
                    return await self._move_to_dead_letter(failure)

                else:
                    logger.error(f"Unknown recovery strategy: {failure.recovery_strategy}")
                    return False

            except Exception as e:
                logger.error(f"Error during recovery: {e}")
                return False

    async def _immediate_reassign(self, failure: FailureRecord) -> bool:
        """Immediately reassign job to a different worker"""
        if not self.config.enable_auto_reassignment:
            logger.info("Auto-reassignment disabled")
            return False

        # Get circuit breaker for failed worker
        if failure.worker_id:
            breaker = self._get_circuit_breaker(failure.worker_id)
            breaker.record_failure()

        # Release job from failed worker
        self.job_queue.release_job_from_worker(
            failure.job_id,
            failure.worker_id or "unknown",
            f"Worker failure: {failure.error_message}",
        )

        # Find new worker (excluding failed one)
        from app.services.task_assignment import AssignmentConstraints

        constraints = AssignmentConstraints(
            exclude_workers={failure.worker_id} if failure.worker_id else set()
        )

        # Reassign job
        result = await self.task_assignment.assign_job(
            job_id=failure.job_id, constraints=constraints
        )

        if result.status.value == "success":
            failure.resolved = True
            failure.resolved_at = datetime.utcnow()
            logger.info(f"Job {failure.job_id} successfully reassigned to {result.worker_id}")
            return True
        else:
            logger.warning(f"Failed to reassign job {failure.job_id}")
            return False

    async def _retry_with_backoff(self, failure: FailureRecord) -> bool:
        """Retry with exponential backoff"""
        if not failure.can_retry():
            logger.info(f"Max retries exceeded for job {failure.job_id}, moving to dead letter")
            await self._move_to_dead_letter(failure)
            return False

        # Calculate delay
        delay = self.config.default_retry_policy.calculate_delay(failure.retry_count)
        failure.retry_count += 1
        failure.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)

        logger.info(
            f"Scheduling retry {failure.retry_count}/{failure.max_retries} "
            f"for job {failure.job_id} in {delay:.1f}s"
        )

        # Wait for delay
        await asyncio.sleep(delay)

        # Attempt reassignment
        return await self._immediate_reassign(failure)

    async def _circuit_breaker_recovery(self, failure: FailureRecord) -> bool:
        """Recover using circuit breaker pattern"""
        if not failure.worker_id:
            return await self._immediate_reassign(failure)

        breaker = self._get_circuit_breaker(failure.worker_id)

        if not breaker.can_attempt():
            logger.warning(
                f"Circuit breaker OPEN for worker {failure.worker_id}, "
                f"reassigning to different worker"
            )
            return await self._immediate_reassign(failure)

        # Try to use same worker
        result = await self.task_assignment.assign_job(job_id=failure.job_id, constraints=None)

        if result.status.value == "success" and result.worker_id == failure.worker_id:
            breaker.record_success()
            failure.resolved = True
            failure.resolved_at = datetime.utcnow()
            return True
        else:
            breaker.record_failure()
            return await self._immediate_reassign(failure)

    async def _checkpoint_recovery(
        self, failure: FailureRecord, checkpoint_id: Optional[str] = None
    ) -> bool:
        """Recover from checkpoint"""
        if not self.config.enable_checkpoint_recovery:
            logger.info("Checkpoint recovery disabled")
            return await self._immediate_reassign(failure)

        # Get latest checkpoint for job
        checkpoints = self.checkpoints.get(failure.job_id, [])

        if not checkpoints:
            logger.warning(f"No checkpoints found for job {failure.job_id}")
            return await self._immediate_reassign(failure)

        # Select checkpoint
        if checkpoint_id:
            checkpoint = next((cp for cp in checkpoints if cp.checkpoint_id == checkpoint_id), None)
        else:
            # Use latest checkpoint
            checkpoint = max(checkpoints, key=lambda cp: cp.created_at)

        if not checkpoint:
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return await self._immediate_reassign(failure)

        logger.info(
            f"Restoring job {failure.job_id} from checkpoint "
            f"{checkpoint.checkpoint_id} (epoch {checkpoint.epoch}, step {checkpoint.step})"
        )

        # Reassign job with checkpoint metadata
        failure.metadata["checkpoint_id"] = checkpoint.checkpoint_id
        failure.metadata["restore_from_epoch"] = checkpoint.epoch
        failure.metadata["restore_from_step"] = checkpoint.step

        return await self._immediate_reassign(failure)

    async def _degraded_mode_recovery(self, failure: FailureRecord) -> bool:
        """Continue in degraded mode with reduced resources"""
        logger.info(f"Attempting degraded mode recovery for job {failure.job_id}")

        # Reduce resource requirements
        from app.services.task_assignment import AssignmentConstraints

        job = self.job_queue.get_job(failure.job_id)
        if not job:
            return False

        # Try with reduced GPU requirements
        constraints = AssignmentConstraints(
            exclude_workers={failure.worker_id} if failure.worker_id else set(),
            min_gpu_count=max(1, job.metadata.requirements.min_gpu_count - 1),
        )

        result = await self.task_assignment.assign_job(
            job_id=failure.job_id, constraints=constraints
        )

        if result.status.value == "success":
            failure.resolved = True
            failure.resolved_at = datetime.utcnow()
            failure.metadata["degraded_mode"] = True
            logger.info(f"Job {failure.job_id} running in degraded mode")
            return True

        return False

    async def _move_to_dead_letter(self, failure: FailureRecord) -> bool:
        """Move failure to dead letter queue"""
        failure.resolved = True
        failure.resolved_at = datetime.utcnow()
        self.dead_letter_queue.append(failure)

        # Cancel job in queue
        self.job_queue.cancel_job(failure.job_id)

        logger.warning(
            f"Job {failure.job_id} moved to dead letter queue "
            f"after {failure.retry_count} retries"
        )
        return True

    # ==================== Circuit Breaker Management ====================

    def _get_circuit_breaker(self, resource_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for resource"""
        if resource_id not in self.circuit_breakers:
            self.circuit_breakers[resource_id] = CircuitBreaker(
                resource_id=resource_id, config=self.config.circuit_breaker_config
            )
        return self.circuit_breakers[resource_id]

    def get_circuit_breaker_status(self, resource_id: str) -> Dict[str, Any]:
        """Get circuit breaker status for resource"""
        breaker = self._get_circuit_breaker(resource_id)
        return breaker.to_dict()

    def reset_circuit_breaker(self, resource_id: str) -> None:
        """Manually reset circuit breaker"""
        if resource_id in self.circuit_breakers:
            self.circuit_breakers[resource_id]._close()
            logger.info(f"Circuit breaker manually reset for {resource_id}")

    # ==================== Checkpoint Management ====================

    def register_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str,
        epoch: int,
        step: int,
        gcs_path: str,
        model_state_size_mb: float,
        optimizer_state_size_mb: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointInfo:
        """Register a checkpoint for a job"""
        checkpoint = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            job_id=job_id,
            epoch=epoch,
            step=step,
            gcs_path=gcs_path,
            created_at=datetime.utcnow(),
            model_state_size_mb=model_state_size_mb,
            optimizer_state_size_mb=optimizer_state_size_mb,
            metadata=metadata or {},
        )

        if job_id not in self.checkpoints:
            self.checkpoints[job_id] = []

        self.checkpoints[job_id].append(checkpoint)

        logger.info(
            f"Checkpoint {checkpoint_id} registered for job {job_id} "
            f"(epoch {epoch}, step {step})"
        )

        return checkpoint

    def get_checkpoints(self, job_id: str) -> List[CheckpointInfo]:
        """Get all checkpoints for a job"""
        return self.checkpoints.get(job_id, [])

    def get_latest_checkpoint(self, job_id: str) -> Optional[CheckpointInfo]:
        """Get latest checkpoint for a job"""
        checkpoints = self.checkpoints.get(job_id, [])
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: cp.created_at)

    # ==================== Dead Letter Queue Management ====================

    def get_dead_letter_queue(self) -> List[FailureRecord]:
        """Get all entries in dead letter queue"""
        return self.dead_letter_queue.copy()

    def retry_from_dead_letter(self, failure_id: str) -> bool:
        """Retry a job from dead letter queue"""
        failure = next((f for f in self.dead_letter_queue if f.failure_id == failure_id), None)

        if not failure:
            logger.warning(f"Failure {failure_id} not found in dead letter queue")
            return False

        # Reset failure record
        failure.resolved = False
        failure.resolved_at = None
        failure.retry_count = 0

        # Remove from dead letter queue
        self.dead_letter_queue.remove(failure)

        # Add back to failure records
        self.failure_records[failure_id] = failure

        logger.info(f"Retrying job {failure.job_id} from dead letter queue")
        return True

    def purge_dead_letter_queue(self, max_age_hours: Optional[int] = None) -> int:
        """Purge old entries from dead letter queue"""
        max_age = max_age_hours or self.config.dead_letter_max_age_hours
        cutoff = datetime.utcnow() - timedelta(hours=max_age)

        initial_count = len(self.dead_letter_queue)

        self.dead_letter_queue = [f for f in self.dead_letter_queue if f.occurred_at > cutoff]

        purged = initial_count - len(self.dead_letter_queue)

        if purged > 0:
            logger.info(f"Purged {purged} entries from dead letter queue")

        return purged

    # ==================== Background Tasks ====================

    async def start_health_monitoring(self) -> None:
        """Start background health monitoring"""
        if self.health_check_task:
            logger.warning("Health monitoring already running")
            return

        async def monitor_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval_seconds)

                    # Detect failures
                    failures = await self.detect_failures()

                    # Attempt recovery for each failure
                    for failure in failures:
                        await self.recover_from_failure(failure)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")

        self.health_check_task = asyncio.create_task(monitor_loop())
        logger.info("Health monitoring started")

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("Health monitoring stopped")

    async def start_retry_scheduler(self) -> None:
        """Start background retry scheduler"""
        if self.retry_task:
            logger.warning("Retry scheduler already running")
            return

        async def retry_loop():
            while True:
                try:
                    await asyncio.sleep(10)  # Check every 10 seconds

                    now = datetime.utcnow()

                    # Find failures ready for retry
                    for failure in self.failure_records.values():
                        if (
                            failure.next_retry_at
                            and failure.next_retry_at <= now
                            and not failure.resolved
                        ):

                            await self.recover_from_failure(failure)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in retry scheduler: {e}")

        self.retry_task = asyncio.create_task(retry_loop())
        logger.info("Retry scheduler started")

    async def stop_retry_scheduler(self) -> None:
        """Stop background retry scheduler"""
        if self.retry_task:
            self.retry_task.cancel()
            try:
                await self.retry_task
            except asyncio.CancelledError:
                pass
            self.retry_task = None
            logger.info("Retry scheduler stopped")

    # ==================== Statistics ====================

    def get_fault_tolerance_stats(self) -> Dict[str, Any]:
        """Get fault tolerance statistics"""
        total_failures = len(self.failure_records)
        resolved_failures = sum(1 for f in self.failure_records.values() if f.resolved)
        pending_failures = total_failures - resolved_failures

        # Count by failure type
        failure_by_type = {}
        for failure in self.failure_records.values():
            failure_type = failure.failure_type.value
            failure_by_type[failure_type] = failure_by_type.get(failure_type, 0) + 1

        # Circuit breaker stats
        circuit_breakers_open = sum(
            1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN
        )

        return {
            "total_failures": total_failures,
            "resolved_failures": resolved_failures,
            "pending_failures": pending_failures,
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "failure_by_type": failure_by_type,
            "circuit_breakers_total": len(self.circuit_breakers),
            "circuit_breakers_open": circuit_breakers_open,
            "total_checkpoints": sum(len(cps) for cps in self.checkpoints.values()),
            "timestamp": datetime.utcnow().isoformat(),
        }
