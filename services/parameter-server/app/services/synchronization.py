"""
Synchronization Strategies Service for Parameter Server

Implements different synchronization strategies for distributed training:
- Synchronous: Wait for all workers before aggregating
- Asynchronous: Process gradients immediately as they arrive
- Semi-synchronous: Configurable staleness threshold with partial worker quorum

Key Features:
- Multiple synchronization modes
- Worker tracking and timeout detection
- Configurable quorum requirements
- Staleness-based filtering
- Round-based synchronization for sync mode
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from app.services.gradient_aggregation import (
    AggregatedGradient,
    AggregationConfig,
    GradientAggregationService,
    GradientUpdate,
)

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class SyncMode(str, Enum):
    """Synchronization mode"""

    SYNCHRONOUS = "synchronous"  # Wait for all workers
    ASYNCHRONOUS = "asynchronous"  # Process immediately
    SEMI_SYNCHRONOUS = "semi_synchronous"  # Wait for quorum with staleness threshold


class WorkerState(str, Enum):
    """Worker state"""

    ACTIVE = "active"
    IDLE = "idle"
    TIMED_OUT = "timed_out"
    EXCLUDED = "excluded"


# ==================== Data Classes ====================


@dataclass
class WorkerInfo:
    """Information about a worker"""

    worker_id: str
    model_id: str
    state: WorkerState = WorkerState.IDLE
    last_seen: datetime = field(default_factory=datetime.utcnow)
    last_gradient_version: Optional[int] = None
    current_round: int = 0
    total_gradients: int = 0
    total_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncRound:
    """Synchronization round information"""

    round_id: int
    model_id: str
    target_version: int
    expected_workers: Set[str]
    received_workers: Set[str] = field(default_factory=set)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    aggregation_result: Optional[AggregatedGradient] = None
    timed_out: bool = False


@dataclass
class SyncConfig:
    """Configuration for synchronization strategy"""

    mode: SyncMode = SyncMode.ASYNCHRONOUS

    # Synchronous mode settings
    min_workers: int = 1  # Minimum workers required
    max_workers: Optional[int] = None  # Maximum workers expected
    sync_timeout_seconds: float = 60.0  # Timeout for synchronous rounds

    # Semi-synchronous mode settings
    worker_quorum: float = 0.8  # Fraction of workers required (0.0-1.0)
    max_staleness: int = 10  # Maximum staleness for semi-sync
    quorum_timeout_seconds: float = 30.0  # Timeout for quorum

    # Asynchronous mode settings
    async_batch_size: int = 1  # Aggregate after N gradients
    async_timeout_seconds: Optional[float] = None  # Optional batching timeout

    # Worker management
    worker_timeout_seconds: float = 300.0  # Mark worker as timed out
    auto_exclude_timeouts: bool = True  # Exclude timed out workers

    # Aggregation config
    aggregation_config: AggregationConfig = field(default_factory=AggregationConfig)


# ==================== Synchronization Service ====================


class SynchronizationService:
    """
    Service for managing synchronization strategies in distributed training.

    Features:
    - Synchronous: Wait for all workers before aggregating
    - Asynchronous: Aggregate immediately (configurable batch size)
    - Semi-synchronous: Wait for quorum with staleness limits
    - Worker tracking and timeout detection
    - Round-based coordination for sync mode
    """

    def __init__(
        self,
        gradient_service: GradientAggregationService,
        default_config: Optional[SyncConfig] = None,
    ):
        """
        Initialize synchronization service.

        Args:
            gradient_service: Gradient aggregation service
            default_config: Default synchronization configuration
        """
        self.gradient_service = gradient_service
        self.default_config = default_config or SyncConfig()

        # Worker tracking
        # Key: worker_id -> WorkerInfo
        self.workers: Dict[str, WorkerInfo] = {}

        # Model-specific sync rounds (for synchronous mode)
        # Key: model_id -> current SyncRound
        self.current_rounds: Dict[str, SyncRound] = {}

        # Round history
        self.round_history: List[SyncRound] = []

        # Callbacks for aggregation completion
        self.aggregation_callbacks: List[Callable[[AggregatedGradient], None]] = []

        # Background tasks for async monitoring
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            f"SynchronizationService initialized with mode: {self.default_config.mode.value}"
        )

    async def submit_gradient(
        self, gradient_update: GradientUpdate, config: Optional[SyncConfig] = None
    ) -> Optional[AggregatedGradient]:
        """
        Submit a gradient update and handle synchronization.

        Args:
            gradient_update: Gradient update from worker
            config: Synchronization configuration (uses default if None)

        Returns:
            AggregatedGradient if aggregation occurred, None otherwise
        """
        config = config or self.default_config

        # Update worker info
        self._update_worker_info(gradient_update)

        # Submit to gradient service
        self.gradient_service.submit_gradient(gradient_update)

        logger.info(
            f"Gradient received from {gradient_update.worker_id} "
            f"for {gradient_update.model_id} (mode: {config.mode.value})"
        )

        # Handle based on sync mode
        if config.mode == SyncMode.SYNCHRONOUS:
            return await self._handle_synchronous(gradient_update, config)
        elif config.mode == SyncMode.ASYNCHRONOUS:
            return await self._handle_asynchronous(gradient_update, config)
        elif config.mode == SyncMode.SEMI_SYNCHRONOUS:
            return await self._handle_semi_synchronous(gradient_update, config)
        else:
            raise ValueError(f"Unknown sync mode: {config.mode}")

    async def _handle_synchronous(
        self, gradient_update: GradientUpdate, config: SyncConfig
    ) -> Optional[AggregatedGradient]:
        """
        Handle synchronous mode: Wait for all workers.
        """
        model_id = gradient_update.model_id
        worker_id = gradient_update.worker_id

        # Get or create current round
        if model_id not in self.current_rounds:
            self.current_rounds[model_id] = self._create_sync_round(
                model_id, gradient_update.version_id, config
            )

        current_round = self.current_rounds[model_id]

        # Add worker to received set
        current_round.received_workers.add(worker_id)

        logger.info(
            f"Sync round {current_round.round_id}: "
            f"{len(current_round.received_workers)}/{len(current_round.expected_workers)} workers"
        )

        # Check if all workers have submitted
        if current_round.received_workers >= current_round.expected_workers:
            return await self._complete_sync_round(model_id, config)

        # Check for timeout
        elapsed = (datetime.utcnow() - current_round.started_at).total_seconds()
        if elapsed > config.sync_timeout_seconds:
            logger.warning(
                f"Sync round {current_round.round_id} timed out "
                f"({len(current_round.received_workers)}/{len(current_round.expected_workers)} workers)"
            )
            current_round.timed_out = True
            return await self._complete_sync_round(model_id, config)

        return None

    async def _handle_asynchronous(
        self, gradient_update: GradientUpdate, config: SyncConfig
    ) -> Optional[AggregatedGradient]:
        """
        Handle asynchronous mode: Process immediately or in batches.
        """
        model_id = gradient_update.model_id

        # Check batch size
        pending = self.gradient_service.get_pending_gradients(model_id)

        if len(pending) >= config.async_batch_size:
            # Aggregate now
            logger.info(f"Async batch size reached ({len(pending)} gradients), aggregating")
            return self._aggregate_gradients(model_id, gradient_update.version_id, config)

        # If timeout configured, schedule aggregation
        if config.async_timeout_seconds is not None:
            await self._schedule_async_aggregation(model_id, config)

        return None

    async def _handle_semi_synchronous(
        self, gradient_update: GradientUpdate, config: SyncConfig
    ) -> Optional[AggregatedGradient]:
        """
        Handle semi-synchronous mode: Wait for quorum with staleness limit.
        """
        model_id = gradient_update.model_id
        worker_id = gradient_update.worker_id

        # Get or create current round
        if model_id not in self.current_rounds:
            self.current_rounds[model_id] = self._create_sync_round(
                model_id, gradient_update.version_id, config
            )

        current_round = self.current_rounds[model_id]
        current_round.received_workers.add(worker_id)

        # Get active workers
        active_workers = self._get_active_workers(model_id)
        num_active = len(active_workers)

        # Calculate quorum
        quorum_size = max(config.min_workers, int(num_active * config.worker_quorum))

        logger.info(
            f"Semi-sync round {current_round.round_id}: "
            f"{len(current_round.received_workers)}/{quorum_size} quorum "
            f"({num_active} active workers)"
        )

        # Check if quorum reached
        if len(current_round.received_workers) >= quorum_size:
            logger.info(f"Quorum reached, aggregating")
            return await self._complete_sync_round(model_id, config)

        # Check for timeout
        elapsed = (datetime.utcnow() - current_round.started_at).total_seconds()
        if elapsed > config.quorum_timeout_seconds:
            logger.warning(
                f"Semi-sync round {current_round.round_id} timed out "
                f"({len(current_round.received_workers)}/{quorum_size} quorum)"
            )
            current_round.timed_out = True

            # Aggregate if at least min_workers
            if len(current_round.received_workers) >= config.min_workers:
                return await self._complete_sync_round(model_id, config)
            else:
                logger.error(
                    f"Insufficient workers for aggregation "
                    f"({len(current_round.received_workers)} < {config.min_workers})"
                )
                # Start new round
                del self.current_rounds[model_id]
                return None

        return None

    async def _complete_sync_round(
        self, model_id: str, config: SyncConfig
    ) -> Optional[AggregatedGradient]:
        """Complete synchronization round and aggregate gradients"""
        if model_id not in self.current_rounds:
            return None

        current_round = self.current_rounds[model_id]

        # Aggregate gradients
        result = self._aggregate_gradients(model_id, current_round.target_version, config)

        # Update round
        current_round.completed_at = datetime.utcnow()
        current_round.aggregation_result = result

        # Add to history
        self.round_history.append(current_round)

        # Start new round
        del self.current_rounds[model_id]

        # Trigger callbacks
        if result:
            for callback in self.aggregation_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Error in aggregation callback: {e}")

        logger.info(
            f"Completed round {current_round.round_id} for {model_id} "
            f"({len(current_round.received_workers)} workers, "
            f"{result.total_samples if result else 0} samples)"
        )

        return result

    def _aggregate_gradients(
        self, model_id: str, current_version: int, config: SyncConfig
    ) -> Optional[AggregatedGradient]:
        """Aggregate pending gradients"""
        result = self.gradient_service.aggregate_gradients(
            model_id=model_id,
            current_version=current_version,
            config=config.aggregation_config,
            clear_buffer=True,
        )
        return result

    def _create_sync_round(
        self, model_id: str, target_version: int, config: SyncConfig
    ) -> SyncRound:
        """Create a new synchronization round"""
        # Get expected workers
        active_workers = self._get_active_workers(model_id)

        # Get next round ID
        round_id = len(self.round_history) + len(self.current_rounds) + 1

        sync_round = SyncRound(
            round_id=round_id,
            model_id=model_id,
            target_version=target_version,
            expected_workers=set(active_workers),
        )

        logger.info(
            f"Created sync round {round_id} for {model_id} "
            f"(expecting {len(active_workers)} workers)"
        )

        return sync_round

    def _update_worker_info(self, gradient_update: GradientUpdate) -> None:
        """Update worker information"""
        worker_id = gradient_update.worker_id

        if worker_id not in self.workers:
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id, model_id=gradient_update.model_id
            )

        worker = self.workers[worker_id]
        worker.state = WorkerState.ACTIVE
        worker.last_seen = datetime.utcnow()
        worker.last_gradient_version = gradient_update.version_id
        worker.total_gradients += 1
        worker.total_samples += gradient_update.num_samples

    def _get_active_workers(self, model_id: str) -> List[str]:
        """Get list of active workers for a model"""
        active = []
        now = datetime.utcnow()

        for worker_id, worker in self.workers.items():
            if worker.model_id != model_id:
                continue

            # Check timeout
            elapsed = (now - worker.last_seen).total_seconds()
            if elapsed > self.default_config.worker_timeout_seconds:
                if worker.state != WorkerState.TIMED_OUT:
                    logger.warning(f"Worker {worker_id} timed out")
                    worker.state = WorkerState.TIMED_OUT
                continue

            if worker.state == WorkerState.ACTIVE or worker.state == WorkerState.IDLE:
                active.append(worker_id)

        return active

    async def _schedule_async_aggregation(self, model_id: str, config: SyncConfig) -> None:
        """Schedule asynchronous aggregation after timeout"""
        # Cancel existing task if any
        if model_id in self._monitoring_tasks:
            self._monitoring_tasks[model_id].cancel()

        async def aggregate_after_timeout():
            await asyncio.sleep(config.async_timeout_seconds)
            pending = self.gradient_service.get_pending_gradients(model_id)
            if pending:
                logger.info(f"Async timeout reached, aggregating {len(pending)} gradients")
                result = self._aggregate_gradients(model_id, pending[0].version_id, config)
                if result:
                    for callback in self.aggregation_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"Error in aggregation callback: {e}")

        task = asyncio.create_task(aggregate_after_timeout())
        self._monitoring_tasks[model_id] = task

    def register_worker(
        self, worker_id: str, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> WorkerInfo:
        """
        Register a new worker.

        Args:
            worker_id: Worker identifier
            model_id: Model identifier
            metadata: Optional worker metadata

        Returns:
            WorkerInfo
        """
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.state = WorkerState.ACTIVE
            worker.last_seen = datetime.utcnow()
            if metadata:
                worker.metadata.update(metadata)
        else:
            worker = WorkerInfo(
                worker_id=worker_id,
                model_id=model_id,
                state=WorkerState.ACTIVE,
                metadata=metadata or {},
            )
            self.workers[worker_id] = worker

        logger.info(f"Registered worker {worker_id} for {model_id}")
        return worker

    def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker.

        Args:
            worker_id: Worker identifier

        Returns:
            True if worker was removed
        """
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.state = WorkerState.EXCLUDED
            logger.info(f"Unregistered worker {worker_id}")
            return True
        return False

    def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker information"""
        return self.workers.get(worker_id)

    def list_workers(
        self, model_id: Optional[str] = None, state: Optional[WorkerState] = None
    ) -> List[WorkerInfo]:
        """
        List workers with optional filtering.

        Args:
            model_id: Filter by model ID
            state: Filter by worker state

        Returns:
            List of WorkerInfo
        """
        workers = list(self.workers.values())

        if model_id:
            workers = [w for w in workers if w.model_id == model_id]

        if state:
            workers = [w for w in workers if w.state == state]

        return workers

    def get_current_round(self, model_id: str) -> Optional[SyncRound]:
        """Get current synchronization round for a model"""
        return self.current_rounds.get(model_id)

    def get_round_history(
        self, model_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[SyncRound]:
        """
        Get synchronization round history.

        Args:
            model_id: Filter by model ID
            limit: Maximum number of records

        Returns:
            List of SyncRound (sorted by round_id desc)
        """
        history = self.round_history

        if model_id:
            history = [r for r in history if r.model_id == model_id]

        # Sort by round_id descending
        history = sorted(history, key=lambda r: r.round_id, reverse=True)

        if limit:
            history = history[:limit]

        return history

    def add_aggregation_callback(self, callback: Callable[[AggregatedGradient], None]) -> None:
        """Add callback to be called when aggregation completes"""
        self.aggregation_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        total_workers = len(self.workers)
        active_workers = len([w for w in self.workers.values() if w.state == WorkerState.ACTIVE])
        timed_out_workers = len(
            [w for w in self.workers.values() if w.state == WorkerState.TIMED_OUT]
        )

        total_rounds = len(self.round_history)
        completed_rounds = len([r for r in self.round_history if r.completed_at is not None])
        timed_out_rounds = len([r for r in self.round_history if r.timed_out])

        # Calculate average round duration
        avg_duration = 0.0
        if completed_rounds > 0:
            durations = [
                (r.completed_at - r.started_at).total_seconds()
                for r in self.round_history
                if r.completed_at is not None
            ]
            avg_duration = sum(durations) / len(durations)

        return {
            "total_workers": total_workers,
            "active_workers": active_workers,
            "timed_out_workers": timed_out_workers,
            "total_rounds": total_rounds,
            "completed_rounds": completed_rounds,
            "timed_out_rounds": timed_out_rounds,
            "avg_round_duration_seconds": avg_duration,
            "active_rounds": len(self.current_rounds),
        }
