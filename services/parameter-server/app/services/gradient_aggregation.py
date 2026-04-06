"""
Gradient Aggregation Service for Parameter Server

Implements gradient aggregation strategies for distributed training including
Federated Averaging (FedAvg), asynchronous averaging, staleness-aware weighting,
and gradient clipping/normalization.

Key Features:
- Federated Averaging (FedAvg) implementation
- Asynchronous gradient aggregation
- Staleness-aware weighting based on version IDs
- Gradient clipping (value and norm)
- Gradient normalization
- Multiple aggregation strategies
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class AggregationStrategy(str, Enum):
    """Gradient aggregation strategy"""

    FEDAVG = "fedavg"  # Federated Averaging (weighted by data size)
    SIMPLE_AVERAGE = "simple_average"  # Simple averaging (equal weights)
    WEIGHTED_AVERAGE = "weighted_average"  # Custom weighted averaging
    MOMENTUM = "momentum"  # Momentum-based aggregation
    ADAPTIVE = "adaptive"  # Adaptive aggregation based on gradient quality


class ClippingStrategy(str, Enum):
    """Gradient clipping strategy"""

    NONE = "none"  # No clipping
    VALUE = "value"  # Clip by value
    NORM = "norm"  # Clip by L2 norm


# ==================== Data Classes ====================


@dataclass
class GradientUpdate:
    """Single gradient update from a worker"""

    worker_id: str
    model_id: str
    version_id: int  # Version of parameters this gradient was computed from
    gradients: Dict[str, torch.Tensor]
    num_samples: int  # Number of samples used for this gradient
    loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedGradient:
    """Aggregated gradient result"""

    model_id: str
    target_version_id: int
    aggregated_gradients: Dict[str, torch.Tensor]
    num_workers: int
    total_samples: int
    worker_ids: List[str]
    strategy: AggregationStrategy
    staleness_weights: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationConfig:
    """Configuration for gradient aggregation"""

    strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    clipping_strategy: ClippingStrategy = ClippingStrategy.NONE
    clip_value: float = 1.0  # For value clipping
    clip_norm: float = 1.0  # For norm clipping
    staleness_weight_decay: float = 0.5  # Decay factor for stale gradients
    max_staleness: int = 10  # Maximum allowed staleness (version diff)
    normalize_gradients: bool = False
    momentum_factor: float = 0.9  # For momentum aggregation
    adaptive_threshold: float = 0.1  # For adaptive aggregation


# ==================== Gradient Aggregation Service ====================


class GradientAggregationService:
    """
    Service for aggregating gradients from multiple workers.

    Features:
    - Multiple aggregation strategies (FedAvg, simple average, weighted, etc.)
    - Staleness-aware weighting (older gradients weighted less)
    - Gradient clipping (value and norm based)
    - Gradient normalization
    - Asynchronous gradient buffering
    """

    def __init__(self, default_config: Optional[AggregationConfig] = None):
        """
        Initialize gradient aggregation service.

        Args:
            default_config: Default aggregation configuration
        """
        self.default_config = default_config or AggregationConfig()

        # Buffer for pending gradient updates
        # Key: model_id -> List[GradientUpdate]
        self.gradient_buffer: Dict[str, List[GradientUpdate]] = defaultdict(list)

        # Momentum state for momentum aggregation
        # Key: model_id -> momentum gradients
        self.momentum_state: Dict[str, Dict[str, torch.Tensor]] = {}

        # Aggregation history
        self.aggregation_history: List[AggregatedGradient] = []

        logger.info("GradientAggregationService initialized")

    def submit_gradient(self, gradient_update: GradientUpdate) -> None:
        """
        Submit a gradient update from a worker.

        Args:
            gradient_update: Gradient update to submit
        """
        model_id = gradient_update.model_id
        self.gradient_buffer[model_id].append(gradient_update)

        logger.info(
            f"Received gradient from {gradient_update.worker_id} "
            f"for {model_id} v{gradient_update.version_id} "
            f"({gradient_update.num_samples} samples)"
        )

    def aggregate_gradients(
        self,
        model_id: str,
        current_version: int,
        config: Optional[AggregationConfig] = None,
        clear_buffer: bool = True,
    ) -> Optional[AggregatedGradient]:
        """
        Aggregate pending gradients for a model.

        Args:
            model_id: Model identifier
            current_version: Current version of model parameters
            config: Aggregation configuration (uses default if None)
            clear_buffer: Whether to clear buffer after aggregation

        Returns:
            AggregatedGradient or None if no gradients to aggregate
        """
        if model_id not in self.gradient_buffer or not self.gradient_buffer[model_id]:
            logger.warning(f"No gradients to aggregate for {model_id}")
            return None

        config = config or self.default_config
        gradient_updates = self.gradient_buffer[model_id]

        # Filter by max staleness
        valid_updates = self._filter_by_staleness(
            gradient_updates, current_version, config.max_staleness
        )

        if not valid_updates:
            logger.warning(
                f"All gradients too stale for {model_id} " f"(current version: {current_version})"
            )
            return None

        logger.info(
            f"Aggregating {len(valid_updates)} gradients for {model_id} "
            f"using {config.strategy.value} strategy"
        )

        # Calculate staleness weights
        staleness_weights = self._calculate_staleness_weights(
            valid_updates, current_version, config.staleness_weight_decay
        )

        # Apply gradient clipping if configured
        if config.clipping_strategy != ClippingStrategy.NONE:
            valid_updates = self._clip_gradients(valid_updates, config)

        # Aggregate based on strategy
        if config.strategy == AggregationStrategy.FEDAVG:
            aggregated = self._federated_averaging(valid_updates, staleness_weights)
        elif config.strategy == AggregationStrategy.SIMPLE_AVERAGE:
            aggregated = self._simple_average(valid_updates, staleness_weights)
        elif config.strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            aggregated = self._weighted_average(valid_updates, staleness_weights)
        elif config.strategy == AggregationStrategy.MOMENTUM:
            aggregated = self._momentum_aggregation(
                model_id, valid_updates, staleness_weights, config.momentum_factor
            )
        elif config.strategy == AggregationStrategy.ADAPTIVE:
            aggregated = self._adaptive_aggregation(
                valid_updates, staleness_weights, config.adaptive_threshold
            )
        else:
            raise ValueError(f"Unknown aggregation strategy: {config.strategy}")

        # Normalize if configured
        if config.normalize_gradients:
            aggregated = self._normalize_gradients(aggregated)

        # Create result
        result = AggregatedGradient(
            model_id=model_id,
            target_version_id=current_version,
            aggregated_gradients=aggregated,
            num_workers=len(valid_updates),
            total_samples=sum(u.num_samples for u in valid_updates),
            worker_ids=[u.worker_id for u in valid_updates],
            strategy=config.strategy,
            staleness_weights=staleness_weights,
            created_at=datetime.utcnow(),
            metadata={
                "clipping_strategy": config.clipping_strategy.value,
                "normalized": config.normalize_gradients,
            },
        )

        # Add to history
        self.aggregation_history.append(result)

        # Clear buffer if requested
        if clear_buffer:
            self.gradient_buffer[model_id] = []

        logger.info(
            f"Aggregated gradients for {model_id}: "
            f"{result.num_workers} workers, {result.total_samples} samples"
        )

        return result

    def _filter_by_staleness(
        self, updates: List[GradientUpdate], current_version: int, max_staleness: int
    ) -> List[GradientUpdate]:
        """Filter gradient updates by staleness threshold"""
        valid = []
        for update in updates:
            staleness = current_version - update.version_id
            if staleness <= max_staleness:
                valid.append(update)
            else:
                logger.debug(
                    f"Filtered out stale gradient from {update.worker_id} "
                    f"(staleness: {staleness})"
                )
        return valid

    def _calculate_staleness_weights(
        self, updates: List[GradientUpdate], current_version: int, decay_factor: float
    ) -> Dict[str, float]:
        """
        Calculate staleness-aware weights for gradient updates.

        Formula: weight = decay_factor ^ staleness
        where staleness = current_version - gradient_version
        """
        weights = {}
        for update in updates:
            staleness = current_version - update.version_id
            weight = decay_factor**staleness
            weights[update.worker_id] = weight
        return weights

    def _clip_gradients(
        self, updates: List[GradientUpdate], config: AggregationConfig
    ) -> List[GradientUpdate]:
        """Apply gradient clipping"""
        clipped_updates = []

        for update in updates:
            clipped_grads = {}

            for name, grad in update.gradients.items():
                if config.clipping_strategy == ClippingStrategy.VALUE:
                    # Clip by value
                    clipped_grads[name] = torch.clamp(grad, -config.clip_value, config.clip_value)
                elif config.clipping_strategy == ClippingStrategy.NORM:
                    # Clip by L2 norm
                    grad_norm = torch.norm(grad)
                    if grad_norm > config.clip_norm:
                        clipped_grads[name] = grad * (config.clip_norm / grad_norm)
                    else:
                        clipped_grads[name] = grad

            # Create new update with clipped gradients
            clipped_update = GradientUpdate(
                worker_id=update.worker_id,
                model_id=update.model_id,
                version_id=update.version_id,
                gradients=clipped_grads,
                num_samples=update.num_samples,
                loss=update.loss,
                metrics=update.metrics,
                received_at=update.received_at,
                metadata=update.metadata,
            )
            clipped_updates.append(clipped_update)

        return clipped_updates

    def _federated_averaging(
        self, updates: List[GradientUpdate], staleness_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) implementation.

        Weights gradients by:
        1. Number of samples (data size)
        2. Staleness weight

        Formula: avg_grad = sum(grad_i * samples_i * staleness_i) / sum(samples_i * staleness_i)
        """
        if not updates:
            return {}

        # Get parameter names from first update
        param_names = list(updates[0].gradients.keys())

        # Calculate total weighted samples
        total_weighted_samples = sum(
            u.num_samples * staleness_weights[u.worker_id] for u in updates
        )

        # Aggregate
        aggregated = {}
        for name in param_names:
            weighted_sum = None

            for update in updates:
                if name not in update.gradients:
                    continue

                weight = (
                    update.num_samples * staleness_weights[update.worker_id]
                ) / total_weighted_samples
                weighted_grad = update.gradients[name] * weight

                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum += weighted_grad

            if weighted_sum is not None:
                aggregated[name] = weighted_sum

        return aggregated

    def _simple_average(
        self, updates: List[GradientUpdate], staleness_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Simple averaging with staleness weighting.

        All workers get equal weight (except for staleness adjustment).
        """
        if not updates:
            return {}

        param_names = list(updates[0].gradients.keys())

        # Calculate total staleness weight
        total_weight = sum(staleness_weights[u.worker_id] for u in updates)

        # Aggregate
        aggregated = {}
        for name in param_names:
            weighted_sum = None

            for update in updates:
                if name not in update.gradients:
                    continue

                weight = staleness_weights[update.worker_id] / total_weight
                weighted_grad = update.gradients[name] * weight

                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum += weighted_grad

            if weighted_sum is not None:
                aggregated[name] = weighted_sum

        return aggregated

    def _weighted_average(
        self, updates: List[GradientUpdate], staleness_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted average using custom weights from metadata.

        Falls back to simple average if weights not provided.
        """
        # Get custom weights from metadata or use equal weights
        custom_weights = {}
        for update in updates:
            weight = update.metadata.get("weight", 1.0)
            custom_weights[update.worker_id] = weight * staleness_weights[update.worker_id]

        total_weight = sum(custom_weights.values())

        if not updates:
            return {}

        param_names = list(updates[0].gradients.keys())

        aggregated = {}
        for name in param_names:
            weighted_sum = None

            for update in updates:
                if name not in update.gradients:
                    continue

                weight = custom_weights[update.worker_id] / total_weight
                weighted_grad = update.gradients[name] * weight

                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum += weighted_grad

            if weighted_sum is not None:
                aggregated[name] = weighted_sum

        return aggregated

    def _momentum_aggregation(
        self,
        model_id: str,
        updates: List[GradientUpdate],
        staleness_weights: Dict[str, float],
        momentum_factor: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Momentum-based aggregation.

        Combines current gradients with momentum from previous aggregations.

        Formula: momentum_t = momentum_factor * momentum_{t-1} + (1 - momentum_factor) * grad_t
        """
        # First aggregate current gradients using FedAvg
        current_agg = self._federated_averaging(updates, staleness_weights)

        # Apply momentum
        if model_id not in self.momentum_state:
            # Initialize momentum with current gradients
            self.momentum_state[model_id] = current_agg
            return current_agg

        # Update momentum
        momentum_grads = {}
        for name, grad in current_agg.items():
            if name in self.momentum_state[model_id]:
                momentum_grads[name] = (
                    momentum_factor * self.momentum_state[model_id][name]
                    + (1 - momentum_factor) * grad
                )
            else:
                momentum_grads[name] = grad

        # Update state
        self.momentum_state[model_id] = momentum_grads

        return momentum_grads

    def _adaptive_aggregation(
        self,
        updates: List[GradientUpdate],
        staleness_weights: Dict[str, float],
        quality_threshold: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Adaptive aggregation based on gradient quality.

        Weights gradients by loss improvement (lower loss = higher weight).
        """
        # Calculate quality weights based on loss
        quality_weights = {}

        # Get losses
        losses = [u.loss for u in updates if u.loss is not None]

        if not losses:
            # No loss information, fall back to simple average
            return self._simple_average(updates, staleness_weights)

        # Invert losses (lower loss = higher weight)
        max_loss = max(losses)
        for update in updates:
            if update.loss is not None:
                # Quality = 1 - (loss / max_loss)
                quality = 1.0 - (update.loss / max_loss)
                quality = max(quality, quality_threshold)  # Minimum quality
                quality_weights[update.worker_id] = quality * staleness_weights[update.worker_id]
            else:
                quality_weights[update.worker_id] = staleness_weights[update.worker_id]

        # Aggregate with quality weights
        total_weight = sum(quality_weights.values())

        param_names = list(updates[0].gradients.keys())

        aggregated = {}
        for name in param_names:
            weighted_sum = None

            for update in updates:
                if name not in update.gradients:
                    continue

                weight = quality_weights[update.worker_id] / total_weight
                weighted_grad = update.gradients[name] * weight

                if weighted_sum is None:
                    weighted_sum = weighted_grad
                else:
                    weighted_sum += weighted_grad

            if weighted_sum is not None:
                aggregated[name] = weighted_sum

        return aggregated

    def _normalize_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalize gradients by L2 norm.

        Ensures all gradients have unit norm.
        """
        normalized = {}
        for name, grad in gradients.items():
            grad_norm = torch.norm(grad)
            if grad_norm > 0:
                normalized[name] = grad / grad_norm
            else:
                normalized[name] = grad
        return normalized

    def get_pending_gradients(self, model_id: str) -> List[GradientUpdate]:
        """
        Get pending gradient updates for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of pending gradient updates
        """
        return self.gradient_buffer.get(model_id, [])

    def clear_buffer(self, model_id: str) -> int:
        """
        Clear gradient buffer for a model.

        Args:
            model_id: Model identifier

        Returns:
            Number of gradients cleared
        """
        count = len(self.gradient_buffer.get(model_id, []))
        self.gradient_buffer[model_id] = []
        logger.info(f"Cleared {count} gradients from buffer for {model_id}")
        return count

    def get_aggregation_history(
        self, model_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[AggregatedGradient]:
        """
        Get aggregation history.

        Args:
            model_id: Filter by model ID (None = all)
            limit: Maximum number of records (None = all)

        Returns:
            List of aggregated gradients (sorted by created_at desc)
        """
        history = self.aggregation_history

        # Filter by model
        if model_id:
            history = [h for h in history if h.model_id == model_id]

        # Sort by created_at descending
        history = sorted(history, key=lambda h: h.created_at, reverse=True)

        # Limit
        if limit:
            history = history[:limit]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        total_aggregations = len(self.aggregation_history)

        # Count by strategy
        strategy_counts = {}
        for agg in self.aggregation_history:
            strategy = agg.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Pending gradients
        pending_gradients = sum(len(grads) for grads in self.gradient_buffer.values())

        models_with_pending = len(
            [model_id for model_id, grads in self.gradient_buffer.items() if grads]
        )

        return {
            "total_aggregations": total_aggregations,
            "strategy_counts": strategy_counts,
            "pending_gradients": pending_gradients,
            "models_with_pending": models_with_pending,
            "models_with_momentum": len(self.momentum_state),
        }
