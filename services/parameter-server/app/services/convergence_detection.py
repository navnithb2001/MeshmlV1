"""
Convergence Detection Service for Parameter Server

Implements convergence detection and early stopping mechanisms:
- Loss monitoring and trend analysis
- Early stopping based on configurable criteria
- Target accuracy/metric validation
- Patience-based stopping
- Plateau detection
- Training state management

Key Features:
- Multiple convergence criteria
- Sliding window analysis
- Metric tracking (loss, accuracy, custom metrics)
- Automatic early stopping recommendations
- Training phase management
- Convergence history and reporting
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==================== Enums ====================


class TrainingPhase(str, Enum):
    """Training phase"""

    NOT_STARTED = "not_started"
    WARMUP = "warmup"  # Initial warmup period
    TRAINING = "training"  # Active training
    PLATEAUED = "plateaued"  # Training has plateaued
    CONVERGED = "converged"  # Training converged successfully
    STOPPED = "stopped"  # Training stopped (early stopping or error)


class ConvergenceCriterion(str, Enum):
    """Convergence detection criterion"""

    LOSS_THRESHOLD = "loss_threshold"  # Loss below threshold
    LOSS_PLATEAU = "loss_plateau"  # Loss stopped improving
    METRIC_THRESHOLD = "metric_threshold"  # Metric above/below threshold
    METRIC_PLATEAU = "metric_plateau"  # Metric stopped improving
    PATIENCE = "patience"  # No improvement for N iterations
    MAX_ITERATIONS = "max_iterations"  # Maximum iterations reached
    GRADIENT_NORM = "gradient_norm"  # Gradient norm below threshold


class MetricDirection(str, Enum):
    """Direction for metric optimization"""

    MINIMIZE = "minimize"  # Lower is better (e.g., loss)
    MAXIMIZE = "maximize"  # Higher is better (e.g., accuracy)


# ==================== Data Classes ====================


@dataclass
class TrainingMetrics:
    """Metrics from a training iteration"""

    iteration: int
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    num_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection"""

    # Loss-based criteria
    loss_threshold: Optional[float] = None  # Stop when loss < threshold
    loss_patience: int = 10  # Iterations without loss improvement
    loss_min_delta: float = 1e-4  # Minimum improvement to count

    # Metric-based criteria
    target_metrics: Dict[str, Tuple[float, MetricDirection]] = field(default_factory=dict)
    # e.g., {"accuracy": (0.95, MetricDirection.MAXIMIZE)}
    metric_patience: int = 10
    metric_min_delta: float = 1e-4

    # Plateau detection
    enable_plateau_detection: bool = True
    plateau_patience: int = 20  # Iterations on plateau before stopping
    plateau_threshold: float = 1e-3  # Max variance to consider plateau

    # Gradient-based
    gradient_norm_threshold: Optional[float] = None  # Stop if gradient too small

    # General
    max_iterations: Optional[int] = None
    warmup_iterations: int = 5  # Ignore first N iterations
    window_size: int = 10  # Size of sliding window for analysis

    # Early stopping
    enable_early_stopping: bool = True
    early_stop_patience: int = 50  # Iterations without any improvement


@dataclass
class ConvergenceResult:
    """Result of convergence detection"""

    converged: bool
    should_stop: bool
    phase: TrainingPhase
    criteria_met: List[ConvergenceCriterion]
    current_iteration: int
    best_iteration: int
    best_loss: float
    best_metrics: Dict[str, float]
    iterations_without_improvement: int
    estimated_iterations_remaining: Optional[int] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Convergence Detection Service ====================


class ConvergenceDetectionService:
    """
    Service for detecting training convergence and recommending early stopping.

    Features:
    - Multi-criteria convergence detection
    - Loss and metric tracking
    - Plateau detection
    - Patience-based early stopping
    - Training phase management
    - Convergence history
    """

    def __init__(self, default_config: Optional[ConvergenceConfig] = None):
        """
        Initialize convergence detection service.

        Args:
            default_config: Default convergence configuration
        """
        self.default_config = default_config or ConvergenceConfig()

        # Per-model training state
        # Key: model_id -> training state
        self.training_states: Dict[str, Dict[str, Any]] = {}

        # Convergence history
        self.convergence_history: List[ConvergenceResult] = []

        logger.info("ConvergenceDetectionService initialized")

    def update_metrics(
        self, model_id: str, metrics: TrainingMetrics, config: Optional[ConvergenceConfig] = None
    ) -> ConvergenceResult:
        """
        Update training metrics and check convergence.

        Args:
            model_id: Model identifier
            metrics: Training metrics from current iteration
            config: Convergence configuration (uses default if None)

        Returns:
            ConvergenceResult with current status
        """
        config = config or self.default_config

        # Initialize state if needed
        if model_id not in self.training_states:
            self._initialize_state(model_id)

        state = self.training_states[model_id]

        # Add metrics to history
        state["metrics_history"].append(metrics)
        state["current_iteration"] = metrics.iteration

        # Update phase
        self._update_training_phase(model_id, config)

        # Check convergence criteria
        result = self._check_convergence(model_id, config)

        # Update best metrics
        self._update_best_metrics(model_id, metrics, config)

        # Add to history
        self.convergence_history.append(result)

        # Log status
        if result.should_stop:
            logger.info(
                f"Training should stop for {model_id}: {result.message} "
                f"(iteration {result.current_iteration})"
            )
        elif result.converged:
            logger.info(
                f"Training converged for {model_id}: {result.message} "
                f"(iteration {result.current_iteration})"
            )

        return result

    def _initialize_state(self, model_id: str) -> None:
        """Initialize training state for a model"""
        self.training_states[model_id] = {
            "metrics_history": deque(maxlen=1000),  # Keep last 1000 iterations
            "current_iteration": 0,
            "phase": TrainingPhase.NOT_STARTED,
            "best_loss": float("inf"),
            "best_metrics": {},
            "best_iteration": 0,
            "iterations_without_improvement": 0,
            "plateau_start_iteration": None,
        }
        logger.info(f"Initialized convergence tracking for {model_id}")

    def _update_training_phase(self, model_id: str, config: ConvergenceConfig) -> None:
        """Update training phase based on current state"""
        state = self.training_states[model_id]
        current_phase = state["phase"]
        iteration = state["current_iteration"]

        # NOT_STARTED -> WARMUP
        if current_phase == TrainingPhase.NOT_STARTED:
            state["phase"] = TrainingPhase.WARMUP
            return

        # WARMUP -> TRAINING
        if current_phase == TrainingPhase.WARMUP:
            if iteration >= config.warmup_iterations:
                state["phase"] = TrainingPhase.TRAINING
            return

        # TRAINING -> PLATEAUED
        if current_phase == TrainingPhase.TRAINING:
            if self._is_plateaued(model_id, config):
                state["phase"] = TrainingPhase.PLATEAUED
                state["plateau_start_iteration"] = iteration
            return

    def _check_convergence(self, model_id: str, config: ConvergenceConfig) -> ConvergenceResult:
        """Check all convergence criteria"""
        state = self.training_states[model_id]
        phase = state["phase"]
        iteration = state["current_iteration"]

        # Don't check during warmup
        if phase == TrainingPhase.WARMUP:
            return ConvergenceResult(
                converged=False,
                should_stop=False,
                phase=phase,
                criteria_met=[],
                current_iteration=iteration,
                best_iteration=state["best_iteration"],
                best_loss=state["best_loss"],
                best_metrics=state["best_metrics"],
                iterations_without_improvement=state["iterations_without_improvement"],
                message="Warmup phase",
            )

        criteria_met = []
        converged = False
        should_stop = False
        message = "Training in progress"

        # Get recent metrics
        if not state["metrics_history"]:
            return ConvergenceResult(
                converged=False,
                should_stop=False,
                phase=phase,
                criteria_met=[],
                current_iteration=iteration,
                best_iteration=state["best_iteration"],
                best_loss=state["best_loss"],
                best_metrics=state["best_metrics"],
                iterations_without_improvement=state["iterations_without_improvement"],
                message="No metrics yet",
            )

        current_metrics = state["metrics_history"][-1]

        # 1. Check max iterations
        if config.max_iterations and iteration >= config.max_iterations:
            criteria_met.append(ConvergenceCriterion.MAX_ITERATIONS)
            should_stop = True
            message = f"Maximum iterations ({config.max_iterations}) reached"

        # 2. Check loss threshold
        if config.loss_threshold and current_metrics.loss <= config.loss_threshold:
            criteria_met.append(ConvergenceCriterion.LOSS_THRESHOLD)
            converged = True
            message = f"Loss threshold met ({current_metrics.loss:.6f} <= {config.loss_threshold})"

        # 3. Check target metrics
        for metric_name, (target_value, direction) in config.target_metrics.items():
            if metric_name in current_metrics.metrics:
                metric_value = current_metrics.metrics[metric_name]

                if direction == MetricDirection.MAXIMIZE:
                    if metric_value >= target_value:
                        criteria_met.append(ConvergenceCriterion.METRIC_THRESHOLD)
                        converged = True
                        message = (
                            f"Target {metric_name} reached ({metric_value:.4f} >= {target_value})"
                        )
                elif direction == MetricDirection.MINIMIZE:
                    if metric_value <= target_value:
                        criteria_met.append(ConvergenceCriterion.METRIC_THRESHOLD)
                        converged = True
                        message = (
                            f"Target {metric_name} reached ({metric_value:.4f} <= {target_value})"
                        )

        # 4. Check gradient norm
        if config.gradient_norm_threshold and current_metrics.gradient_norm is not None:
            if current_metrics.gradient_norm < config.gradient_norm_threshold:
                criteria_met.append(ConvergenceCriterion.GRADIENT_NORM)
                converged = True
                message = f"Gradient norm below threshold ({current_metrics.gradient_norm:.6f})"

        # 5. Check loss plateau (patience-based)
        if config.enable_plateau_detection and phase == TrainingPhase.PLATEAUED:
            plateau_duration = iteration - state["plateau_start_iteration"]
            if plateau_duration >= config.plateau_patience:
                criteria_met.append(ConvergenceCriterion.LOSS_PLATEAU)
                should_stop = True
                message = f"Training plateaued for {plateau_duration} iterations"

        # 6. Check early stopping (no improvement)
        if config.enable_early_stopping:
            if state["iterations_without_improvement"] >= config.early_stop_patience:
                criteria_met.append(ConvergenceCriterion.PATIENCE)
                should_stop = True
                message = f"No improvement for {state['iterations_without_improvement']} iterations"

        # Update phase if converged or stopped
        if converged:
            state["phase"] = TrainingPhase.CONVERGED
        elif should_stop:
            state["phase"] = TrainingPhase.STOPPED

        # Estimate remaining iterations
        estimated_remaining = None
        if config.max_iterations:
            estimated_remaining = config.max_iterations - iteration

        return ConvergenceResult(
            converged=converged,
            should_stop=should_stop or converged,
            phase=state["phase"],
            criteria_met=criteria_met,
            current_iteration=iteration,
            best_iteration=state["best_iteration"],
            best_loss=state["best_loss"],
            best_metrics=state["best_metrics"],
            iterations_without_improvement=state["iterations_without_improvement"],
            estimated_iterations_remaining=estimated_remaining,
            message=message,
        )

    def _update_best_metrics(
        self, model_id: str, metrics: TrainingMetrics, config: ConvergenceConfig
    ) -> None:
        """Update best loss and metrics"""
        state = self.training_states[model_id]

        # Check if loss improved
        loss_improved = False
        if metrics.loss < state["best_loss"] - config.loss_min_delta:
            state["best_loss"] = metrics.loss
            state["best_iteration"] = metrics.iteration
            loss_improved = True

        # Check if metrics improved
        metrics_improved = False
        for metric_name, value in metrics.metrics.items():
            if metric_name in config.target_metrics:
                target_value, direction = config.target_metrics[metric_name]

                if metric_name not in state["best_metrics"]:
                    state["best_metrics"][metric_name] = value
                    metrics_improved = True
                else:
                    best_value = state["best_metrics"][metric_name]

                    if direction == MetricDirection.MAXIMIZE:
                        if value > best_value + config.metric_min_delta:
                            state["best_metrics"][metric_name] = value
                            metrics_improved = True
                    elif direction == MetricDirection.MINIMIZE:
                        if value < best_value - config.metric_min_delta:
                            state["best_metrics"][metric_name] = value
                            metrics_improved = True

        # Update iterations without improvement
        if loss_improved or metrics_improved:
            state["iterations_without_improvement"] = 0
        else:
            state["iterations_without_improvement"] += 1

    def _is_plateaued(self, model_id: str, config: ConvergenceConfig) -> bool:
        """Check if training has plateaued"""
        if not config.enable_plateau_detection:
            return False

        state = self.training_states[model_id]
        history = state["metrics_history"]

        # Need enough data
        if len(history) < config.window_size:
            return False

        # Get recent losses
        recent_losses = [m.loss for m in list(history)[-config.window_size :]]

        # Calculate variance
        variance = np.var(recent_losses)

        # Plateau if variance is very small
        return variance < config.plateau_threshold

    def get_training_state(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get current training state for a model"""
        return self.training_states.get(model_id)

    def get_metrics_history(
        self, model_id: str, limit: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """
        Get metrics history for a model.

        Args:
            model_id: Model identifier
            limit: Maximum number of records (None = all)

        Returns:
            List of TrainingMetrics (newest first)
        """
        if model_id not in self.training_states:
            return []

        history = list(self.training_states[model_id]["metrics_history"])
        history.reverse()  # Newest first

        if limit:
            history = history[:limit]

        return history

    def get_convergence_summary(self, model_id: str) -> Dict[str, Any]:
        """Get convergence summary for a model"""
        if model_id not in self.training_states:
            return {"model_id": model_id, "status": "not_started"}

        state = self.training_states[model_id]
        history = list(state["metrics_history"])

        if not history:
            return {"model_id": model_id, "status": "initialized", "phase": state["phase"].value}

        current_metrics = history[-1]

        # Calculate improvement rate
        loss_history = (
            [m.loss for m in history[-10:]] if len(history) >= 10 else [m.loss for m in history]
        )
        improvement_rate = 0.0
        if len(loss_history) > 1:
            improvement_rate = (loss_history[0] - loss_history[-1]) / len(loss_history)

        return {
            "model_id": model_id,
            "phase": state["phase"].value,
            "current_iteration": state["current_iteration"],
            "best_iteration": state["best_iteration"],
            "current_loss": current_metrics.loss,
            "best_loss": state["best_loss"],
            "best_metrics": state["best_metrics"],
            "iterations_without_improvement": state["iterations_without_improvement"],
            "total_iterations": len(history),
            "improvement_rate": improvement_rate,
            "is_plateaued": state["phase"] == TrainingPhase.PLATEAUED,
        }

    def reset_training(self, model_id: str) -> bool:
        """
        Reset training state for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if reset, False if model not found
        """
        if model_id not in self.training_states:
            return False

        del self.training_states[model_id]
        logger.info(f"Reset convergence tracking for {model_id}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get convergence detection statistics"""
        total_models = len(self.training_states)

        # Count by phase
        phase_counts = {}
        for state in self.training_states.values():
            phase = state["phase"].value
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Recent convergence results
        recent_results = self.convergence_history[-100:] if self.convergence_history else []
        converged_count = len([r for r in recent_results if r.converged])
        stopped_count = len([r for r in recent_results if r.should_stop and not r.converged])

        return {
            "total_models_tracked": total_models,
            "phase_distribution": phase_counts,
            "recent_convergences": converged_count,
            "recent_early_stops": stopped_count,
            "total_convergence_checks": len(self.convergence_history),
        }
