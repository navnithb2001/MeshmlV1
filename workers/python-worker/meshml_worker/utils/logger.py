"""
Logging utilities for MeshML Worker
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors

        Args:
            record: Log record

        Returns:
            Formatted string
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logger(
    name: str = "meshml_worker",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    colored: bool = True,
) -> logging.Logger:
    """Setup logger with console and optional file handlers

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        colored: Use colored output for console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if colored:
        console_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """Logger for training metrics and progress"""

    def __init__(self, log_dir: Path, model_id: str):
        """Initialize training logger

        Args:
            log_dir: Directory for training logs
            model_id: Model ID
        """
        self.log_dir = log_dir
        self.model_id = model_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{model_id}_{timestamp}.log"

        self.logger = setup_logger(name=f"training.{model_id}", log_file=self.log_file)

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        accuracy: Optional[float] = None,
        other_metrics: Optional[dict] = None,
    ) -> None:
        """Log epoch metrics

        Args:
            epoch: Epoch number
            loss: Training loss
            accuracy: Optional accuracy
            other_metrics: Other metrics to log
        """
        msg = f"Epoch {epoch}: loss={loss:.4f}"

        if accuracy is not None:
            msg += f", accuracy={accuracy:.4f}"

        if other_metrics:
            for key, value in other_metrics.items():
                msg += f", {key}={value:.4f}"

        self.logger.info(msg)

    def log_iteration(self, iteration: int, loss: float, lr: Optional[float] = None) -> None:
        """Log iteration metrics

        Args:
            iteration: Iteration number
            loss: Batch loss
            lr: Learning rate
        """
        msg = f"Iteration {iteration}: loss={loss:.4f}"

        if lr is not None:
            msg += f", lr={lr:.6f}"

        self.logger.debug(msg)

    def log_checkpoint(self, epoch: int, path: Path) -> None:
        """Log checkpoint save

        Args:
            epoch: Epoch number
            path: Checkpoint path
        """
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {path}")

    def log_convergence(self, converged: bool, reason: str) -> None:
        """Log convergence status

        Args:
            converged: Whether training converged
            reason: Convergence reason
        """
        if converged:
            self.logger.info(f"Training converged: {reason}")
        else:
            self.logger.info(f"Training stopped: {reason}")
