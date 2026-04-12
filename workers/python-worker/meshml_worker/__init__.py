"""
MeshML Python Worker

Federated learning worker for distributed training.
"""

__version__ = "0.1.0"
__author__ = "MeshML Team"

from meshml_worker.config import WorkerConfig
from meshml_worker.main import MeshMLWorker
from meshml_worker.utils.optimization import (
    MemoryProfiler,
    OptimizedDataLoader,
    PerformanceBenchmark,
    optimize_dataloader_settings,
)

__all__ = [
    "WorkerConfig",
    "MeshMLWorker",
    "MemoryProfiler",
    "PerformanceBenchmark",
    "OptimizedDataLoader",
    "optimize_dataloader_settings",
]
