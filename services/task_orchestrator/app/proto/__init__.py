"""Generated protocol buffer modules for this component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REQUIRED = [
    "task_orchestrator_pb2.py",
    "task_orchestrator_pb2_grpc.py",
    "dataset_sharder_pb2.py",
    "dataset_sharder_pb2_grpc.py",
    "model_registry_pb2.py",
    "model_registry_pb2_grpc.py",
    "metrics_pb2.py",
    "metrics_pb2_grpc.py",
]


def _ensure_generated() -> None:
    proto_dir = Path(__file__).resolve().parent
    if all((proto_dir / name).exists() for name in _REQUIRED):
        return

    generator = Path(__file__).resolve().parents[2] / "scripts" / "generate_protos.py"
    subprocess.run([sys.executable, str(generator)], check=True)


_ensure_generated()

from . import (
    dataset_sharder_pb2,
    dataset_sharder_pb2_grpc,
    metrics_pb2,
    metrics_pb2_grpc,
    model_registry_pb2,
    model_registry_pb2_grpc,
    task_orchestrator_pb2,
    task_orchestrator_pb2_grpc,
)

__all__ = [
    "task_orchestrator_pb2",
    "task_orchestrator_pb2_grpc",
    "dataset_sharder_pb2",
    "dataset_sharder_pb2_grpc",
    "model_registry_pb2",
    "model_registry_pb2_grpc",
    "metrics_pb2",
    "metrics_pb2_grpc",
]
