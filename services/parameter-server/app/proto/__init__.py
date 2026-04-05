"""Generated protocol buffer modules for this component."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REQUIRED = [
    "parameter_server_pb2.py",
    "parameter_server_pb2_grpc.py",
    "model_registry_pb2.py",
    "model_registry_pb2_grpc.py",
]


def _ensure_generated() -> None:
    proto_dir = Path(__file__).resolve().parent
    if all((proto_dir / name).exists() for name in _REQUIRED):
        return

    generator = Path(__file__).resolve().parents[2] / "scripts" / "generate_protos.py"
    subprocess.run([sys.executable, str(generator)], check=True)


_ensure_generated()

from . import (
    model_registry_pb2,
    model_registry_pb2_grpc,
    parameter_server_pb2,
    parameter_server_pb2_grpc,
)

__all__ = [
    "parameter_server_pb2",
    "parameter_server_pb2_grpc",
    "model_registry_pb2",
    "model_registry_pb2_grpc",
]
