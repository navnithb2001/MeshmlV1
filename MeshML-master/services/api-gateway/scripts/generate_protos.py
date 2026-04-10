#!/usr/bin/env python3
"""Generate gRPC Python stubs for this component."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / "proto"
OUT_DIR = ROOT / "app/proto"
PROTO_FILES = [
    PROTO_DIR / "task_orchestrator.proto",
    PROTO_DIR / "model_registry.proto",
    PROTO_DIR / "dataset_sharder.proto",
]


def _run_protoc() -> None:
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        *[str(proto_file) for proto_file in PROTO_FILES],
    ]
    subprocess.run(cmd, check=True)


def _fix_grpc_imports() -> None:
    pattern = re.compile(r"^import (\w+_pb2) as (.+)$", re.MULTILINE)
    for grpc_file in OUT_DIR.glob("*_pb2_grpc.py"):
        content = grpc_file.read_text()
        updated = pattern.sub(r"from . import \1 as \2", content)
        if updated != content:
            grpc_file.write_text(updated)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_protoc()
    _fix_grpc_imports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
