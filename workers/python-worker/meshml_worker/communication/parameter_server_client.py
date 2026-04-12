"""Parameter Server gRPC client wrapper used by Trainer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

from .grpc_client import GRPCClient


class ParameterServerClient:
    """Adapter exposing Trainer-compatible methods over gRPC transport."""

    def __init__(self, grpc_url: str):
        self._client = GRPCClient(SimpleNamespace(grpc_url=grpc_url))

    def connect(self) -> bool:
        self._client.connect()
        return True

    def disconnect(self) -> None:
        self._client.disconnect()

    def register_worker(
        self, worker_id: str, model_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Worker registration is handled by Task Orchestrator stream.
        return {
            "worker_id": worker_id,
            "model_id": model_id,
            "metadata": metadata or {},
            "status": "ok",
        }

    def get_model_version(self, model_id: str) -> int:
        version = self._client.get_model_version(model_id)
        return int(version.get("current_version", 0))

    def get_weights(self, model_id: str, version_id: Optional[int] = None) -> Dict[str, Any]:
        state_dict, pulled_version = self._client.get_weights(
            job_id=model_id,
            worker_id="worker",
            epoch=0,
        )
        if version_id is not None and pulled_version < version_id:
            return {}
        return state_dict

    def push_gradients(
        self,
        worker_id: str,
        model_id: str,
        version_id: int,
        gradients: Dict[str, Any],
        num_samples: int,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload_metrics = dict(metrics or {})
        if loss is not None:
            payload_metrics["loss"] = float(loss)
        return self._client.push_gradients(
            job_id=model_id,
            worker_id=worker_id,
            gradients=gradients,
            batch_id=0,
            epoch=0,
            batch_size=num_samples,
            learning_rate=float(payload_metrics.get("learning_rate", 0.001)),
            metadata=payload_metrics,
        )
