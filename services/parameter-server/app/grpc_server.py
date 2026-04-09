"""gRPC server for Parameter Server."""

import asyncio
import gzip
import io
import logging
import math
import os
import pickle
from typing import Optional

import grpc
import numpy as np
import torch
from app.proto import parameter_server_pb2, parameter_server_pb2_grpc
from app.services.gradient_aggregation import GradientAggregationService, GradientUpdate
from app.services.parameter_storage import ParameterStorageService

logger = logging.getLogger(__name__)


def _decompress(data: bytes, compression_type: str) -> bytes:
    if compression_type == "gzip":
        return gzip.decompress(data)
    return data


def _compress(data: bytes, compression_type: str) -> bytes:
    if compression_type == "gzip":
        return gzip.compress(data)
    return data


class ParameterServerServicer(parameter_server_pb2_grpc.ParameterServerServicer):
    def __init__(self):
        self.storage = ParameterStorageService()
        self.aggregator = GradientAggregationService()
        self.optimizer_type = os.getenv("PARAMETER_OPTIMIZER", "sgd").lower()
        self.momentum = float(os.getenv("SGD_MOMENTUM", "0.9"))
        self.staleness_lambda = float(os.getenv("STALENESS_LAMBDA", "0.3"))

    def _load_tensor_dict(self, data: bytes) -> dict:
        if not data:
            return {}
        buffer = io.BytesIO(data)
        arrays = np.load(buffer, allow_pickle=False)
        return {name: torch.from_numpy(arrays[name]) for name in arrays.files}

    def _dump_tensor_dict(self, data: dict) -> bytes:
        buffer = io.BytesIO()
        arrays = {}
        for name, value in data.items():
            if isinstance(value, torch.Tensor):
                arrays[name] = value.detach().cpu().numpy()
            else:
                arrays[name] = np.asarray(value)
        np.savez(buffer, **arrays)
        return buffer.getvalue()

    def _get_learning_rate(self, model_id: str, fallback: float) -> float:
        if not self.storage.enable_redis or not self.storage.redis_client:
            return fallback
        key = f"lr:{model_id}"
        raw = self.storage.redis_client.get(key)
        if raw is None:
            return fallback
        try:
            return float(raw)
        except Exception:
            return fallback

    def _get_current_version(self, model_id: str) -> int:
        if not self.storage.enable_redis or not self.storage.redis_client:
            return self.storage.get_current_version(model_id) or 0
        key = f"params:{model_id}:current_version"
        raw = self.storage.redis_client.get(key)
        if raw is None:
            return 0
        try:
            return int(raw)
        except Exception:
            return 0

    def _load_current_weights(self, model_id: str, version_id: int) -> Optional[dict]:
        if self.storage.enable_redis and self.storage.redis_client and version_id > 0:
            data = self.storage.redis_client.get(f"params:{model_id}:v{version_id}")
            if data:
                return self._load_tensor_dict(data)
        return None

    def _load_optimizer_state(self, model_id: str) -> dict:
        if not self.storage.enable_redis or not self.storage.redis_client:
            return {}
        key = f"optim:{model_id}"
        data = self.storage.redis_client.get(key)
        if not data:
            return {}
        try:
            return pickle.loads(data)
        except Exception:
            return {}

    def _save_optimizer_state(self, model_id: str, state: dict) -> None:
        if not self.storage.enable_redis or not self.storage.redis_client:
            return
        key = f"optim:{model_id}"
        self.storage.redis_client.set(key, pickle.dumps(state))

    async def PullWeights(self, request, context):
        try:
            model_id = request.job_id
            params, current_version = self.storage.get_latest_parameters(model_id)
            if params is None:
                return parameter_server_pb2.WeightsResponse(
                    model_state_dict=b"",
                    version=0,
                    epoch=request.epoch,
                    is_updated=False,
                    compression_type="none",
                    uncompressed_size=0,
                )

            payload = self._dump_tensor_dict(params)
            compressed = _compress(payload, "gzip")
            current_version = current_version or 0
            is_updated = current_version > request.current_version

            return parameter_server_pb2.WeightsResponse(
                model_state_dict=compressed,
                version=current_version,
                epoch=request.epoch,
                is_updated=is_updated,
                compression_type="gzip",
                uncompressed_size=len(payload),
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def PushGradients(self, request, context):
        try:
            if not self.storage.enable_redis or not self.storage.redis_client:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Redis not available")

            payload = _decompress(request.gradients, request.compression_type)
            gradients = self._load_tensor_dict(payload) if payload else {}

            lock = self.storage.redis_client.lock(
                f"lock:params:{request.job_id}", timeout=10, blocking_timeout=10
            )
            if not lock.acquire(blocking=True):
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Lock busy")
            try:
                current_version = self._get_current_version(request.job_id)
                params = self._load_current_weights(request.job_id, current_version)
                if params is None:
                    return parameter_server_pb2.GradientsAck(
                        success=False,
                        message="No parameters available for model",
                        new_version=current_version,
                        staleness_accepted=False,
                        staleness_threshold=0,
                    )

                staleness = max(0, current_version - request.version)
                staleness_weight = math.exp(-self.staleness_lambda * staleness)
                lr = self._get_learning_rate(request.job_id, request.learning_rate)
                effective_lr = lr * staleness_weight

                momentum_key = f"momentum:{request.job_id}"
                momentum_state = {}
                raw_momentum = self.storage.redis_client.get(momentum_key)
                if raw_momentum:
                    try:
                        momentum_state = self._load_tensor_dict(raw_momentum)
                    except Exception:
                        momentum_state = {}

                for name, param in params.items():
                    grad = gradients.get(name)
                    if grad is None:
                        continue
                    if not isinstance(grad, torch.Tensor):
                        grad = torch.tensor(grad)
                    grad_t = grad.detach().clone()

                    buf = momentum_state.get(name)
                    if buf is None:
                        buf = torch.zeros_like(grad_t)
                    buf = self.momentum * buf + grad_t
                    momentum_state[name] = buf
                    params[name] = param - effective_lr * buf

                new_version = current_version + 1
                self.storage._persist_to_redis(request.job_id, new_version, params)
                self.storage.parameters[request.job_id] = params
                self.storage.current_versions[request.job_id] = new_version
                self.storage.redis_client.set(momentum_key, self._dump_tensor_dict(momentum_state))

                current_version = new_version
            finally:
                try:
                    lock.release()
                except Exception:
                    pass

            return parameter_server_pb2.GradientsAck(
                success=True,
                message="accepted",
                new_version=current_version,
                staleness_accepted=staleness > 0,
                staleness_threshold=10,
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetOptimizerState(self, request, context):
        try:
            return parameter_server_pb2.OptimizerStateResponse(
                optimizer_state=b"",
                optimizer_type="adam",
                version=self.storage.get_current_version(request.job_id) or 0,
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetModelVersion(self, request, context):
        try:
            current_version = self.storage.get_current_version(request.job_id) or 0
            return parameter_server_pb2.VersionResponse(
                current_version=current_version,
                epoch=0,
                total_updates=0,
                last_update_timestamp=int(asyncio.get_event_loop().time()),
            )
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def create_grpc_services() -> ParameterServerServicer:
    return ParameterServerServicer()


async def start_grpc_server(app, host: str, port: int) -> None:
    server = grpc.aio.server()
    servicer = create_grpc_services()
    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    app.state.grpc_server = server
    logger.info(f"Parameter Server gRPC started on {host}:{port}")
