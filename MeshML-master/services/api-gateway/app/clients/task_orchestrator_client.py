"""Task Orchestrator gRPC client for job submission."""

import logging
import os
from typing import Optional

import grpc
from app.proto import task_orchestrator_pb2, task_orchestrator_pb2_grpc

logger = logging.getLogger(__name__)


class TaskOrchestratorClient:
    """Minimal async gRPC client for Task Orchestrator."""

    def __init__(self, grpc_url: Optional[str] = None):
        self.grpc_url = grpc_url or os.getenv(
            "TASK_ORCHESTRATOR_GRPC_URL", "task-orchestrator-service:50051"
        )

    async def initiate_training(
        self, submission: task_orchestrator_pb2.JobSubmission
    ) -> task_orchestrator_pb2.JobSubmissionAck:
        async with grpc.aio.insecure_channel(self.grpc_url) as channel:
            stub = task_orchestrator_pb2_grpc.TaskOrchestratorStub(channel)
            return await stub.InitiateTraining(submission)
