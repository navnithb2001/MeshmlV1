from types import SimpleNamespace
from unittest.mock import patch

import grpc
import pytest
from app.grpc_server import DatasetSharderServicer
from app.proto import dataset_sharder_pb2


class FakeContext:
    def __init__(self) -> None:
        self.calls = []

    async def abort(self, code, details):
        self.calls.append((code, details))
        raise RuntimeError(details)


@pytest.mark.asyncio
async def test_shard_dataset_rejects_unsupported_format() -> None:
    with (
        patch("app.grpc_server._get_batch_manager"),
        patch("app.grpc_server.DataDistributor") as mock_distributor,
    ):
        mock_distributor.return_value = SimpleNamespace()

        servicer = DatasetSharderServicer()
        context = FakeContext()
        request = dataset_sharder_pb2.ShardDatasetRequest(
            dataset_id="ds1",
            dataset_path="/tmp/data.tar.gz",
            format="unsupported",
            strategy="stratified",
            num_shards=2,
        )

        with pytest.raises(RuntimeError, match="Unsupported dataset format"):
            await servicer.ShardDataset(request, context)

        assert context.calls[0][0] == grpc.StatusCode.INVALID_ARGUMENT
