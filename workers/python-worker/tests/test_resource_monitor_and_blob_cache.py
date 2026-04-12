import asyncio
from pathlib import Path

import pytest
from meshml_worker.main import BlobCache, ResourceMonitor


@pytest.mark.asyncio
async def test_blob_cache_uses_cached_file_when_hash_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "model.py"
    file_path.write_text("print('ok')")

    cache = BlobCache(tmp_path / "cache.json")
    expected_hash = await cache.hash_file(file_path)
    await cache._write_cache({"model:1": {"url": "u", "sha256": expected_hash}})

    should_use = await cache.should_use_cached("model:1", file_path, expected_hash)
    assert should_use is True


@pytest.mark.asyncio
async def test_blob_cache_skips_cache_when_hash_mismatch(tmp_path: Path) -> None:
    file_path = tmp_path / "model.py"
    file_path.write_text("print('ok')")

    cache = BlobCache(tmp_path / "cache.json")
    await cache._write_cache({"model:1": {"url": "u", "sha256": "wrong"}})

    should_use = await cache.should_use_cached("model:1", file_path, "wrong")
    assert should_use is False


@pytest.mark.asyncio
async def test_resource_monitor_pauses_on_high_usage(monkeypatch) -> None:
    pause_event = asyncio.Event()

    monkeypatch.setattr("meshml_worker.main.psutil.cpu_percent", lambda interval=0.1: 91.0)
    monkeypatch.setattr(
        "meshml_worker.main.psutil.virtual_memory",
        lambda: type("Vm", (), {"percent": 40.0})(),
    )

    monitor = ResourceMonitor(pause_event=pause_event, cpu_threshold=80.0, memory_threshold=80.0)
    await monitor.check_once()

    assert pause_event.is_set() is True


@pytest.mark.asyncio
async def test_resource_monitor_resumes_when_usage_normal(monkeypatch) -> None:
    pause_event = asyncio.Event()
    pause_event.set()

    monkeypatch.setattr("meshml_worker.main.psutil.cpu_percent", lambda interval=0.1: 10.0)
    monkeypatch.setattr(
        "meshml_worker.main.psutil.virtual_memory",
        lambda: type("Vm", (), {"percent": 20.0})(),
    )

    monitor = ResourceMonitor(pause_event=pause_event, cpu_threshold=80.0, memory_threshold=80.0)
    await monitor.check_once()

    assert pause_event.is_set() is False
