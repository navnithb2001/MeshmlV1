"""Persistence helpers for sharded batch metadata."""

from typing import Any, Iterable

from app.db import AsyncSessionLocal
from sqlalchemy import text


async def persist_batches(job_id: str, model_id: str, batches: Iterable[Any]) -> int:
    """
    Persist generated batches into data_batches table.

    Returns number of rows inserted.
    """
    rows = 0
    async with AsyncSessionLocal() as session:
        for batch in batches:
            await session.execute(
                text(
                    """
                    INSERT INTO data_batches (id, job_id, model_id, gcs_path, status)
                    VALUES (:id, :job_id, :model_id, :gcs_path, 'AVAILABLE')
                    ON CONFLICT (id) DO UPDATE
                    SET
                        job_id = EXCLUDED.job_id,
                        model_id = EXCLUDED.model_id,
                        gcs_path = EXCLUDED.gcs_path,
                        status = 'AVAILABLE',
                        updated_at = NOW()
                    """
                ),
                {
                    "id": batch.batch_id,
                    "job_id": job_id,
                    "model_id": model_id,
                    "gcs_path": batch.storage_path,
                },
            )
            rows += 1
        await session.commit()
    return rows
