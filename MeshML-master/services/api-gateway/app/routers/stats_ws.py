"""WebSocket endpoint for live job metrics."""

import asyncio
import json
import logging
import time
import uuid

import redis.asyncio as redis
from app.models.job import Job
from app.utils.database import AsyncSessionLocal
from app.utils.redis_client import get_redis
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}/stats")
async def job_stats_ws(
    websocket: WebSocket,
    job_id: str,
    redis_client: redis.Redis = Depends(get_redis),
):
    await websocket.accept()
    if redis_client is None:
        await websocket.close(code=1011)
        return
    channel = f"live_stats:{job_id}"
    pubsub = None
    last_status = None
    next_status_check = 0.0

    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message.get("data") is not None:
                try:
                    payload = json.loads(message["data"])
                    await websocket.send_json(payload)
                except Exception:
                    await websocket.send_text(message["data"].decode("utf-8"))
            now = time.monotonic()
            if now >= next_status_check:
                next_status_check = now + 2.0
                try:
                    job_uuid = uuid.UUID(job_id)
                except Exception:
                    job_uuid = None
                if job_uuid:
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(select(Job.status).where(Job.id == job_uuid))
                        status = result.scalar_one_or_none()
                else:
                    status = None
                if status and status != last_status:
                    last_status = status
                    if str(status).upper() == "COMPLETED":
                        await websocket.send_json(
                            {
                                "type": "JOB_STATUS_CHANGE",
                                "job_id": job_id,
                                "status": "COMPLETED",
                            }
                        )
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception:
                pass
