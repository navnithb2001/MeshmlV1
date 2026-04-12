"""
Metrics Service - Main Application

Receives training metrics over gRPC, publishes to Redis, and persists to PostgreSQL.
"""

import asyncio
import logging
import os
import sys

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import Base, engine
from app.grpc_server import start_grpc_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MeshML Metrics Service",
    description="Real-time training metrics ingestion service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "metrics-service"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url)},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting MeshML Metrics Service...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        redis_password = os.getenv("REDIS_PASSWORD", "")
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        if redis_password:
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
        else:
            redis_url = f"redis://{redis_host}:{redis_port}/0"

    app.state.redis_client = redis.from_url(redis_url, decode_responses=False)
    try:
        await app.state.redis_client.ping()
        logger.info("✓ Redis connection initialized")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        app.state.redis_client = None

    grpc_host = os.getenv("GRPC_HOST", "0.0.0.0")
    grpc_port = int(os.getenv("GRPC_PORT", "50055"))
    try:
        await start_grpc_server(app, grpc_host, grpc_port)
        logger.info(f"✅ gRPC server ready on {grpc_host}:{grpc_port}")
    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server: {e}")
        logger.warning("⚠️ gRPC server not available")

    progress_stop = asyncio.Event()
    app.state.progress_stop = progress_stop
    app.state.progress_task = asyncio.create_task(_progress_bridge(progress_stop))

    logger.info("✅ Metrics Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Metrics Service...")
    grpc_server = getattr(app.state, "grpc_server", None)
    if grpc_server:
        try:
            await grpc_server.stop(grace=1)
            logger.info("✅ gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
    redis_client = getattr(app.state, "redis_client", None)
    if redis_client:
        await redis_client.close()
    stop_event = getattr(app.state, "progress_stop", None)
    task = getattr(app.state, "progress_task", None)
    if stop_event:
        stop_event.set()
    if task:
        try:
            await task
        except Exception:
            pass


async def _progress_bridge(stop_event: asyncio.Event, interval_seconds: int = 7) -> None:
    """
    Periodically aggregate batch completion and update jobs.progress.
    """
    from app.db import AsyncSessionLocal
    from sqlalchemy import text

    while not stop_event.is_set():
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    text(
                        "SELECT job_id, COUNT(*) AS total_batches, "
                        "SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_batches "
                        "FROM data_batches GROUP BY job_id"
                    )
                )
                rows = result.fetchall()
                for row in rows:
                    job_id = row[0]
                    total_batches = int(row[1] or 0)
                    completed_batches = int(row[2] or 0)
                    await session.execute(
                        text(
                            "UPDATE jobs "
                            "SET progress = jsonb_set("
                            "  jsonb_set(COALESCE(progress, '{}'::jsonb), '{current_batch}', to_jsonb(:completed), true),"
                            "  '{total_batches}', to_jsonb(:total), true"
                            ") "
                            "WHERE id::text = :job_id"
                        ),
                        {
                            "completed": completed_batches,
                            "total": total_batches,
                            "job_id": str(job_id),
                        },
                    )
                await session.commit()
        except Exception as e:
            logger.error(f"Progress bridge error: {e}")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info",
    )
