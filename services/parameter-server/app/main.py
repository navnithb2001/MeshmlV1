"""
Parameter Server Service - Main Application

Manages model parameters, aggregates gradients, handles synchronization,
and provides parameter distribution for distributed training.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis import Redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

from app.grpc_server import start_grpc_server

# Import routers
from app.routers import convergence, distribution, gradients, models, parameters, synchronization
from app.services.model_registry_client import ModelRegistryClient
from app.services.parameter_storage import ParameterStorageService
from app.services.persistence_loop import PersistenceLoop

# Redis connection
redis_client = None


def _resolve_redis_config() -> tuple[str, int, int]:
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        parsed = urlparse(redis_url)
        host = parsed.hostname or "redis"
        port = parsed.port or 6379
        db = int((parsed.path or "/0").lstrip("/") or 0)
        return host, port, db
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return host, port, db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global redis_client

    # Startup
    logger.info("🚀 Starting Parameter Server Service...")

    try:
        # Initialize Redis connection
        redis_host, redis_port, _redis_db = _resolve_redis_config()

        redis_client = Redis(
            host=redis_host,
            port=redis_port,
            db=_redis_db,
            decode_responses=False,
            socket_connect_timeout=5,
        )

        # Test connection
        redis_client.ping()
        logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")

        # Store Redis client in app state
        app.state.redis = redis_client

    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        logger.warning("⚠️ Starting without Redis - some features may not work")
        app.state.redis = None

    logger.info("✅ Parameter Server Service started successfully")

    grpc_host = os.getenv("GRPC_HOST", "0.0.0.0")
    grpc_port = int(os.getenv("GRPC_PORT", "50054"))
    try:
        await start_grpc_server(app, grpc_host, grpc_port)
        logger.info(f"✅ gRPC server ready on {grpc_host}:{grpc_port}")
    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server: {e}")
        logger.warning("⚠️ gRPC server not available")

    persistence_stop = asyncio.Event()
    redis_host, redis_port, redis_db = _resolve_redis_config()
    persistence = PersistenceLoop(
        storage=ParameterStorageService(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            enable_redis=True,
        ),
        model_registry=ModelRegistryClient(),
        checkpoint_interval=int(os.getenv("CHECKPOINT_INTERVAL", "50")),
        final_version=int(os.getenv("FINAL_MODEL_VERSION", "500")),
        poll_interval=float(os.getenv("PERSISTENCE_POLL_INTERVAL", "5")),
    )
    persistence_task = asyncio.create_task(persistence.run(persistence_stop))
    app.state.persistence_stop = persistence_stop
    app.state.persistence_task = persistence_task

    yield

    # Shutdown
    logger.info("👋 Shutting down Parameter Server Service...")
    stop_event = getattr(app.state, "persistence_stop", None)
    task = getattr(app.state, "persistence_task", None)
    if stop_event:
        stop_event.set()
    if task:
        try:
            await task
        except Exception:
            pass
    grpc_server = getattr(app.state, "grpc_server", None)
    if grpc_server:
        try:
            await grpc_server.stop(grace=1)
            logger.info("✅ gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
    if redis_client:
        try:
            redis_client.close()
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")


# Create FastAPI application
app = FastAPI(
    title="Parameter Server Service",
    description="Model parameter management and gradient aggregation for distributed training",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc), "path": str(request.url)},
    )


# Include routers
app.include_router(gradients.router)
app.include_router(parameters.router)
app.include_router(models.router)
app.include_router(synchronization.router)
app.include_router(convergence.router)
app.include_router(distribution.router)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    redis_status = "healthy" if app.state.redis else "unavailable"

    try:
        if app.state.redis:
            app.state.redis.ping()
            redis_status = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = "unhealthy"

    return {
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "service": "parameter-server",
        "version": "1.0.0",
        "redis": redis_status,
    }


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Parameter Server",
        "version": "1.0.0",
        "description": "Model parameter management and gradient aggregation",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "gradients": "/gradients",
            "parameters": "/parameters",
            "models": "/models",
            "synchronization": "/synchronization",
            "convergence": "/convergence",
            "distribution": "/distribution",
        },
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8003"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
