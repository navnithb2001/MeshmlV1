"""
Task Orchestrator Service - Main Application

Coordinates distributed training tasks, manages worker registration,
handles job queuing and assignment, and provides fault tolerance.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

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

# Import gRPC server
from app.grpc_server import start_grpc_server

# Redis connection
redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global redis_client

    # Startup
    logger.info("🚀 Starting Task Orchestrator Service...")

    try:
        # Initialize Redis connection
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        redis_client = Redis(
            host=redis_host, port=redis_port, decode_responses=False, socket_connect_timeout=5
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

    grpc_host = os.getenv("GRPC_HOST", "0.0.0.0")
    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    try:
        await start_grpc_server(app, grpc_host, grpc_port)
        logger.info(f"✅ gRPC server ready on {grpc_host}:{grpc_port}")
    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server: {e}")
        logger.warning("⚠️ gRPC server not available - workers cannot connect")

    logger.info("✅ Task Orchestrator Service started successfully")

    yield

    # Shutdown
    logger.info("👋 Shutting down Task Orchestrator Service...")
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
    title="Task Orchestrator Service",
    description="Distributed training task coordination and worker management",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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
        "service": "task-orchestrator",
        "version": "1.0.0",
        "redis": redis_status,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8002"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")
