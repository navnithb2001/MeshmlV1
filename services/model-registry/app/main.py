"""
MeshML Model Registry Service

FastAPI application for model lifecycle management, storage, and discovery.
Handles model upload, validation, versioning, and search functionality.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .database import create_tables, get_db_session
from .grpc_server import start_grpc_server
from .routers import lifecycle, models, search, versions
from .storage.gcs_client import GCSClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Starting MeshML Model Registry Service...")

    # Create database tables
    await create_tables()
    logger.info("✅ Database tables verified")

    # Initialize GCS client (optional - will use local storage if GCS unavailable)
    try:
        gcs_client = GCSClient()
        await gcs_client.initialize()
        app.state.gcs_client = gcs_client
        logger.info(f"✅ GCS bucket '{settings.GCS_BUCKET_NAME}' ready")
    except Exception as e:
        logger.warning(f"⚠️  GCS not available (will use local storage): {e}")
        app.state.gcs_client = None

    grpc_host = settings.GRPC_HOST if hasattr(settings, "GRPC_HOST") else "0.0.0.0"
    grpc_port = int(os.getenv("GRPC_PORT", "50052"))
    try:
        await start_grpc_server(app, grpc_host, grpc_port)
        logger.info(f"✅ gRPC server ready on {grpc_host}:{grpc_port}")
    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server: {e}")
        logger.warning("⚠️ gRPC server not available")

    logger.info("✨ Model Registry Service ready!")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Model Registry Service...")
    grpc_server = getattr(app.state, "grpc_server", None)
    if grpc_server:
        try:
            await grpc_server.stop(grace=1)
            logger.info("✅ gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")


# Create FastAPI app
app = FastAPI(
    title="MeshML Model Registry",
    description="Model lifecycle management and discovery service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
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
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        },
    )


# Include routers
app.include_router(models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(lifecycle.router, prefix="/api/v1/lifecycle", tags=["Lifecycle"])
app.include_router(versions.router, prefix="/api/v1/versions", tags=["Versions"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {"service": "MeshML Model Registry", "version": "1.0.0", "status": "operational"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "model-registry",
        "database": "connected",
        "storage": "ready",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
