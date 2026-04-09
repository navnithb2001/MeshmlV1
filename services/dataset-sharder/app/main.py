"""
Dataset Sharder Service - Main Application

FastAPI application for dataset sharding and distribution:
- Dataset loading (ImageFolder, COCO, CSV)
- Sharding algorithms (Random, Stratified, Non-IID, Sequential)
- Batch storage (Local + GCS)
- Data distribution to workers
"""

import logging
import os
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.grpc_server import start_grpc_server
from app.routers import distribution, sharding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="MeshML Dataset Sharder",
    description="Dataset sharding and distribution service for distributed ML training",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(distribution.router, prefix="/api/v1", tags=["distribution"])
app.include_router(sharding.router, prefix="/api/v1", tags=["sharding"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MeshML Dataset Sharder",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dataset-sharder"}


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(status_code=404, content={"error": "Not found", "path": str(request.url)})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting MeshML Dataset Sharder Service...")

    # Create storage directories
    os.makedirs(settings.LOCAL_STORAGE_PATH, exist_ok=True)
    logger.info(f"✓ Local storage ready: {settings.LOCAL_STORAGE_PATH}")

    grpc_host = os.getenv("GRPC_HOST", "0.0.0.0")
    grpc_port = int(os.getenv("GRPC_PORT", "50053"))
    try:
        await start_grpc_server(app, grpc_host, grpc_port)
        logger.info(f"✅ gRPC server ready on {grpc_host}:{grpc_port}")
    except Exception as e:
        logger.error(f"❌ Failed to start gRPC server: {e}")
        logger.warning("⚠️ gRPC server not available")

    logger.info("✅ Dataset Sharder Service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Dataset Sharder Service...")
    grpc_server = getattr(app.state, "grpc_server", None)
    if grpc_server:
        try:
            await grpc_server.stop(grace=1)
            logger.info("✅ gRPC server stopped")
        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")
