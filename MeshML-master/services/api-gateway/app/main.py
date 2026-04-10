"""
MeshML API Gateway - Main Application

FastAPI application for:
- Group management
- Job submission and monitoring
- Worker registration
- Authentication
- Real-time metrics
"""

import logging
import time

from app.middleware.security import SecurityHeadersMiddleware
from app.routers import (
    auth,
    datasets,
    groups,
    invitations,
    jobs,
    models,
    monitoring,
    parameters,
    stats_ws,
    workers,
)
from app.utils.database import close_db, init_db
from app.utils.db_init import init_database
from app.utils.redis_client import close_redis, init_redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MeshML API Gateway",
    description="REST API for MeshML distributed training platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(status_code=404, content={"error": "Not found", "path": str(request.url)})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting MeshML API Gateway...")

    # Initialize database
    await init_db()
    logger.info("✓ Database connection initialized")

    # Create tables if they don't exist
    await init_database()
    logger.info("✓ Database schema ready")

    # Initialize Redis
    await init_redis()
    logger.info("✓ Redis connection initialized")

    logger.info("MeshML API Gateway started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    logger.info("Shutting down MeshML API Gateway...")

    await close_db()
    await close_redis()

    logger.info("MeshML API Gateway shut down")


# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(groups.router, prefix="/api/groups", tags=["Groups"])
app.include_router(invitations.router, prefix="/api/invitations", tags=["Invitations"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(workers.router, prefix="/api/workers", tags=["Workers"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(parameters.router, prefix="/api/ps", tags=["Parameter Server"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(stats_ws.router, prefix="/api", tags=["Live Metrics"])


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for load balancers"""
    return {"status": "healthy", "service": "api-gateway", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "MeshML API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
