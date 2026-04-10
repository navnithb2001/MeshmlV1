"""Redis client for caching and real-time data"""

import logging
import os

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Redis URL from environment
# Try to get REDIS_URL first, otherwise build from components
REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    # Build from individual components
    redis_password = os.getenv("REDIS_PASSWORD", "")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")

    # Include password if provided (format: redis://:password@host:port/db)
    if redis_password:
        REDIS_URL = f"redis://:{redis_password}@{redis_host}:{redis_port}/0"
    else:
        REDIS_URL = f"redis://{redis_host}:{redis_port}/0"

logger.info(f"Connecting to Redis at: {REDIS_URL.split('@')[1] if '@' in REDIS_URL else REDIS_URL}")

# Global Redis client
redis_client: redis.Redis = None


async def init_redis():
    """Initialize Redis connection"""
    global redis_client

    try:
        redis_client = redis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True, max_connections=50
        )

        # Test connection
        await redis_client.ping()
        logger.info("Redis connection established")

    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise


async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


def get_redis() -> redis.Redis:
    """
    Dependency for getting Redis client

    Usage:
        @app.get("/cache")
        async def get_cache(redis: Redis = Depends(get_redis)):
            ...
    """
    return redis_client
