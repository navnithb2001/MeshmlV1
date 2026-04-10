"""
Parameter Server Proxy Router

Proxies requests from external workers to the internal Parameter Server HTTP API.
This allows workers to communicate via HTTP through the exposed API Gateway
without requiring gRPC port exposure.
"""

import logging
import os
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

logger = logging.getLogger(__name__)

router = APIRouter()

# Parameter Server internal service URL (Kubernetes DNS)
PARAMETER_SERVER_URL = os.getenv(
    "PARAMETER_SERVER_URL", "http://parameter-server.meshml.svc.cluster.local:8003"
)

# HTTP client for proxying requests
http_client = httpx.AsyncClient(timeout=30.0)


@router.get("/parameters/{model_id}")
async def get_parameters(model_id: str, version_id: int = None):
    """
    Get current parameters for a model

    Proxies to Parameter Server: GET /parameters/{model_id}
    """
    try:
        url = f"{PARAMETER_SERVER_URL}/parameters/{model_id}"
        params = {"version_id": version_id} if version_id else {}

        logger.info(f"Proxying GET parameters request for model {model_id}")

        response = await http_client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Parameter server returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error proxying to parameter server: {e}")
        raise HTTPException(status_code=503, detail="Parameter server unavailable")


@router.get("/parameters/{model_id}/version")
async def get_parameter_version(model_id: str):
    """
    Get current version ID for a model

    Proxies to Parameter Server: GET /parameters/{model_id}/version
    """
    try:
        url = f"{PARAMETER_SERVER_URL}/parameters/{model_id}/version"

        logger.info(f"Proxying GET version request for model {model_id}")

        response = await http_client.get(url)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Parameter server returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error proxying to parameter server: {e}")
        raise HTTPException(status_code=503, detail="Parameter server unavailable")


@router.put("/parameters/{model_id}/learning-rate")
async def update_learning_rate(model_id: str, request: Request):
    """
    Update learning rate for a model in Parameter Server

    Proxies to Parameter Server: PUT /parameters/{model_id}/learning-rate
    """
    try:
        body = await request.json()
        url = f"{PARAMETER_SERVER_URL}/parameters/{model_id}/learning-rate"

        logger.info(f"Proxying learning rate update for model {model_id}")

        response = await http_client.put(url, json=body)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Parameter server returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error proxying to parameter server: {e}")
        raise HTTPException(status_code=503, detail="Parameter server unavailable")


@router.post("/gradients/submit")
async def submit_gradients(request: Request):
    """
    Submit gradients from a worker

    Proxies to Parameter Server: POST /gradients/submit
    """
    try:
        body = await request.json()
        url = f"{PARAMETER_SERVER_URL}/gradients/submit"

        logger.info(f"Proxying gradient submission from worker {body.get('worker_id')}")

        response = await http_client.post(url, json=body)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Parameter server returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error proxying to parameter server: {e}")
        raise HTTPException(status_code=503, detail="Parameter server unavailable")


@router.get("/gradients/pending/{model_id}")
async def get_pending_gradients(model_id: str):
    """
    Get pending gradients for a model

    Proxies to Parameter Server: GET /gradients/pending/{model_id}
    """
    try:
        url = f"{PARAMETER_SERVER_URL}/gradients/pending/{model_id}"

        logger.info(f"Proxying GET pending gradients for model {model_id}")

        response = await http_client.get(url)
        response.raise_for_status()

        return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(f"Parameter server returned error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error(f"Error proxying to parameter server: {e}")
        raise HTTPException(status_code=503, detail="Parameter server unavailable")


@router.on_event("shutdown")
async def shutdown_http_client():
    """Close HTTP client on shutdown"""
    await http_client.aclose()
