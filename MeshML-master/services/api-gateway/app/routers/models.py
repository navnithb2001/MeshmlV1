"""
Model upload endpoints for API Gateway.

Implements user-facing model upload and registry registration.
"""

import ast
import hashlib
import logging
import os
from typing import Any, Dict, Optional

import httpx
from app.clients.model_registry_client import ModelRegistryClient
from app.models.user import User
from app.proto import model_registry_pb2
from app.routers.auth import get_current_user
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Supported contract values ───────────────────────────────────────────────

SUPPORTED_TASK_TYPES = {"classification", "regression", "binary"}
SUPPORTED_LOSSES = {"cross_entropy", "mse", "mae", "bce_with_logits", "bce"}


def _validate_model_python_source(content: bytes) -> Dict[str, Any]:
    """
    Static checks for uploaded model source.

    Returns:
        Extracted MODEL_METADATA dict (after validation).

    Raises:
        HTTPException 422 on any validation failure with a clear message.
    """
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model file must be UTF-8 encoded Python source.",
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Python syntax error at line {exc.lineno}: {exc.msg}",
        )

    has_create_model = False
    metadata_node: Optional[ast.Assign] = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "create_model":
            has_create_model = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_METADATA":
                    metadata_node = node

    if not has_create_model:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model code must define create_model().",
        )
    if metadata_node is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model code must define MODEL_METADATA.",
        )

    # ── Statically evaluate MODEL_METADATA ───────────────────────────────────
    try:
        metadata: Dict[str, Any] = ast.literal_eval(metadata_node.value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "MODEL_METADATA must be a plain Python dict literal "
                "(no function calls or variables)."
            ),
        )

    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="MODEL_METADATA must be a dict.",
        )

    # ── Validate existing base fields ─────────────────────────────────────────
    base_required = ["name", "version", "framework", "input_shape", "output_shape"]
    missing_base = [f for f in base_required if f not in metadata]
    if missing_base:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"MODEL_METADATA missing required base fields: {missing_base}",
        )

    # ── Validate new contract fields ──────────────────────────────────────────
    contract_required = ["task_type", "loss", "metrics"]
    missing_contract = [f for f in contract_required if f not in metadata]
    if missing_contract:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"MODEL_METADATA missing required contract fields: {missing_contract}. "
                f"These must be defined in your model.py — the model dictates the math."
            ),
        )

    task_type = metadata["task_type"]
    loss = metadata["loss"]
    metrics = metadata["metrics"]

    if task_type not in SUPPORTED_TASK_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"MODEL_METADATA.task_type '{task_type}' is not supported. "
                f"Supported values: {sorted(SUPPORTED_TASK_TYPES)}"
            ),
        )

    if loss not in SUPPORTED_LOSSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"MODEL_METADATA.loss '{loss}' is not supported. "
                f"Supported values: {sorted(SUPPORTED_LOSSES)}"
            ),
        )

    if not isinstance(metrics, list) or len(metrics) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="MODEL_METADATA.metrics must be a non-empty list of strings.",
        )
    invalid_metrics = [m for m in metrics if not isinstance(m, str)]
    if invalid_metrics:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"MODEL_METADATA.metrics entries must be strings, got: {invalid_metrics}",
        )

    logger.info(
        f"Model metadata validated: task_type={task_type}, loss={loss}, metrics={metrics}"
    )
    return metadata


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    group_id: str = Form(...),
    description: Optional[str] = Form(None),
    architecture_type: Optional[str] = Form(None),
    dataset_type: Optional[str] = Form(None),
    version: Optional[str] = Form("1.0.0"),
    current_user: User = Depends(get_current_user),
):
    """
    Upload model.py via API Gateway.

    Flow:
    1. Static validation of MODEL_METADATA contract
    2. Register model in Model Registry (gRPC), passing task_type via metadata
    3. Upload file to signed URL (HTTP PUT)
    4. Finalize upload (gRPC)
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Missing filename")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate and extract MODEL_METADATA
    model_metadata = _validate_model_python_source(content)

    file_hash = hashlib.sha256(content).hexdigest()
    file_size = len(content)

    # Build the metadata map to pass to Model Registry — store contract fields
    registry_metadata = {
        "task_type": str(model_metadata.get("task_type", "")),
        "loss": str(model_metadata.get("loss", "")),
        "metrics": ",".join(model_metadata.get("metrics", [])),
    }
    # Use task_type as architecture_type if not provided explicitly
    effective_arch = architecture_type or str(model_metadata.get("task_type", ""))

    client = ModelRegistryClient()
    try:
        registration = await client.register_new_model(
            model_registry_pb2.RegisterModelRequest(
                name=name,
                description=description or "",
                group_id=group_id,
                created_by_user_id=str(current_user.id),
                architecture_type=effective_arch,
                dataset_type=dataset_type or "",
                version=version or "1.0.0",
                metadata=registry_metadata,
            )
        )
    except Exception as e:
        logger.error(f"Model registration failed: {e}")
        raise HTTPException(status_code=502, detail="Model registry registration failed")

    try:
        async with httpx.AsyncClient(timeout=60) as http_client:
            put_resp = await http_client.put(
                registration.upload_url, content=content, headers={"Content-Type": "text/x-python"}
            )
            if put_resp.status_code not in (200, 201):
                raise HTTPException(
                    status_code=502, detail=f"Signed upload failed: {put_resp.status_code}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload to signed URL failed: {e}")
        raise HTTPException(status_code=502, detail="Signed upload failed")

    try:
        await client.finalize_model_upload(
            model_registry_pb2.FinalizeModelUploadRequest(
                model_id=registration.model_id,
                gcs_path=registration.gcs_path,
                file_size_bytes=file_size,
                file_hash=file_hash,
            )
        )
    except Exception as e:
        logger.error(f"Finalize model upload failed: {e}")
        raise HTTPException(status_code=502, detail="Finalize upload failed")

    return {
        "model_id": registration.model_id,
        "gcs_path": registration.gcs_path,
        "file_size_bytes": file_size,
        "file_hash": file_hash,
        "task_type": model_metadata.get("task_type"),
        "loss": model_metadata.get("loss"),
        "metrics": model_metadata.get("metrics"),
    }




@router.get("/{model_id}/download")
async def download_final_model(model_id: int, current_user: User = Depends(get_current_user)):
    """
    Get signed download URL for final model artifact.
    """
    client = ModelRegistryClient()
    response = await client.get_final_model_download_url(model_id=model_id)
    if not response.found:
        raise HTTPException(status_code=404, detail="Final model not found")
    return {
        "model_id": model_id,
        "download_url": response.download_url,
        "storage_path": response.storage_path,
        "expires_in_seconds": response.expires_in_seconds,
    }


@router.get("/{model_id}/checkpoints/{version}")
async def download_checkpoint(
    model_id: int, version: str, current_user: User = Depends(get_current_user)
):
    """
    Get signed download URL for a specific checkpoint version.
    """
    base_url = os.getenv("MODEL_REGISTRY_URL", "http://model-registry:8004")
    url = f"{base_url}/api/v1/models/{model_id}/checkpoints/{version}"
    try:
        async with httpx.AsyncClient(timeout=30) as http_client:
            resp = await http_client.get(url)
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Checkpoint not found")
            if resp.status_code >= 400:
                raise HTTPException(status_code=502, detail="Model registry error")
            return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Checkpoint download lookup failed: {e}")
        raise HTTPException(status_code=502, detail="Model registry unavailable")
