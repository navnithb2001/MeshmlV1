"""
Models router - Core model CRUD operations
Implements TASK-11.1 (storage) and TASK-11.2 (lifecycle management)
"""

import logging
import os
from typing import Optional
from uuid import UUID

import boto3
from botocore.config import Config
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db_session
from ..lifecycle import LifecycleManager
from ..models import Model, ModelState
from ..schemas import ModelCreate, ModelResponse, ModelStateUpdate, ModelUpdate, UploadUrlResponse
from ..storage import GCSClient

router = APIRouter()
logger = logging.getLogger(__name__)

# Test user UUID (in production, this would come from authentication)
TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def get_gcs_client(request: Request) -> GCSClient:
    """Dependency to get GCS client"""
    return request.app.state.gcs_client


@router.post("/", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def create_model(model_data: ModelCreate, db: AsyncSession = Depends(get_db_session)):
    """
    Create a new model entry
    State will be UPLOADING until file is uploaded
    """
    # Create model record
    model = Model(
        name=model_data.name,
        description=model_data.description,
        group_id=model_data.group_id,
        created_by_user_id=TEST_USER_ID,  # Using test user UUID
        architecture_type=model_data.architecture_type,
        dataset_type=model_data.dataset_type,
        model_metadata=model_data.metadata,
        version=model_data.version,
        parent_model_id=model_data.parent_model_id,
        state=ModelState.UPLOADING,
    )

    db.add(model)
    await db.commit()
    await db.refresh(model)

    logger.info(f"Created model {model.id}: {model.name}")

    return model


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int, db: AsyncSession = Depends(get_db_session)):
    """Get model by ID"""
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    return model


@router.put("/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: int, model_data: ModelUpdate, db: AsyncSession = Depends(get_db_session)
):
    """Update model metadata"""
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    # Update fields
    update_data = model_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(model, field, value)

    await db.commit()
    await db.refresh(model)

    logger.info(f"Updated model {model_id}")

    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db_session),
    gcs: GCSClient = Depends(get_gcs_client),
):
    """
    Delete a model (soft delete by marking as DEPRECATED)
    Also deletes file from GCS
    """
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    # Deprecate model using lifecycle manager
    lifecycle = LifecycleManager(db)
    try:
        await lifecycle.deprecate(model_id, reason="Model deleted by user")
    except ValueError as e:
        # Already deprecated or invalid transition
        pass

    # Delete file from GCS
    if model.gcs_path:
        try:
            await gcs.delete_file(model_id)
            logger.info(f"Deleted GCS file for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to delete GCS file for model {model_id}: {e}")

    logger.info(f"Deleted model {model_id}")


@router.post("/{model_id}/upload-url", response_model=UploadUrlResponse)
async def get_upload_url(
    model_id: int,
    db: AsyncSession = Depends(get_db_session),
    gcs: GCSClient = Depends(get_gcs_client),
):
    """
    Generate signed URL for direct file upload to GCS
    Client uploads directly to this URL, then calls /complete-upload
    """
    # Verify model exists and is in UPLOADING state
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    if model.state != ModelState.UPLOADING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model must be in UPLOADING state, currently {model.state}",
        )

    # Generate signed URL
    upload_url = await gcs.generate_upload_url(model_id)
    gcs_path = gcs.get_gcs_uri(gcs.get_model_path(model_id))

    return UploadUrlResponse(
        upload_url=upload_url, model_id=model_id, gcs_path=gcs_path, expires_in_seconds=3600
    )


@router.post("/{model_id}/upload", response_model=ModelResponse)
async def upload_model_file(
    model_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    gcs: GCSClient = Depends(get_gcs_client),
):
    """
    Direct file upload endpoint (alternative to signed URL)
    Uploads file and transitions to VALIDATING state
    """
    # Verify model exists
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    # Validate file
    if not file.filename.endswith(".py"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only .py files are allowed"
        )

    # Read file content
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)

    if file_size_mb > settings.MAX_MODEL_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size_mb:.2f}MB (max {settings.MAX_MODEL_SIZE_MB}MB)",
        )

    # Upload to GCS
    gcs_path, file_size, file_hash = await gcs.upload_file(model_id, content, file.filename)

    # Update model record
    model.gcs_path = gcs_path
    model.file_size_bytes = file_size
    model.file_hash = file_hash

    # Transition to VALIDATING
    lifecycle = LifecycleManager(db)
    await lifecycle.start_validation(model_id)

    await db.commit()
    await db.refresh(model)

    logger.info(f"Uploaded file for model {model_id}: {gcs_path}")

    return model


@router.get("/{model_id}/download")
async def download_model_file(
    model_id: int,
    db: AsyncSession = Depends(get_db_session),
    gcs: GCSClient = Depends(get_gcs_client),
):
    """
    Download model file
    Returns signed URL for direct download
    """
    # Verify model exists and is ready
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    if model.state != ModelState.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model not ready for download (state: {model.state})",
        )

    # Generate download URL
    download_url = await gcs.generate_download_url(model_id)

    # Increment download count
    model.download_count += 1
    await db.commit()

    logger.info(f"Generated download URL for model {model_id}")

    return {
        "download_url": download_url,
        "model_id": model_id,
        "filename": "model.py",
        "expires_in_seconds": 3600,
    }


@router.get("/{model_id}/checkpoints/{version}")
async def download_checkpoint(
    model_id: int,
    version: str,
    db: AsyncSession = Depends(get_db_session),
    gcs: GCSClient = Depends(get_gcs_client),
):
    """
    Download a specific checkpoint version.
    Returns signed URL for direct download.
    """
    result = await db.execute(select(Model).where(Model.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model {model_id} not found"
        )

    version_tag = version if str(version).startswith("v") else f"v{version}"
    checkpoint_key = f"checkpoints/{model_id}/{version_tag}.pt"

    emulator_url = os.getenv("STORAGE_EMULATOR_URL")
    if emulator_url:
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        client = boto3.client(
            "s3",
            endpoint_url=emulator_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        bucket = settings.GCS_BUCKET_NAME
        try:
            client.head_object(Bucket=bucket, Key=checkpoint_key)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        download_url = client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": checkpoint_key}, ExpiresIn=3600
        )
    else:
        if not gcs or not gcs.bucket:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Storage not available"
            )
        blob = gcs.bucket.blob(checkpoint_key)
        if not blob.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Checkpoint not found"
            )
        download_url = blob.generate_signed_url(version="v4", expiration=3600, method="GET")

    return {
        "model_id": model_id,
        "checkpoint_version": version_tag,
        "download_url": download_url,
        "expires_in_seconds": 3600,
    }


@router.post("/{model_id}/state", response_model=ModelResponse)
async def update_model_state(
    model_id: int, state_update: ModelStateUpdate, db: AsyncSession = Depends(get_db_session)
):
    """
    Update model state (lifecycle transition)
    Used by validation service to mark models as READY or FAILED
    """
    lifecycle = LifecycleManager(db)

    try:
        model = await lifecycle.transition_state(
            model_id, state_update.state, state_update.validation_message
        )

        logger.info(f"Model {model_id} state updated to {state_update.state}")

        return model

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
