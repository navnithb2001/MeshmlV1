"""
Dataset upload and management endpoints

Endpoints:
- POST /api/datasets/upload - Upload dataset from local files
- POST /api/datasets/from-url - Download dataset from URL
- GET /api/datasets - List uploaded datasets
- GET /api/datasets/{dataset_id} - Get dataset details
- DELETE /api/datasets/{dataset_id} - Delete dataset
"""

import logging
import os
import shutil
import tarfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import asyncio
import boto3
import csv
import json
import openpyxl
import pandas as pd
from botocore.config import Config
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage
from app.clients.dataset_sharder_client import DatasetSharderClient
from app.models.dataset import Dataset  # We'll create this model
from app.models.job import Job
from app.models.user import User
from app.proto import dataset_sharder_pb2
from app.routers.auth import get_current_user
from app.schemas.dataset import (
    DatasetFromURLRequest,
    DatasetListResponse,
    DatasetResponse,
    DatasetUploadResponse,
)
from app.utils.database import AsyncSessionLocal, get_db
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status, Form
from fastapi.responses import JSONResponse
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)
router = APIRouter()

# Storage configuration
UPLOAD_DIR = Path("/tmp/meshml-uploads")  # Temporary upload location
GCS_BUCKET = "meshml-datasets"  # GCS bucket for permanent storage


def _to_dataset_response(dataset: Dataset) -> DatasetResponse:
    return DatasetResponse(
        id=str(dataset.id),
        name=dataset.name,
        format=dataset.format,
        upload_type=dataset.upload_type,
        source_url=dataset.source_url,
        local_path=dataset.local_path,
        gcs_path=dataset.gcs_path,
        total_size_bytes=dataset.total_size_bytes,
        file_count=dataset.file_count,
        num_samples=dataset.num_samples,
        num_classes=dataset.num_classes,
        num_shards=dataset.num_shards,
        shard_strategy=dataset.shard_strategy,
        sharded_at=dataset.sharded_at,
        status=dataset.status,
        error_message=dataset.error_message,
        metadata=dataset.dataset_metadata or {},
        uploaded_by=str(dataset.uploaded_by),
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = Form(None),
    dataset_format: Optional[str] = Form(None),  # imagefolder, coco, csv, auto-detect if None
    shard_strategy: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload dataset from local files

    Supports:
    - Multiple file upload (images, archives, CSV)
    - Automatic format detection
    - Streaming to GCS
    - Background sharding trigger
    """
    logger.info(f"User {current_user.email} uploading dataset with {len(files)} files")

    # Generate dataset ID
    dataset_id = str(uuid.uuid4())
    dataset_name = dataset_name or f"dataset_{dataset_id[:8]}"

    # Create upload directory
    upload_path = UPLOAD_DIR / dataset_id
    upload_path.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded files fast
        total_size = 0
        file_count = 0

        for file in files:
            file_path = upload_path / file.filename

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Fast Write stream to disk
            with open(file_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    total_size += len(chunk)

            file_count += 1
            logger.info(f"Saved file: {file.filename} ({total_size / 1024 / 1024:.2f} MB)")

        # Create database record
        dataset = Dataset(
            id=dataset_id,
            name=dataset_name,
            format=dataset_format or "unknown",
            upload_type="file_upload",
            local_path=str(upload_path),
            gcs_path=f"gs://{GCS_BUCKET}/{dataset_id}/",
            total_size_bytes=total_size,
            file_count=file_count,
            num_samples=0,
            num_classes=0,
            status="processing",
            uploaded_by=current_user.id,
            dataset_metadata={},
        )

        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        logger.info(f"Dataset {dataset_id} uploaded successfully, queueing background processing")

        if background_tasks:
            background_tasks.add_task(
                _process_local_upload_background,
                dataset_id,
                str(upload_path),
                dataset_format,
                shard_strategy,
            )

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            name=dataset_name,
            format=dataset_format or "unknown",
            status="processing",
            total_size_bytes=total_size,
            file_count=file_count,
            num_samples=0,
            num_classes=0,
            local_path=str(upload_path),
            gcs_path=f"gs://{GCS_BUCKET}/{dataset_id}/",
            message="Dataset uploaded successfully. Processing in background.",
            uploaded_at=dataset.created_at.isoformat(),
        )

    except HTTPException:
        # Cleanup on user/input errors
        if upload_path.exists():
            shutil.rmtree(upload_path, ignore_errors=True)
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}", exc_info=True)

        # Cleanup on failure
        if upload_path.exists():
            shutil.rmtree(upload_path, ignore_errors=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset upload failed: {str(e)}",
        )


@router.post(
    "/from-url", response_model=DatasetUploadResponse, status_code=status.HTTP_202_ACCEPTED
)
async def create_dataset_from_url(
    request: DatasetFromURLRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Download dataset from URL (Google Drive, Dropbox, HTTP/HTTPS)

    Supports:
    - Direct download URLs
    - Google Drive share links
    - Dropbox share links
    - Public S3/GCS URLs

    Args:
        request: Dataset URL and configuration
        current_user: Authenticated user
        db: Database session
        background_tasks: FastAPI background tasks

    Returns:
        Dataset upload response (processing starts in background)
    """
    logger.info(f"User {current_user.email} requesting dataset from URL: {request.url}")

    # Generate dataset ID
    dataset_id = str(uuid.uuid4())
    dataset_name = request.name or f"dataset_{dataset_id[:8]}"

    # Parse and validate URL
    download_url = _parse_dataset_url(request.url)

    # Create database record
    dataset = Dataset(
        id=dataset_id,
        name=dataset_name,
        format=request.format or "unknown",
        upload_type="url_download",
        source_url=request.url,
        local_path=f"/tmp/meshml-uploads/{dataset_id}",
        gcs_path=f"gs://{GCS_BUCKET}/{dataset_id}/",
        status="pending",
        uploaded_by=current_user.id,
        dataset_metadata={"original_url": request.url},
    )

    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    logger.info(f"Dataset {dataset_id} download queued")

    # Schedule background download
    background_tasks.add_task(
        _download_from_url_background, dataset_id, download_url, request.format
    )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        name=dataset_name,
        format=request.format or "unknown",
        status="pending",
        message="Dataset download started. Check status with GET /api/datasets/{dataset_id}",
        uploaded_at=dataset.created_at.isoformat(),
    )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    format: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all datasets uploaded by user or group

    Args:
        format: Filter by format (imagefolder, coco, csv)
        status: Filter by status (pending, uploaded, sharding, available, failed)
        current_user: Authenticated user
        db: Database session

    Returns:
        List of datasets
    """
    query = select(Dataset).where(Dataset.uploaded_by == current_user.id)

    if format:
        query = query.where(Dataset.format == format)

    if status:
        query = query.where(Dataset.status == status)

    result = await db.execute(query.order_by(Dataset.created_at.desc()))
    datasets = result.scalars().all()

    logger.info(f"Retrieved {len(datasets)} datasets for user {current_user.email}")

    return DatasetListResponse(
        datasets=[_to_dataset_response(d) for d in datasets], total=len(datasets)
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dataset details by ID

    Args:
        dataset_id: Dataset ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Dataset information
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # Check ownership
    if str(dataset.uploaded_by) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this dataset"
        )

    return _to_dataset_response(dataset)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete dataset and associated files

    Args:
        dataset_id: Dataset ID
        current_user: Authenticated user
        db: Database session
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # Check ownership
    if str(dataset.uploaded_by) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this dataset"
        )

    # Soft-Cancel active dependent jobs before dataset deletion
    await db.execute(
        update(Job)
        .where(Job.dataset_id == dataset_id)
        .where(Job.status.notin_(["completed", "failed"]))
        .values(
            status="failed",
            error_message="Source dataset was deleted by user."
        )
    )
    # Commit the failure state for jobs so they persist even if the file rm operations fail
    await db.commit()

    # Delete local files
    if dataset.local_path and Path(dataset.local_path).exists():
        shutil.rmtree(dataset.local_path, ignore_errors=True)

    # Delete from GCS/MinIO
    if dataset.gcs_path:
        try:

            def _parse_gcs_uri(uri: str) -> tuple[str, str]:
                if not uri.startswith("gs://"):
                    raise ValueError(f"Invalid GCS URI: {uri}")
                without_scheme = uri[5:]
                parts = without_scheme.split("/", 1)
                bucket_name = parts[0]
                prefix = parts[1] if len(parts) > 1 else ""
                if prefix and not prefix.endswith("/"):
                    prefix += "/"
                return bucket_name, prefix

            emulator_url = os.getenv("STORAGE_EMULATOR_URL")
            bucket_name, prefix = _parse_gcs_uri(dataset.gcs_path)

            if emulator_url:
                try:
                    storage_client = storage.Client(
                        credentials=AnonymousCredentials(),
                        project="local",
                        client_options={"api_endpoint": emulator_url},
                    )
                    bucket = storage_client.bucket(bucket_name)
                    blobs = list(bucket.list_blobs(prefix=prefix))
                    for blob in blobs:
                        blob.delete()
                except Exception:
                    client = boto3.client(
                        "s3",
                        endpoint_url=emulator_url,
                        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
                        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
                        region_name="us-east-1",
                        config=Config(signature_version="s3v4"),
                    )
                    objects = [
                        {"Key": obj["Key"]}
                        for obj in client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get(
                            "Contents", []
                        )
                    ]
                    if objects:
                        client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
            else:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blobs = list(bucket.list_blobs(prefix=prefix))
                for blob in blobs:
                    blob.delete()
        except Exception as e:
            logger.warning(f"Failed to delete dataset objects from storage: {e}")

    # Delete database record
    await db.delete(dataset)
    await db.commit()

    # Cleanup data_batches tied to jobs using this dataset
    try:
        await db.execute(
            text(
                "DELETE FROM data_batches "
                "WHERE job_id IN (SELECT id::text FROM jobs WHERE dataset_id = :dataset_id)"
            ),
            {"dataset_id": dataset_id},
        )
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to cleanup data_batches for dataset {dataset_id}: {e}")

    logger.info(f"Deleted dataset {dataset_id}")


# ==================== Helper Functions ====================


def _detect_dataset_format(path: Path) -> Optional[str]:
    """
    Detect dataset format from directory structure

    Returns:
        Dataset format string or None
    """
    # Check for ImageFolder (class subdirectories)
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if subdirs:
        # Check if subdirectories contain images
        for subdir in subdirs[:3]:  # Check first 3 dirs
            image_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
            if image_files:
                return "imagefolder"

    # Check for COCO (annotations directory)
    if (path / "annotations").exists() and (path / "images").exists():
        return "coco"

    # Check for CSV
    csv_files = list(path.glob("*.csv"))
    if csv_files:
        return "csv"

    return None


def _normalize_single_root_directory(path: Path) -> None:
    """
    Flatten extracted archives that contain a single wrapper directory.

    Example:
      /tmp/meshml-uploads/<id>/cifar10_imagefolder/<class dirs...>
      -> /tmp/meshml-uploads/<id>/<class dirs...>
    """
    try:
        entries = list(path.iterdir())
        subdirs = [entry for entry in entries if entry.is_dir()]
        files = [entry for entry in entries if entry.is_file()]

        if files or len(subdirs) != 1:
            return

        wrapper = subdirs[0]
        for child in wrapper.iterdir():
            shutil.move(str(child), str(path / child.name))
        wrapper.rmdir()
        logger.info("Flattened single-root dataset directory: %s", wrapper.name)
    except Exception as e:
        logger.warning("Failed to normalize dataset root at %s: %s", path, e)


def _validate_dataset_structure(path: Path, format: str) -> dict:
    """
    Validate dataset structure based on format

    Returns:
        dict with validation result
    """
    try:
        if format == "imagefolder":
            return _validate_imagefolder(path)
        elif format == "coco":
            return _validate_coco(path)
        elif format == "csv":
            return _validate_csv(path)
        else:
            return {"valid": False, "error": f"Unsupported dataset format: {format}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_imagefolder(path: Path) -> dict:
    """Validate ImageFolder format"""
    subdirs = [d for d in path.iterdir() if d.is_dir()]

    if not subdirs:
        return {"valid": False, "error": "No class directories found"}

    class_names = []
    total_samples = 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    for subdir in subdirs:
        class_names.append(subdir.name)
        images = [f for f in subdir.iterdir() if f.suffix.lower() in image_extensions]
        total_samples += len(images)

    return {
        "valid": True,
        "num_samples": total_samples,
        "num_classes": len(class_names),
        "metadata": {"class_names": class_names, "format": "imagefolder"},
    }


def _validate_coco(path: Path) -> dict:
    """Validate COCO format"""
    annotations_dir = path / "annotations"
    if not annotations_dir.exists():
        return {"valid": False, "error": "No annotations directory found"}

    # Find annotation file
    ann_files = list(annotations_dir.glob("*.json"))
    if not ann_files:
        return {"valid": False, "error": "No annotation JSON files found"}

    # Load first annotation file
    with open(ann_files[0], "r") as f:
        data = json.load(f)

    num_images = len(data.get("images", []))
    num_categories = len(data.get("categories", []))

    return {
        "valid": True,
        "num_samples": num_images,
        "num_classes": num_categories,
        "metadata": {"format": "coco", "annotation_file": ann_files[0].name},
    }


def _validate_csv(path: Path) -> dict:
    """Validate CSV format"""
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        return {"valid": False, "error": "No CSV files found"}

    # Read first CSV to get info
    with open(csv_files[0], "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for _ in reader)

    return {
        "valid": True,
        "num_samples": row_count,
        "num_classes": 0,  # Can't determine from CSV
        "metadata": {"format": "csv", "columns": header, "file": csv_files[0].name},
    }


def _validate_xlsx(path: Path) -> dict:
    """Validate XLSX format"""
    xlsx_files = list(path.glob("*.xlsx")) + list(path.glob("*.xls"))
    if not xlsx_files:
        return {"valid": False, "error": "No XLSX files found"}

    # Read first XLSX file
    df = pd.read_excel(xlsx_files[0])
    row_count = len(df)
    columns = df.columns.tolist()

    return {
        "valid": True,
        "num_samples": row_count,
        "num_classes": 0,  # Can't determine from spreadsheet
        "metadata": {"format": "xlsx", "columns": columns, "file": xlsx_files[0].name},
    }


def _parse_dataset_url(url: str) -> str:
    """
    Parse and convert dataset URLs to direct download links

    Handles:
    - Google Drive: https://drive.google.com/file/d/{id}/view
    - Dropbox: https://www.dropbox.com/s/{id}/file.zip?dl=0
    """
    # Google Drive
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    # Dropbox
    if "dropbox.com" in url:
        return url.replace("?dl=0", "?dl=1")

    # Direct URL
    return url


# ==================== Background Tasks ====================


async def _process_local_upload_background(
    dataset_id: str,
    upload_path_str: str,
    dataset_format: Optional[str],
    shard_strategy: Optional[str],
):
    """Process uploaded dataset in background: extract, validate, and trigger GCS upload"""
    upload_path = Path(upload_path_str)
    try:
        # Check if any file is an archive (zip, tar, tar.gz)
        archive_file = None
        for file_path in upload_path.iterdir():
            if file_path.suffix in [".zip", ".tar", ".gz"]:
                archive_file = file_path
                break

        # Extract archive if found
        if archive_file:
            logger.info(f"Extracting archive in background: {archive_file.name}")
            extract_path = upload_path / "extracted"
            
            def _extract_blocking():
                extract_path.mkdir(exist_ok=True)
                if archive_file.suffix == ".zip":
                    with zipfile.ZipFile(archive_file, "r") as zip_ref:
                        zip_ref.extractall(extract_path)
                elif archive_file.suffix == ".tar" or archive_file.name.endswith(".tar.gz"):
                    with tarfile.open(archive_file, "r:*") as tar_ref:
                        tar_ref.extractall(extract_path)

                # Move extracted contents to main upload path
                for item in extract_path.iterdir():
                    shutil.move(str(item), str(upload_path))

                # Remove archive and extract directory
                archive_file.unlink()
                extract_path.rmdir()

            await asyncio.to_thread(_extract_blocking)

        # Flatten wrapper directory if archive extracted to a single top-level folder.
        _normalize_single_root_directory(upload_path)

        # Detect dataset format if not provided
        if dataset_format is None:
            detected_format = _detect_dataset_format(upload_path)
            if detected_format:
                dataset_format = detected_format
                logger.info(f"Detected dataset format: {dataset_format}")
            else:
                raise ValueError(
                    "Could not detect dataset format. "
                    "Supported formats: imagefolder, coco, csv."
                )

        supported_formats = {"imagefolder", "coco", "csv", "xlsx"}
        if dataset_format not in supported_formats:
            raise ValueError(f"Unsupported dataset format: {dataset_format}")

        # Validate dataset structure
        validation_result = _validate_dataset_structure(upload_path, dataset_format)
        if not validation_result["valid"]:
            raise ValueError(f"Invalid dataset structure: {validation_result['error']}")

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset:
                dataset.format = dataset_format
                dataset.num_samples = validation_result.get("num_samples", 0)
                dataset.num_classes = validation_result.get("num_classes", 0)
                dataset.dataset_metadata = validation_result.get("metadata", {})
                await session.commit()

        # Chain to GCS upload & sharding
        await _upload_to_gcs_background(
            dataset_id=dataset_id,
            local_path=str(upload_path),
            gcs_path=f"gs://{GCS_BUCKET}/{dataset_id}/",
            dataset_format=dataset_format,
            num_shards=None,
            shard_strategy=shard_strategy,
            batch_size=None,
        )

    except Exception as e:
        logger.error(f"Background processing failed for dataset {dataset_id}: {e}", exc_info=True)
        if upload_path.exists():
            shutil.rmtree(upload_path, ignore_errors=True)
        await _mark_dataset_failed(dataset_id, str(e))


async def _upload_to_gcs_background(
    dataset_id: str,
    local_path: str,
    gcs_path: str,
    dataset_format: Optional[str],
    num_shards: Optional[int],
    shard_strategy: Optional[str],
    batch_size: Optional[int],
):
    """Upload dataset to GCS in background and trigger sharding"""
    try:
        logger.info(f"Starting GCS upload for dataset {dataset_id}")

        def _parse_gcs_uri(uri: str) -> tuple[str, str]:
            if not uri.startswith("gs://"):
                raise ValueError(f"Invalid GCS URI: {uri}")
            without_scheme = uri[5:]
            parts = without_scheme.split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            return bucket_name, prefix

        bucket_name, prefix = _parse_gcs_uri(gcs_path)
        local_root = Path(local_path)
        if local_root.is_file():
            files = [local_root]
        else:
            files = [
                Path(root) / filename
                for root, _, filenames in os.walk(local_root)
                for filename in filenames
            ]
        emulator_url = os.getenv("STORAGE_EMULATOR_URL")
        def _blocking_upload():
            if emulator_url:
                # MinIO/S3-compatible upload path for local dev.
                s3_client = boto3.client(
                    "s3",
                    endpoint_url=emulator_url,
                    aws_access_key_id=(
                        os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or "meshml"
                    ),
                    aws_secret_access_key=(
                        os.getenv("AWS_SECRET_ACCESS_KEY")
                        or os.getenv("MINIO_ROOT_PASSWORD")
                        or "meshml_minio_password"
                    ),
                    region_name="us-east-1",
                    config=Config(signature_version="s3v4"),
                )
                try:
                    s3_client.head_bucket(Bucket=bucket_name)
                except Exception:
                    s3_client.create_bucket(Bucket=bucket_name)
                for file_path in files:
                    rel_path = str(file_path.relative_to(local_root)).replace(os.sep, "/")
                    object_key = f"{prefix}{rel_path}" if rel_path else prefix.rstrip("/")
                    s3_client.upload_file(str(file_path), bucket_name, object_key)
            else:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                if not bucket.exists():
                    bucket = storage_client.create_bucket(bucket_name)
                for file_path in files:
                    rel_path = str(file_path.relative_to(local_root)).replace(os.sep, "/")
                    blob_name = f"{prefix}{rel_path}" if rel_path else prefix.rstrip("/")
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))

        await asyncio.to_thread(_blocking_upload)

        logger.info(f"GCS upload completed for dataset {dataset_id}")
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset:
                dataset.status = "uploaded"
                dataset.gcs_path = gcs_path
                await session.commit()
    except Exception as e:
        logger.error(f"GCS upload failed for dataset {dataset_id}: {e}")
        await _mark_dataset_failed(dataset_id, str(e))


async def _mark_dataset_failed(dataset_id: str, message: str) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if dataset:
            dataset.status = "failed"
            dataset.error_message = message[:1000]
            await session.commit()


async def _download_from_url_background(dataset_id: str, url: str, format: str):
    """Download dataset from URL in background"""
    try:
        logger.info(f"Starting download for dataset {dataset_id} from {url}")

        download_path = Path(f"/tmp/meshml-uploads/{dataset_id}")
        download_path.mkdir(parents=True, exist_ok=True)

        # Download file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Download failed with status {response.status}")

                # Save to file
                file_path = download_path / "dataset.zip"
                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

        # Extract if archive
        if file_path.suffix == ".zip":
            def _extract_url_blocking():
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(download_path)
                file_path.unlink()
            
            await asyncio.to_thread(_extract_url_blocking)

        logger.info(f"Download completed for dataset {dataset_id}")

        _normalize_single_root_directory(download_path)

        dataset_format = format
        if not dataset_format or dataset_format == "unknown":
            detected = _detect_dataset_format(download_path)
            dataset_format = detected or "unknown"

        validation_result = _validate_dataset_structure(download_path, dataset_format)
        if not validation_result["valid"]:
            raise Exception(f"Invalid dataset structure: {validation_result.get('error')}")

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
            dataset = result.scalar_one_or_none()
            if dataset:
                dataset.status = "uploaded"
                dataset.format = dataset_format
                dataset.num_samples = validation_result.get("num_samples", 0)
                dataset.num_classes = validation_result.get("num_classes", 0)
                dataset.dataset_metadata = validation_result.get("metadata", {})
                await session.commit()

        await _upload_to_gcs_background(
            dataset_id=dataset_id,
            local_path=str(download_path),
            gcs_path=f"gs://{GCS_BUCKET}/{dataset_id}/",
            dataset_format=dataset_format,
            num_shards=None,
            shard_strategy=None,
            batch_size=None,
        )
    except Exception as e:
        logger.error(f"Download failed for dataset {dataset_id}: {e}")
        await _mark_dataset_failed(dataset_id, str(e))
