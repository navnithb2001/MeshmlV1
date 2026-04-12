"""Dataset loading utilities for various ML dataset formats.

This module provides loaders for common ML dataset formats with support for
large file streaming to avoid out-of-memory errors.
"""

import csv
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import boto3
import numpy as np
from app.core.storage import get_dataset_storage
from botocore.config import Config
from PIL import Image

logger = logging.getLogger(__name__)


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid gs URI: {gs_uri}")
    stripped = gs_uri[5:]
    parts = stripped.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _emulator_s3_client():
    endpoint = os.getenv("STORAGE_EMULATOR_URL")
    if not endpoint:
        return None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
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


def _list_gs_objects(gs_prefix: str) -> List[Tuple[str, int]]:
    s3 = _emulator_s3_client()
    bucket_name, prefix = _parse_gs_uri(gs_prefix)
    if s3:
        paginator = s3.get_paginator("list_objects_v2")
        objects: List[Tuple[str, int]] = []
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                objects.append((item["Key"], int(item.get("Size", 0))))
        return objects

    storage_client = get_dataset_storage()
    bucket = storage_client.bucket
    return [(blob.name, int(blob.size or 0)) for blob in bucket.list_blobs(prefix=prefix)]


def _download_gs_bytes(gs_uri: str) -> bytes:
    s3 = _emulator_s3_client()
    bucket_name, key = _parse_gs_uri(gs_uri)
    if s3:
        return s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()

    storage_client = get_dataset_storage()
    bucket = storage_client.bucket
    return bucket.blob(key).download_as_bytes()


def _download_gs_text(gs_uri: str) -> str:
    return _download_gs_bytes(gs_uri).decode("utf-8")


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    IMAGEFOLDER = "imagefolder"
    COCO = "coco"
    CSV = "csv"
    TFRECORD = "tfrecord"
    UNKNOWN = "unknown"


@dataclass
class DatasetMetadata:
    """Metadata for a loaded dataset."""

    format: DatasetFormat
    total_samples: int
    num_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    total_size_bytes: int
    sample_shape: Optional[Tuple[int, ...]] = None
    features: Optional[List[str]] = None  # For CSV datasets


@dataclass
class DataSample:
    """A single data sample from a dataset."""

    data: Any  # Image data, feature vector, etc.
    label: Union[int, str]
    metadata: Dict[str, Any]  # Additional info like filepath, bounding boxes, etc.
    sample_id: str


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, dataset_path: str, format: Optional[DatasetFormat] = None):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset (local or GCS)
            format: Dataset format (auto-detected if None)
        """
        self.dataset_path = dataset_path
        self.format = format
        self.metadata: Optional[DatasetMetadata] = None
        self._is_gcs = dataset_path.startswith("gs://")

    def load_metadata(self) -> DatasetMetadata:
        """
        Load dataset metadata without loading all data.

        Returns:
            DatasetMetadata instance
        """
        raise NotImplementedError

    def stream_samples(self, batch_size: int = 32) -> Iterator[List[DataSample]]:
        """
        Stream dataset samples in batches to avoid OOM.

        Args:
            batch_size: Number of samples per batch

        Yields:
            Batches of DataSample instances
        """
        raise NotImplementedError

    def get_sample(self, index: int) -> DataSample:
        """
        Get a specific sample by index.

        Args:
            index: Sample index

        Returns:
            DataSample instance
        """
        raise NotImplementedError


class ImageFolderLoader(DatasetLoader):
    """Loader for ImageFolder format datasets."""

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path, DatasetFormat.IMAGEFOLDER)
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []  # (filepath, class_idx)

    def load_metadata(self) -> DatasetMetadata:
        """Load ImageFolder metadata."""
        if self._is_gcs:
            return self._load_metadata_gcs()
        else:
            return self._load_metadata_local()

    def _load_metadata_local(self) -> DatasetMetadata:
        """Load metadata from local filesystem."""
        dataset_dir = Path(self.dataset_path)

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        # Get class directories
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Count samples per class
        class_distribution = {}
        total_samples = 0
        total_size = 0

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]

            class_distribution[class_name] = len(image_files)
            total_samples += len(image_files)

            # Store sample paths
            for img_file in image_files:
                self.samples.append((str(img_file), self.class_to_idx[class_name]))
                total_size += img_file.stat().st_size

        self.metadata = DatasetMetadata(
            format=DatasetFormat.IMAGEFOLDER,
            total_samples=total_samples,
            num_classes=len(class_names),
            class_names=class_names,
            class_distribution=class_distribution,
            total_size_bytes=total_size,
        )

        logger.info(
            f"Loaded ImageFolder metadata: {total_samples} samples, {len(class_names)} classes"
        )
        return self.metadata

    def _load_metadata_gcs(self) -> DatasetMetadata:
        """Load metadata from GCS."""
        # Parse GCS path: gs://bucket/path/to/dataset
        bucket_name, blob_prefix = _parse_gs_uri(self.dataset_path)
        if not blob_prefix.endswith("/"):
            blob_prefix += "/"
        blobs = _list_gs_objects(f"gs://{bucket_name}/{blob_prefix}")

        # Extract class directories and files
        class_files: Dict[str, List[Tuple[str, int]]] = {}  # class_name -> [(blob_name, size)]
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

        for blob_name, blob_size in blobs:
            # Skip directory markers
            if blob_name.endswith("/"):
                continue

            # Get relative path from prefix
            relative_path = blob_name[len(blob_prefix) :].lstrip("/")
            parts = relative_path.split("/")

            if len(parts) >= 2:  # class_name/image.jpg
                class_name = parts[0]
                filename = parts[-1]

                # Check if it's an image
                if Path(filename).suffix.lower() in image_extensions:
                    if class_name not in class_files:
                        class_files[class_name] = []
                    class_files[class_name].append((blob_name, blob_size))

        # Build metadata
        class_names = sorted(class_files.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        class_distribution = {}
        total_samples = 0
        total_size = 0

        for class_name, files in class_files.items():
            class_distribution[class_name] = len(files)
            total_samples += len(files)
            total_size += sum(size for _, size in files)

            # Store sample paths
            for blob_name, _ in files:
                self.samples.append(
                    (f"gs://{bucket_name}/{blob_name}", self.class_to_idx[class_name])
                )

        self.metadata = DatasetMetadata(
            format=DatasetFormat.IMAGEFOLDER,
            total_samples=total_samples,
            num_classes=len(class_names),
            class_names=class_names,
            class_distribution=class_distribution,
            total_size_bytes=total_size,
        )

        logger.info(
            f"Loaded GCS ImageFolder metadata: {total_samples} samples, {len(class_names)} classes"
        )
        return self.metadata

    def stream_samples(self, batch_size: int = 32) -> Iterator[List[DataSample]]:
        """Stream ImageFolder samples in batches."""
        if not self.metadata:
            self.load_metadata()

        batch = []

        for idx, (filepath, label) in enumerate(self.samples):
            try:
                # Load image
                if self._is_gcs:
                    sample = self._load_gcs_image(filepath, label, idx)
                else:
                    sample = self._load_local_image(filepath, label, idx)

                batch.append(sample)

                # Yield batch when full
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                logger.warning(f"Failed to load sample {idx} from {filepath}: {e}")
                continue

        # Yield remaining samples
        if batch:
            yield batch

    def _load_local_image(self, filepath: str, label: int, idx: int) -> DataSample:
        """Load image from local filesystem."""
        with Image.open(filepath) as img:
            img_array = np.array(img)

        return DataSample(
            data=img_array,
            label=label,
            metadata={
                "filepath": filepath,
                "shape": img_array.shape,
                "class_name": self.metadata.class_names[label],
            },
            sample_id=f"sample_{idx}",
        )

    def _load_gcs_image(self, gcs_path: str, label: int, idx: int) -> DataSample:
        """Load image from GCS."""
        image_data = _download_gs_bytes(gcs_path)

        # Load with PIL
        with Image.open(BytesIO(image_data)) as img:
            img_array = np.array(img)

        return DataSample(
            data=img_array,
            label=label,
            metadata={
                "gcs_path": gcs_path,
                "shape": img_array.shape,
                "class_name": self.metadata.class_names[label],
            },
            sample_id=f"sample_{idx}",
        )

    def get_sample(self, index: int) -> DataSample:
        """Get a specific sample by index."""
        if not self.metadata:
            self.load_metadata()

        if index >= len(self.samples):
            raise IndexError(f"Sample index {index} out of range (0-{len(self.samples)-1})")

        filepath, label = self.samples[index]

        if self._is_gcs:
            return self._load_gcs_image(filepath, label, index)
        else:
            return self._load_local_image(filepath, label, index)


class COCOLoader(DatasetLoader):
    """Loader for COCO format datasets."""

    def __init__(self, dataset_path: str, annotations_file: str = "annotations.json"):
        super().__init__(dataset_path, DatasetFormat.COCO)
        self.annotations_file = annotations_file
        self.coco_data: Optional[Dict[str, Any]] = None
        self.image_id_to_annotations: Dict[int, List[Dict[str, Any]]] = {}

    def load_metadata(self) -> DatasetMetadata:
        """Load COCO metadata."""
        # Load annotations JSON
        if self._is_gcs:
            annotations_path = f"{self.dataset_path}/{self.annotations_file}"
            annotations_json = _download_gs_text(annotations_path)
            self.coco_data = json.loads(annotations_json)
        else:
            annotations_path = Path(self.dataset_path) / self.annotations_file
            with open(annotations_path, "r") as f:
                self.coco_data = json.load(f)

        # Parse COCO structure
        images = self.coco_data.get("images", [])
        annotations = self.coco_data.get("annotations", [])
        categories = self.coco_data.get("categories", [])

        # Build image_id -> annotations mapping
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Count samples per category
        category_names = [cat["name"] for cat in categories]
        class_distribution = {cat["name"]: 0 for cat in categories}

        for ann in annotations:
            cat_id = ann["category_id"]
            cat_name = next(cat["name"] for cat in categories if cat["id"] == cat_id)
            class_distribution[cat_name] += 1

        total_size = 0
        if not self._is_gcs:
            # Calculate total size for local files
            images_dir = Path(self.dataset_path) / "images"
            if images_dir.exists():
                for img_info in images:
                    img_path = images_dir / img_info["file_name"]
                    if img_path.exists():
                        total_size += img_path.stat().st_size

        self.metadata = DatasetMetadata(
            format=DatasetFormat.COCO,
            total_samples=len(images),
            num_classes=len(categories),
            class_names=category_names,
            class_distribution=class_distribution,
            total_size_bytes=total_size,
        )

        logger.info(f"Loaded COCO metadata: {len(images)} images, {len(categories)} categories")
        return self.metadata

    def stream_samples(self, batch_size: int = 32) -> Iterator[List[DataSample]]:
        """Stream COCO samples in batches."""
        if not self.metadata or not self.coco_data:
            self.load_metadata()

        batch = []
        images = self.coco_data["images"]

        for idx, img_info in enumerate(images):
            try:
                # Load image
                img_path = f"{self.dataset_path}/images/{img_info['file_name']}"

                if self._is_gcs:
                    sample = self._load_gcs_coco_image(img_path, img_info, idx)
                else:
                    sample = self._load_local_coco_image(img_path, img_info, idx)

                batch.append(sample)

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                logger.warning(f"Failed to load COCO sample {idx}: {e}")
                continue

        if batch:
            yield batch

    def _load_local_coco_image(
        self, filepath: str, img_info: Dict[str, Any], idx: int
    ) -> DataSample:
        """Load COCO image from local filesystem."""
        with Image.open(filepath) as img:
            img_array = np.array(img)

        # Get annotations for this image
        image_id = img_info["id"]
        annotations = self.image_id_to_annotations.get(image_id, [])

        return DataSample(
            data=img_array,
            label=annotations[0]["category_id"] if annotations else -1,
            metadata={
                "filepath": filepath,
                "image_id": image_id,
                "annotations": annotations,
                "height": img_info.get("height"),
                "width": img_info.get("width"),
            },
            sample_id=f"coco_{image_id}",
        )

    def _load_gcs_coco_image(self, gcs_path: str, img_info: Dict[str, Any], idx: int) -> DataSample:
        """Load COCO image from GCS."""
        image_data = _download_gs_bytes(gcs_path)

        with Image.open(BytesIO(image_data)) as img:
            img_array = np.array(img)

        image_id = img_info["id"]
        annotations = self.image_id_to_annotations.get(image_id, [])

        return DataSample(
            data=img_array,
            label=annotations[0]["category_id"] if annotations else -1,
            metadata={
                "gcs_path": gcs_path,
                "image_id": image_id,
                "annotations": annotations,
                "height": img_info.get("height"),
                "width": img_info.get("width"),
            },
            sample_id=f"coco_{image_id}",
        )

    def get_sample(self, index: int) -> DataSample:
        """Get a specific COCO sample by index."""
        if not self.metadata or not self.coco_data:
            self.load_metadata()

        images = self.coco_data["images"]
        if index >= len(images):
            raise IndexError(f"Sample index {index} out of range")

        img_info = images[index]
        img_path = f"{self.dataset_path}/images/{img_info['file_name']}"

        if self._is_gcs:
            return self._load_gcs_coco_image(img_path, img_info, index)
        else:
            return self._load_local_coco_image(img_path, img_info, index)


class CSVLoader(DatasetLoader):
    """Loader for CSV format datasets."""

    def __init__(self, dataset_path: str, label_column: str = "label"):
        super().__init__(dataset_path, DatasetFormat.CSV)
        self.label_column = label_column
        self.data: List[Dict[str, Any]] = []
        self.feature_columns: List[str] = []

    def load_metadata(self) -> DatasetMetadata:
        """Load CSV metadata."""
        # Read CSV file
        if self._is_gcs:
            csv_content = _download_gs_text(self.dataset_path)
            reader = csv.DictReader(csv_content.splitlines())
            self.data = list(reader)
        else:
            with open(self.dataset_path, "r") as f:
                reader = csv.DictReader(f)
                self.data = list(reader)

        # Extract feature columns
        if self.data:
            all_columns = list(self.data[0].keys())
            self.feature_columns = [col for col in all_columns if col != self.label_column]

        # Count class distribution
        labels = [row[self.label_column] for row in self.data]
        unique_labels = sorted(set(labels))
        class_distribution = {label: labels.count(label) for label in unique_labels}

        file_size = 0
        if not self._is_gcs:
            file_size = Path(self.dataset_path).stat().st_size

        self.metadata = DatasetMetadata(
            format=DatasetFormat.CSV,
            total_samples=len(self.data),
            num_classes=len(unique_labels),
            class_names=unique_labels,
            class_distribution=class_distribution,
            total_size_bytes=file_size,
            features=self.feature_columns,
        )

        logger.info(
            f"Loaded CSV metadata: {len(self.data)} samples, {len(unique_labels)} classes, {len(self.feature_columns)} features"
        )
        return self.metadata

    def stream_samples(self, batch_size: int = 32) -> Iterator[List[DataSample]]:
        """Stream CSV samples in batches."""
        if not self.metadata:
            self.load_metadata()

        batch = []

        for idx, row in enumerate(self.data):
            # Extract features
            features = {col: row[col] for col in self.feature_columns}
            label = row[self.label_column]

            sample = DataSample(
                data=features, label=label, metadata={"row_index": idx}, sample_id=f"csv_{idx}"
            )

            batch.append(sample)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def get_sample(self, index: int) -> DataSample:
        """Get a specific CSV sample by index."""
        if not self.metadata:
            self.load_metadata()

        if index >= len(self.data):
            raise IndexError(f"Sample index {index} out of range")

        row = self.data[index]
        features = {col: row[col] for col in self.feature_columns}
        label = row[self.label_column]

        return DataSample(
            data=features, label=label, metadata={"row_index": index}, sample_id=f"csv_{index}"
        )


def create_loader(
    dataset_path: str, format: Optional[DatasetFormat] = None, **kwargs
) -> DatasetLoader:
    """
    Factory function to create appropriate dataset loader.

    Args:
        dataset_path: Path to dataset
        format: Dataset format (auto-detected if None)
        **kwargs: Additional arguments for specific loaders

    Returns:
        DatasetLoader instance
    """
    # Auto-detect format if not provided
    if format is None:
        if dataset_path.endswith(".csv"):
            format = DatasetFormat.CSV
        elif "annotations.json" in dataset_path or "coco" in dataset_path.lower():
            format = DatasetFormat.COCO
        else:
            # Default to ImageFolder
            format = DatasetFormat.IMAGEFOLDER

    # Create appropriate loader
    if format == DatasetFormat.IMAGEFOLDER:
        return ImageFolderLoader(dataset_path)
    elif format == DatasetFormat.COCO:
        return COCOLoader(dataset_path, **kwargs)
    elif format == DatasetFormat.CSV:
        return CSVLoader(dataset_path, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset format: {format}")
