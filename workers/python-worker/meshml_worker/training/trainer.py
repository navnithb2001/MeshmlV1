"""
Training Loop Implementation

Handles:
- Data shard downloading
- Local training on assigned data
- Gradient computation and upload
- Checkpoint management
- Progress tracking
"""

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from meshml_worker.communication.heartbeat import HeartbeatSender
from meshml_worker.communication.parameter_server_client import ParameterServerClient
from meshml_worker.config import WorkerConfig
from meshml_worker.training.dataloader import download_data_shard
from meshml_worker.training.model_loader import ModelLoader
from meshml_worker.utils.checkpoint import CheckpointManager
from meshml_worker.utils.logger import TrainingLogger
from meshml_worker.utils.optimization import (
    MemoryProfiler,
    OptimizedDataLoader,
    PerformanceBenchmark,
)
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop implementation

    Features:
    - Download and use data shards
    - Local training with PyTorch
    - Gradient computation and upload
    - Mixed precision training
    - Checkpoint management
    - Progress tracking
    - Heartbeat monitoring
    - Memory profiling and optimization
    - Performance benchmarking
    """

    def __init__(
        self,
        config: WorkerConfig,
        grpc_client: ParameterServerClient,
        device: str,
        orchestrator_client: Optional[Any] = None,
        metrics_client: Optional[Any] = None,
        job_id: Optional[str] = None,
        model_path: Optional[Path] = None,
        data_paths: Optional[List[Path]] = None,
        pause_event: Optional[asyncio.Event] = None,
    ):
        """Initialize trainer

        Args:
            config: Worker configuration
            grpc_client: Parameter Server gRPC client
            device: Training device
            orchestrator_client: Optional Task Orchestrator client for reporting
            job_id: Optional job ID (for orchestrated training)
            model_path: Optional path to model file (overrides default model)
            data_paths: Optional list of data shard paths (overrides data loading)
        """
        self.config = config
        self.grpc_client = grpc_client  # Note: keeping name for backwards compatibility
        self.device = device

        # Orchestrator integration
        self.orchestrator_client = orchestrator_client
        self.metrics_client = metrics_client
        self.job_id = job_id
        self.model_path = model_path
        self.data_paths = data_paths
        self.pause_event = pause_event

        # Components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.scaler: Optional[GradScaler] = None

        # Runtime task config (from MODEL_METADATA)
        self.task_type: str = "classification"  # default until model is loaded
        self.loss_name: str = "cross_entropy"
        self.metric_names: list = ["accuracy"]

        # Data
        self.train_loader: Optional[Any] = None
        self.val_loader: Optional[Any] = None

        # State
        self.model_id: Optional[str] = None  # Current training model ID
        self.current_epoch = 0
        self.current_iteration = 0
        self.global_version = 0

        # Managers
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.training_logger: Optional[TrainingLogger] = None
        self.heartbeat: Optional[HeartbeatSender] = None

        # Optimization tools
        self.memory_profiler: Optional[MemoryProfiler] = None
        self.performance_benchmark: Optional[PerformanceBenchmark] = None

        # Mixed precision
        if config.training.mixed_precision and device.startswith("cuda"):
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")

        logger.info(f"Trainer initialized: device={device}")

    async def train(
        self, model_id: str, job_id: str, batch_ids: List[str], epochs: Optional[int] = None
    ) -> None:
        """Training loop with Task Orchestrator integration

        This method:
        1. Loads data from pre-downloaded shards
        2. Trains the model
        3. Reports progress to Task Orchestrator after each batch

        Args:
            model_id: Model ID
            job_id: Job ID from Task Orchestrator
            batch_ids: List of batch IDs assigned to this worker
            epochs: Number of epochs (None = train until convergence)
        """
        logger.info(f"Starting orchestrated training: model_id={model_id}, job_id={job_id}")
        logger.info(f"Assigned batches: {len(batch_ids)}")

        try:
            # Initialize components (without data loading and checkpoint)
            self._initialize_training_minimal(model_id)

            # Load data - use pre-downloaded shards
            if self.data_paths:
                logger.info(f"Using {len(self.data_paths)} pre-downloaded data shards")
                self.train_loader = self._create_dataloader_from_shards(
                    shard_paths=self.data_paths,
                    batch_size=self.config.training.batch_size,
                    num_workers=self.config.training.num_workers,
                )
            else:
                logger.warning("No pre-downloaded data paths, cannot train")
                raise ValueError("Orchestrated mode requires pre-downloaded data_paths")

            # Training loop
            max_epochs = epochs or 100

            if self.metrics_client:
                await self.metrics_client.start(
                    job_id=job_id, worker_id=self.config.worker.id or "unknown"
                )

            for epoch in range(self.current_epoch, max_epochs):
                self.current_epoch = epoch

                logger.info("-" * 60)
                logger.info("-" * 60)
                logger.info("\n")
                logger.info(f"Epoch {epoch + 1}/{max_epochs}")

                # Train one epoch with batch-level reporting
                epoch_loss, epoch_metrics = await self._train_epoch_with_reporting(
                    epoch=epoch, job_id=job_id, batch_ids=batch_ids
                )

                # Log epoch results
                self.training_logger.log_epoch(
                    epoch=epoch,
                    loss=epoch_loss,
                    accuracy=epoch_metrics.get("accuracy"),
                    other_metrics=epoch_metrics,
                )

                # Save checkpoint
                self._save_checkpoint(epoch, epoch_loss, epoch_metrics)

                logger.info(
                    f"Epoch {epoch + 1} completed: loss={epoch_loss:.4f}, "
                    f"accuracy={epoch_metrics.get('accuracy', 'N/A')}"
                )
                logger.info("\n")
                logger.info("-" * 60)
                logger.info("-" * 60)

            logger.info("Orchestrated training completed successfully")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Orchestrated training failed: {e}", exc_info=True)
            raise
        finally:
            if self.metrics_client:
                await self.metrics_client.close()
            self._cleanup()

    def _compute_metrics(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Metric Factory — compute requested metrics based on task_type and metric_names.

        Returns:
            dict of metric_name -> float value
        """
        results: dict = {}

        for metric in self.metric_names:
            try:
                if metric == "accuracy":
                    if self.task_type == "classification":
                        _, predicted = torch.max(output, 1)
                        results["accuracy"] = 100.0 * (predicted == target.long()).float().mean().item()
                    elif self.task_type == "binary":
                        predicted = (torch.sigmoid(output.squeeze()) > 0.5).float()
                        results["accuracy"] = 100.0 * (predicted == target.squeeze()).float().mean().item()
                    else:
                        results["accuracy"] = 0.0  # not meaningful for regression
                elif metric == "mae":
                    results["mae"] = torch.mean(torch.abs(output.squeeze() - target.squeeze())).item()
                elif metric == "mse":
                    results["mse"] = torch.mean((output.squeeze() - target.squeeze()) ** 2).item()
                else:
                    logger.debug(f"Unknown metric '{metric}', skipping.")
            except Exception as e:
                logger.warning(f"Failed computing metric '{metric}': {e}")

        return results

    def _prepare_batch(
        self, raw_data: Any, raw_target: Any
    ) -> tuple:
        """
        Generalized batch parsing: handle (data, target) tuples, dicts, or Tensors.
        Casts targets to the correct dtype for the current task.

        Returns:
            (data_tensor, target_tensor)
        """
        # Handle dict-style batches
        if isinstance(raw_data, dict):
            data = raw_data.get("input", raw_data.get("data", next(iter(raw_data.values()))))
        else:
            data = raw_data

        if isinstance(raw_target, dict):
            target = raw_target.get("label", raw_target.get("target", next(iter(raw_target.values()))))
        else:
            target = raw_target

        data = torch.as_tensor(data, dtype=torch.float32).to(self.device)

        # Cast target dtype based on task
        if self.task_type in ("regression", "binary"):
            target = torch.as_tensor(target, dtype=torch.float32).to(self.device)
        else:
            target = torch.as_tensor(target, dtype=torch.long).to(self.device)

        return data, target

    def _apply_output_transform(self, output: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Per-task output/target adjustment before loss computation.
        - regression/binary: squeeze output to 1D, ensure target is 1D float
        - classification: leave as-is
        """
        if self.task_type in ("regression", "binary"):
            output = output.squeeze(-1)
            target = target.squeeze(-1)
        return output, target

    async def _train_epoch_with_reporting(
        self, epoch: int, job_id: str, batch_ids: List[str]
    ) -> tuple:
        """Train one epoch with batch-level progress reporting to Task Orchestrator"""
        if self.model is None or self.train_loader is None:
            raise RuntimeError("Model or data not loaded")

        self.model.train()
        epoch_loss = 0.0
        batch_count = 0

        # Accumulate metrics across batches
        epoch_metric_sums: dict = {m: 0.0 for m in self.metric_names}

        batches_per_shard = (
            len(self.train_loader) // len(batch_ids) if batch_ids else len(self.train_loader)
        )
        current_batch_idx = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            if self.pause_event:
                while self.pause_event.is_set():
                    await asyncio.sleep(1.0)

            # Generalized batch unpacking
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                raw_data, raw_target = batch
            elif isinstance(batch, dict):
                keys = list(batch.keys())
                raw_data, raw_target = batch[keys[0]], batch[keys[1]]
            else:
                logger.warning(f"Unexpected batch type: {type(batch)}, skipping.")
                continue

            data, target = self._prepare_batch(raw_data, raw_target)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    output = self.model(data)
                    output, target_for_loss = self._apply_output_transform(output, target)
                    loss = self.criterion(output, target_for_loss)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                output, target_for_loss = self._apply_output_transform(output, target)
                loss = self.criterion(output, target_for_loss)
                loss.backward()
                self.optimizer.step()

            self._push_gradients(batch_idx=batch_idx, epoch=epoch, loss=loss.item())

            # Metric computation
            with torch.no_grad():
                batch_metrics = self._compute_metrics(output, target_for_loss)

            epoch_loss += loss.item()
            batch_count += 1
            for m, val in batch_metrics.items():
                epoch_metric_sums[m] = epoch_metric_sums.get(m, 0.0) + val

            # Progress bar
            avg_loss = epoch_loss / batch_count
            postfix = {"loss": f"{avg_loss:.4f}"}
            if "accuracy" in batch_metrics:
                postfix["acc"] = f"{batch_metrics['accuracy']:.2f}%"
            elif "mae" in batch_metrics:
                postfix["mae"] = f"{batch_metrics['mae']:.4f}"
            pbar.set_postfix(postfix)

            self.current_iteration += 1
            if self.metrics_client:
                await self.metrics_client.send(
                    step=self.current_iteration,
                    loss=loss.item(),
                    accuracy=batch_metrics.get("accuracy", 0.0),
                    timestamp_ms=int(time.time() * 1000),
                )

            if batch_ids and (batch_idx + 1) % batches_per_shard == 0 and current_batch_idx < len(batch_ids):
                current_batch_idx += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        avg_metrics = {
            m: epoch_metric_sums.get(m, 0.0) / batch_count
            for m in self.metric_names
            if batch_count > 0
        }

        return avg_loss, avg_metrics

    def _initialize_training_minimal(self, model_id: str) -> None:
        """Initialize training components for orchestrated mode (minimal version)

        This skips data loading since data_paths are provided externally.

        Args:
            model_id: Model ID
        """
        logger.info("Initializing training components (minimal for orchestrated mode)...")

        # Store model ID
        self.model_id = model_id

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.storage.checkpoints_dir, model_id=model_id
        )

        # Initialize training logger
        self.training_logger = TrainingLogger(
            log_dir=self.config.storage.base_dir / "logs", model_id=model_id
        )

        # Initialize optimization tools
        self.memory_profiler = MemoryProfiler(device=self.device)
        self.performance_benchmark = PerformanceBenchmark()
        logger.info("Memory profiler and performance benchmark initialized")

        # Register worker with Parameter Server
        try:
            self.grpc_client.register_worker(
                worker_id=self.config.worker.id or "unknown",
                model_id=model_id,
                metadata={"device": self.device},
            )
            logger.info(f"Worker registered with Parameter Server for model {model_id}")
        except Exception as e:
            logger.warning(f"Failed to register worker: {e}")

        # Load model definition
        self._load_model(model_id)

        # Initialize optimizer
        self._initialize_optimizer()

        # Fetch initial weights from Parameter Server
        params_job_id = self.job_id or model_id
        self._fetch_weights(params_job_id)

        logger.info("Training initialization complete (orchestrated mode)")

    def _load_model(self, model_id: str) -> None:
        """Load model definition

        Args:
            model_id: Model ID
        """
        logger.info(f"Loading model definition for {model_id}")

        # Require model path from Model Registry (orchestrated mode only)
        if not self.model_path or not self.model_path.exists():
            raise ValueError(
                f"Model path not provided or does not exist. "
                f"Models must be downloaded from Model Registry. "
                f"Expected path: {self.model_path}"
            )

        logger.info(f"Using model from Model Registry: {self.model_path}")
        model_source = str(self.model_path)

        # Load model
        model_loader = ModelLoader(models_dir=self.config.storage.models_dir)
        create_model, create_dataloader, metadata = model_loader.load_model(
            model_source=model_source, model_id=model_id
        )

        # Create model instance
        self.model = create_model(device=self.device)
        self.model = self.model.to(self.device)  # Ensure model is on correct device
        self.create_dataloader_fn = create_dataloader

        # ── Loss Factory based on MODEL_METADATA ─────────────────────────────
        self.task_type = str(metadata.get("task_type", "classification"))
        self.loss_name = str(metadata.get("loss", "cross_entropy"))
        self.metric_names = list(metadata.get("metrics", ["accuracy"]))

        _LOSS_FACTORY: dict = {
            "cross_entropy": nn.CrossEntropyLoss,
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
            "bce_with_logits": nn.BCEWithLogitsLoss,
            "bce": nn.BCELoss,
        }
        criterion_cls = _LOSS_FACTORY.get(self.loss_name, nn.CrossEntropyLoss)
        self.criterion = criterion_cls()

        logger.info(
            f"Model loaded: {metadata['name']} v{metadata['version']} "
            f"| task={self.task_type} loss={self.loss_name} metrics={self.metric_names}"
        )

    def _create_dataloader_from_shards(
        self, shard_paths: List[Path], batch_size: int, num_workers: int
    ) -> torch.utils.data.DataLoader:
        """
        Create DataLoader from downloaded shard paths

        Args:
            shard_paths: List of paths to extracted shard directories
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        import pickle
        from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

        logger.info(f"Creating DataLoader from {len(shard_paths)} shards")

        # Label dtype: long for classification, float for regression/binary
        is_float_target = self.task_type in ("regression", "binary")

        # Preferred path: load pre-downloaded serialized batch payloads.
        shard_datasets = []
        for shard_path in shard_paths:
            try:
                batch_file = shard_path / "batch.data"
                if batch_file.exists():
                    batch_bytes = batch_file.read_bytes()
                    try:
                        payload = pickle.loads(batch_bytes)
                    except ModuleNotFoundError:
                        class _DataSampleCompat:
                            def __init__(self, data, label, metadata, sample_id):
                                self.data = data
                                self.label = label
                                self.metadata = metadata
                                self.sample_id = sample_id

                        class _CompatUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module == "app.services.dataset_loader" and name == "DataSample":
                                    return _DataSampleCompat
                                return super().find_class(module, name)

                        payload = _CompatUnpickler(io.BytesIO(batch_bytes)).load()
                    samples = payload.get("samples", [])
                    if not samples:
                        logger.warning(f"Shard {shard_path} has no samples")
                        continue

                    xs = []
                    ys = []
                    for sample in samples:
                        sample_data = sample.get("data") if isinstance(sample, dict) else sample.data
                        sample_label = (
                            sample.get("label") if isinstance(sample, dict) else sample.label
                        )
                        x = torch.as_tensor(sample_data, dtype=torch.float32)
                        # Convert HWC -> CHW for image tensors.
                        if x.ndim == 3:
                            x = x.permute(2, 0, 1)
                        elif x.ndim == 2:
                            x = x.unsqueeze(0)
                        xs.append(x)
                        ys.append(float(sample_label) if is_float_target else int(sample_label))

                    label_dtype = torch.float32 if is_float_target else torch.long
                    dataset = TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=label_dtype))
                    shard_datasets.append(dataset)
                    logger.debug(
                        f"Loaded serialized shard from {batch_file}: {len(dataset)} samples "
                        f"(label_dtype={label_dtype})"
                    )
                    continue
            except Exception as e:
                logger.warning(f"Failed to load serialized shard from {shard_path}: {e}")

            # Backward compatibility: if shard directory contains image folders, use torchvision.
            try:
                from torchvision import datasets, transforms

                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
                )
                dataset = datasets.ImageFolder(root=str(shard_path), transform=transform)
                shard_datasets.append(dataset)
                logger.debug(f"Loaded imagefolder shard from {shard_path}: {len(dataset)} samples")
            except Exception as img_err:
                logger.warning(f"Failed to load shard from {shard_path}: {img_err}")
                continue

        if not shard_datasets:
            raise ValueError("No valid shards could be loaded from serialized batches or imagefolders")

        combined_dataset = ConcatDataset(shard_datasets)
        logger.info(f"Combined dataset: {len(combined_dataset)} total samples")

        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return dataloader

    def _initialize_optimizer(self) -> None:
        """Initialize optimizer"""
        # Use Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        logger.info("Optimizer initialized: Adam(lr=0.001)")

    def _fetch_weights(self, params_job_id: str) -> None:
        """Fetch initial weights from Parameter Server

        Args:
            params_job_id: Parameter Server job identifier
        """
        logger.info("Fetching initial weights from Parameter Server...")

        try:
            # Get current model version
            version = self.grpc_client.get_model_version(params_job_id)
            self.global_version = version
            logger.info(f"Current Parameter Server version: {version}")

            # Get weights (if available)
            state_dict = self.grpc_client.get_weights(params_job_id, version)

            # Load state dict into model if we got weights
            if state_dict and self.model is not None:
                try:
                    self.model.load_state_dict(state_dict)
                    logger.info(f"Loaded weights from Parameter Server: version={version}")
                except Exception as e:
                    logger.warning(f"Failed to load state dict: {e}, using current weights")

            logger.info(f"Synced with Parameter Server: version={version}")

        except Exception as e:
            # It's okay if weights don't exist yet (new model)
            logger.info(f"No weights available from Parameter Server (new model): {e}")
            logger.info("Using randomly initialized weights")
            self.global_version = 0

    def _train_epoch(self, epoch: int) -> tuple:
        """Train one epoch with profiling and benchmarking

        Args:
            epoch: Current epoch

        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0

        # Start epoch benchmark
        if self.performance_benchmark:
            self.performance_benchmark.start_epoch()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            # Start batch benchmark
            if self.performance_benchmark:
                self.performance_benchmark.start_batch()

            # Move to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass with optional profiling
            if self.memory_profiler and batch_idx % 10 == 0:  # Profile every 10th batch
                with self.memory_profiler.profile(f"epoch_{epoch}_batch_{batch_idx}"):
                    loss, predictions = self._train_batch(data, target, batch_idx, epoch)
            else:
                loss, predictions = self._train_batch(data, target, batch_idx, epoch)

            # Update metrics
            epoch_loss += loss
            num_batches += 1

            # Calculate accuracy (only for classification tasks)
            # Check if target is categorical (1D) or continuous (multi-dimensional)
            if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[1] == 1):
                # Classification task
                _, predicted = torch.max(predictions, 1)
                total += target.size(0)
                if len(target.shape) == 2:
                    target = target.squeeze(1)
                correct += (predicted == target).sum().item()
            else:
                # Regression task - skip accuracy calculation
                total += target.size(0)

            # End batch benchmark
            if self.performance_benchmark:
                self.performance_benchmark.end_batch(batch_size=data.size(0))

            # Update progress bar
            current_loss = epoch_loss / num_batches
            if correct > 0 and total > 0:
                current_acc = 100.0 * correct / total
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})
            else:
                pbar.set_postfix({"loss": f"{current_loss:.4f}"})

            # Periodic checkpoint
            if (batch_idx + 1) % 100 == 0:
                self._update_heartbeat_status(
                    state="training",
                    current_epoch=epoch,
                    current_batch=batch_idx,
                    total_batches=len(self.train_loader),
                    loss=current_loss,
                )

        # End epoch benchmark
        if self.performance_benchmark:
            self.performance_benchmark.end_epoch()

        avg_loss = epoch_loss / num_batches
        accuracy = 100.0 * correct / total

        metrics = {"accuracy": accuracy, "num_batches": num_batches, "num_samples": total}

        # Log performance stats every few epochs
        if self.performance_benchmark and epoch % 5 == 0:
            self.performance_benchmark.print_summary()

        return avg_loss, metrics

    def _train_batch(
        self, data: torch.Tensor, target: torch.Tensor, batch_idx: int, epoch: int
    ) -> tuple:
        """Train single batch

        Args:
            data: Input data
            target: Target labels
            batch_idx: Batch index
            epoch: Current epoch

        Returns:
            Tuple of (loss, predictions)
        """
        self.optimizer.zero_grad()

        # Mixed precision training
        if self.scaler is not None:
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )

            self.optimizer.step()

        # Push gradients to Parameter Server (periodically)
        if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
            self._push_gradients(batch_idx, epoch, loss.item())

        self.current_iteration += 1

        return loss.item(), output

    def _push_gradients(self, batch_idx: int, epoch: int, loss: float) -> None:
        """Push gradients to Parameter Server

        Args:
            batch_idx: Batch index
            epoch: Current epoch
            loss: Current loss
        """
        try:
            # Extract gradients
            gradients = {}
            gradient_norm = 0.0

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.cpu()  # Keep as tensor for HTTP serialization
                    gradient_norm += param.grad.norm().item() ** 2

            gradient_norm = gradient_norm**0.5

            # Prepare metadata
            metadata = {"gradient_norm": gradient_norm, "computation_time_ms": 0}

            # Push to Parameter Server via HTTP
            response = self.grpc_client.push_gradients(
                worker_id=self.config.worker.id or "unknown",
                model_id=self.job_id or self.model_id or "unknown",
                version_id=self.global_version,
                gradients=gradients,
                num_samples=self.config.training.batch_size,
                loss=loss,
                metrics=metadata,
            )

            if response.get("success"):
                self.global_version = int(response.get("new_version", self.global_version))
            logger.debug(f"Gradients pushed: batch={batch_idx}, response={response}")

        except Exception as e:
            logger.warning(f"Failed to push gradients: {e}")

    def _save_checkpoint(self, epoch: int, loss: float, metrics: Dict[str, Any]) -> None:
        """Save training checkpoint

        Args:
            epoch: Current epoch
            loss: Current loss
            metrics: Training metrics
        """
        try:
            self.checkpoint_manager.save_checkpoint(
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                epoch=epoch,
                iteration=self.current_iteration,
                loss=loss,
                metrics=metrics,
                is_best=(epoch == 0 or loss < getattr(self, "_best_loss", float("inf"))),
            )

            self._best_loss = min(loss, getattr(self, "_best_loss", float("inf")))

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)

            # Load model state
            self.model.load_state_dict(checkpoint_data["model_state_dict"])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            # Load training state
            self.current_epoch = checkpoint_data["epoch"] + 1
            self.current_iteration = checkpoint_data["iteration"]

            logger.info(f"Checkpoint loaded: epoch={self.current_epoch}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring"""
        self.heartbeat = HeartbeatSender(worker_id=self.config.worker.id, heartbeat_interval=30)

        # Set heartbeat callback
        def heartbeat_callback(data):
            # In production, would send via HTTP or gRPC
            logger.debug(f"Heartbeat: {data}")
            return True

        self.heartbeat.set_heartbeat_callback(heartbeat_callback)
        self.heartbeat.start()

        logger.info("Heartbeat started")

    def _update_heartbeat_status(self, **kwargs) -> None:
        """Update heartbeat status

        Args:
            **kwargs: Status fields to update
        """
        if self.heartbeat:
            self.heartbeat.update_status(**kwargs)

    def _cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up...")

        if self.heartbeat:
            self.heartbeat.stop()

        # Update final status
        self._update_heartbeat_status(state="idle")

        logger.info("Cleanup complete")
