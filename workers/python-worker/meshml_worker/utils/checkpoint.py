"""
Checkpoint management for saving and loading training state
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints

    Features:
    - Save/load model state
    - Save/load optimizer state
    - Save/load training metadata
    - Keep N best checkpoints
    - Automatic checkpoint rotation
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model_id: str,
        keep_best_n: int = 3,
        keep_every_n_epochs: Optional[int] = None,
    ):
        """Initialize checkpoint manager

        Args:
            checkpoint_dir: Directory for checkpoints
            model_id: Model ID
            keep_best_n: Number of best checkpoints to keep
            keep_every_n_epochs: Save checkpoint every N epochs (optional)
        """
        self.checkpoint_dir = checkpoint_dir
        self.model_id = model_id
        self.keep_best_n = keep_best_n
        self.keep_every_n_epochs = keep_every_n_epochs

        # Create checkpoint directory
        self.model_checkpoint_dir = checkpoint_dir / model_id
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file
        self.metadata_file = self.model_checkpoint_dir / "checkpoints.json"
        self.metadata: Dict[str, Any] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata

        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {"checkpoints": [], "best_checkpoint": None, "best_loss": float("inf")}

    def _save_metadata(self) -> None:
        """Save checkpoint metadata"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]],
        epoch: int,
        iteration: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """Save training checkpoint

        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            epoch: Current epoch
            iteration: Current iteration
            loss: Current loss
            metrics: Optional metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for checkpoint management")

        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_iter{iteration}_{timestamp}.pt"
        checkpoint_path = self.model_checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "iteration": iteration,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": timestamp,
            "model_id": self.model_id,
        }

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Update metadata
        checkpoint_info = {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "iteration": iteration,
            "loss": loss,
            "metrics": metrics or {},
            "timestamp": timestamp,
            "is_best": is_best,
        }

        self.metadata["checkpoints"].append(checkpoint_info)

        # Update best checkpoint
        if is_best or loss < self.metadata["best_loss"]:
            self.metadata["best_checkpoint"] = str(checkpoint_path)
            self.metadata["best_loss"] = loss

            # Create symlink to best checkpoint
            best_link = self.model_checkpoint_dir / "best_checkpoint.pt"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(checkpoint_path.name)
            logger.info(f"Updated best checkpoint: {checkpoint_path}")

        # Save metadata
        self._save_metadata()

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load checkpoint

        Args:
            checkpoint_path: Path to checkpoint (None = load best)

        Returns:
            Checkpoint data
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for checkpoint management")

        # Use best checkpoint if path not specified
        if checkpoint_path is None:
            if self.metadata["best_checkpoint"]:
                checkpoint_path = Path(self.metadata["best_checkpoint"])
            else:
                raise FileNotFoundError("No checkpoints available")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

        return checkpoint_data

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint

        Returns:
            Path to latest checkpoint or None
        """
        if not self.metadata["checkpoints"]:
            return None

        latest = max(self.metadata["checkpoints"], key=lambda x: x["epoch"])
        return Path(latest["path"])

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint

        Returns:
            Path to best checkpoint or None
        """
        if self.metadata["best_checkpoint"]:
            return Path(self.metadata["best_checkpoint"])
        return None

    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy"""
        if not self.metadata["checkpoints"]:
            return

        # Sort by loss (best first)
        sorted_checkpoints = sorted(self.metadata["checkpoints"], key=lambda x: x["loss"])

        # Keep best N
        checkpoints_to_keep = set()
        for checkpoint in sorted_checkpoints[: self.keep_best_n]:
            checkpoints_to_keep.add(checkpoint["path"])

        # Keep every N epochs if configured
        if self.keep_every_n_epochs:
            for checkpoint in self.metadata["checkpoints"]:
                if checkpoint["epoch"] % self.keep_every_n_epochs == 0:
                    checkpoints_to_keep.add(checkpoint["path"])

        # Delete old checkpoints
        new_checkpoints = []
        for checkpoint in self.metadata["checkpoints"]:
            if checkpoint["path"] in checkpoints_to_keep:
                new_checkpoints.append(checkpoint)
            else:
                # Delete checkpoint file
                checkpoint_path = Path(checkpoint["path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Deleted old checkpoint: {checkpoint_path}")

        self.metadata["checkpoints"] = new_checkpoints
        self._save_metadata()

    def list_checkpoints(self) -> list:
        """List all available checkpoints

        Returns:
            List of checkpoint info
        """
        return self.metadata["checkpoints"]
