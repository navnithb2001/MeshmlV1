"""
Device and memory optimization utilities

Provides memory profiling, performance benchmarking, and DataLoader optimization.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Profile memory usage during training

    Tracks memory allocation and usage for CUDA, MPS, and CPU.

    Example:
        >>> profiler = MemoryProfiler(device="cuda:0")
        >>> with profiler.profile("training_step"):
        ...     # Training code
        ...     pass
        >>> stats = profiler.get_stats()
    """

    def __init__(self, device: str):
        """Initialize memory profiler

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu")
        """
        self.device = device
        self.device_type = device.split(":")[0]
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._enabled = self._check_available()

    def _check_available(self) -> bool:
        """Check if profiling is available for device"""
        if self.device_type == "cuda":
            try:
                import torch

                return torch.cuda.is_available()
            except ImportError:
                return False
        elif self.device_type == "mps":
            try:
                import torch

                return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            except ImportError:
                return False
        return True  # CPU always available

    def get_current_memory(self) -> Dict[str, int]:
        """Get current memory usage

        Returns:
            Dictionary with memory stats in bytes
        """
        if not self._enabled:
            return {}

        try:
            import torch

            if self.device_type == "cuda":
                device_id = int(self.device.split(":")[-1])
                return {
                    "allocated": torch.cuda.memory_allocated(device_id),
                    "reserved": torch.cuda.memory_reserved(device_id),
                    "max_allocated": torch.cuda.max_memory_allocated(device_id),
                    "max_reserved": torch.cuda.max_memory_reserved(device_id),
                }
            elif self.device_type == "mps":
                # MPS doesn't have detailed memory APIs yet
                return {
                    "allocated": (
                        torch.mps.current_allocated_memory()
                        if hasattr(torch.mps, "current_allocated_memory")
                        else 0
                    ),
                    "driver_allocated": (
                        torch.mps.driver_allocated_memory()
                        if hasattr(torch.mps, "driver_allocated_memory")
                        else 0
                    ),
                }
            else:
                # CPU memory tracking (basic)
                import psutil

                process = psutil.Process()
                mem_info = process.memory_info()
                return {
                    "rss": mem_info.rss,  # Resident Set Size
                    "vms": mem_info.vms,  # Virtual Memory Size
                }
        except Exception as e:
            logger.debug(f"Failed to get memory stats: {e}")
            return {}

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        """Profile a code block

        Args:
            name: Name for this profiling session

        Yields:
            None
        """
        if not self._enabled:
            yield
            return

        # Get memory before
        mem_before = self.get_current_memory()
        start_time = time.time()

        try:
            yield
        finally:
            # Get memory after
            mem_after = self.get_current_memory()
            duration = time.time() - start_time

            # Calculate differences
            mem_diff = {}
            for key in mem_before:
                if key in mem_after:
                    mem_diff[f"{key}_diff"] = mem_after[key] - mem_before[key]

            # Store stats
            self.stats[name] = {
                "duration_seconds": duration,
                "memory_before": mem_before,
                "memory_after": mem_after,
                "memory_diff": mem_diff,
            }

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling statistics

        Args:
            name: Optional name to get specific stats

        Returns:
            Dictionary of statistics
        """
        if name:
            return self.stats.get(name, {})
        return self.stats

    def reset(self) -> None:
        """Reset all statistics"""
        self.stats.clear()

        if self.device_type == "cuda":
            try:
                import torch

                device_id = int(self.device.split(":")[-1])
                torch.cuda.reset_peak_memory_stats(device_id)
            except Exception:
                pass

    def print_summary(self) -> None:
        """Print a summary of profiling results"""
        if not self.stats:
            logger.info("No profiling data available")
            return

        logger.info("=" * 60)
        logger.info("Memory Profiling Summary")
        logger.info("=" * 60)

        for name, stats in self.stats.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Duration: {stats['duration_seconds']:.4f}s")

            if "memory_diff" in stats:
                for key, value in stats["memory_diff"].items():
                    mb_value = value / (1024 * 1024)
                    logger.info(f"  {key}: {mb_value:+.2f} MB")


class PerformanceBenchmark:
    """Benchmark training performance

    Measures throughput, latency, and other performance metrics.

    Example:
        >>> benchmark = PerformanceBenchmark()
        >>> benchmark.start_epoch()
        >>> for batch in dataloader:
        ...     benchmark.start_batch()
        ...     # Training code
        ...     benchmark.end_batch(batch_size=32)
        >>> benchmark.end_epoch()
        >>> benchmark.print_summary()
    """

    def __init__(self):
        """Initialize benchmark"""
        self.epoch_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None
        self.epoch_times: list[float] = []
        self.batch_times: list[float] = []
        self.samples_processed: int = 0
        self.batches_processed: int = 0

    def start_epoch(self) -> None:
        """Start epoch timer"""
        self.epoch_start_time = time.time()

    def end_epoch(self) -> None:
        """End epoch timer"""
        if self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time
            self.epoch_times.append(elapsed)
            self.epoch_start_time = None

    def start_batch(self) -> None:
        """Start batch timer"""
        self.batch_start_time = time.time()

    def end_batch(self, batch_size: int) -> None:
        """End batch timer

        Args:
            batch_size: Number of samples in batch
        """
        if self.batch_start_time is not None:
            elapsed = time.time() - self.batch_start_time
            self.batch_times.append(elapsed)
            self.samples_processed += batch_size
            self.batches_processed += 1
            self.batch_start_time = None

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics

        Returns:
            Dictionary with performance metrics
        """
        stats = {}

        if self.epoch_times:
            stats["avg_epoch_time"] = sum(self.epoch_times) / len(self.epoch_times)
            stats["total_epochs"] = len(self.epoch_times)

        if self.batch_times:
            stats["avg_batch_time"] = sum(self.batch_times) / len(self.batch_times)
            stats["total_batches"] = len(self.batch_times)
            stats["samples_per_second"] = self.samples_processed / sum(self.batch_times)
            stats["batches_per_second"] = self.batches_processed / sum(self.batch_times)

        stats["total_samples"] = self.samples_processed

        return stats

    def reset(self) -> None:
        """Reset all statistics"""
        self.epoch_times.clear()
        self.batch_times.clear()
        self.samples_processed = 0
        self.batches_processed = 0

    def print_summary(self) -> None:
        """Print benchmark summary"""
        stats = self.get_stats()

        # Check if we have meaningful data (not just total_samples: 0)
        has_data = len(stats) > 1 or (
            len(stats) == 1 and "total_samples" in stats and stats["total_samples"] > 0
        )

        if not has_data:
            logger.info("No benchmark data available")
            return

        logger.info("=" * 60)
        logger.info("Performance Benchmark Summary")
        logger.info("=" * 60)

        if "avg_epoch_time" in stats:
            logger.info(f"Average epoch time: {stats['avg_epoch_time']:.2f}s")
            logger.info(f"Total epochs: {stats['total_epochs']}")

        if "avg_batch_time" in stats:
            logger.info(f"Average batch time: {stats['avg_batch_time']:.4f}s")
            logger.info(f"Samples per second: {stats['samples_per_second']:.2f}")
            logger.info(f"Batches per second: {stats['batches_per_second']:.2f}")
            logger.info(f"Total samples: {stats['total_samples']}")


def optimize_dataloader_settings(device: str, batch_size: int, dataset_size: int) -> Dict[str, Any]:
    """Get optimized DataLoader settings

    Recommends optimal settings for num_workers, pin_memory, and prefetch_factor
    based on device and dataset characteristics.

    Args:
        device: Device string (e.g., "cuda:0", "mps", "cpu")
        batch_size: Batch size
        dataset_size: Total dataset size

    Returns:
        Dictionary with recommended DataLoader settings
    """
    import multiprocessing

    device_type = device.split(":")[0]
    cpu_count = multiprocessing.cpu_count()

    # Default settings
    settings = {
        "pin_memory": False,
        "num_workers": 0,
        "persistent_workers": False,
        "prefetch_factor": None,
    }

    # Enable pin_memory for CUDA
    if device_type == "cuda":
        settings["pin_memory"] = True

    # Determine optimal num_workers
    # Rule of thumb: 4 workers per GPU, but cap at cpu_count - 1
    if device_type in ["cuda", "mps"]:
        # Use multiple workers for GPU training
        settings["num_workers"] = min(4, max(1, cpu_count - 1))

        # Enable persistent workers if we have multiple workers
        if settings["num_workers"] > 0:
            settings["persistent_workers"] = True
            settings["prefetch_factor"] = 2
    else:
        # CPU training - use fewer workers to avoid overhead
        settings["num_workers"] = min(2, max(0, cpu_count // 2))

        if settings["num_workers"] > 0:
            settings["persistent_workers"] = True
            settings["prefetch_factor"] = 2

    # For small datasets, reduce workers
    batches = dataset_size // batch_size
    if batches < 10:
        settings["num_workers"] = 0
        settings["persistent_workers"] = False
        settings["prefetch_factor"] = None

    logger.info(f"Optimized DataLoader settings for {device}:")
    logger.info(f"  num_workers: {settings['num_workers']}")
    logger.info(f"  pin_memory: {settings['pin_memory']}")
    logger.info(f"  persistent_workers: {settings['persistent_workers']}")
    logger.info(f"  prefetch_factor: {settings['prefetch_factor']}")

    return settings


class OptimizedDataLoader:
    """Memory-efficient DataLoader wrapper

    Automatically applies optimization settings based on device and dataset.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = OptimizedDataLoader(
        ...     dataset=dataset,
        ...     batch_size=32,
        ...     device="cuda:0"
        ... )
        >>> for batch in loader:
        ...     # Training code
        ...     pass
    """

    def __init__(self, dataset: Any, batch_size: int, device: str, shuffle: bool = True, **kwargs):
        """Initialize optimized DataLoader

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            device: Device string
            shuffle: Whether to shuffle data
            **kwargs: Additional DataLoader arguments (override optimizations)
        """
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("PyTorch is required for OptimizedDataLoader")

        # Get optimized settings
        optimized_settings = optimize_dataloader_settings(
            device=device, batch_size=batch_size, dataset_size=len(dataset)
        )

        # Merge with user-provided kwargs (user kwargs take precedence)
        final_kwargs = {**optimized_settings, **kwargs}

        # Fix: If num_workers is 0, can't use prefetch_factor or persistent_workers
        if final_kwargs.get("num_workers", 0) == 0:
            final_kwargs["prefetch_factor"] = None
            final_kwargs["persistent_workers"] = False

        # Create DataLoader
        self.dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, **final_kwargs
        )

        self.device = device
        self.batch_size = batch_size

    def __iter__(self):
        """Iterate over batches"""
        return iter(self.dataloader)

    def __len__(self):
        """Get number of batches"""
        return len(self.dataloader)


def benchmark_device_performance(device: str, model: Any, input_shape: tuple) -> Dict[str, float]:
    """Benchmark device performance

    Runs inference to measure device throughput.

    Args:
        device: Device string
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, ...)

    Returns:
        Dictionary with benchmark results
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available for benchmarking")
        return {}

    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Synchronize if CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    # Synchronize if CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    batch_size = input_shape[0]
    samples_per_second = (num_iterations * batch_size) / elapsed
    ms_per_batch = (elapsed / num_iterations) * 1000

    results = {
        "samples_per_second": samples_per_second,
        "ms_per_batch": ms_per_batch,
        "total_iterations": num_iterations,
        "total_time_seconds": elapsed,
    }

    logger.info(f"Device {device} benchmark:")
    logger.info(f"  Samples/sec: {samples_per_second:.2f}")
    logger.info(f"  ms/batch: {ms_per_batch:.2f}")

    return results
