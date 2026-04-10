"""
Heartbeat Sender for Worker Status Monitoring

Periodically sends heartbeat signals to Parameter Server to indicate
the worker is alive and active.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class HeartbeatSender:
    """Send periodic heartbeat signals to Parameter Server

    Features:
    - Periodic heartbeat transmission
    - Worker status reporting
    - Training metrics included
    - Automatic retry on failure
    - Thread-safe operation
    """

    def __init__(
        self, worker_id: str, heartbeat_interval: int = 30, timeout: int = 10, max_retries: int = 3
    ):
        """Initialize heartbeat sender

        Args:
            worker_id: Worker identifier
            heartbeat_interval: Interval between heartbeats in seconds
            timeout: Timeout for heartbeat request in seconds
            max_retries: Maximum retry attempts
        """
        self.worker_id = worker_id
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.max_retries = max_retries

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat: Optional[float] = None
        self._failed_attempts = 0

        # Heartbeat callback (set by worker)
        self._heartbeat_callback: Optional[Callable] = None

        # Current worker status
        self._status: Dict[str, Any] = {
            "state": "idle",
            "current_epoch": 0,
            "current_batch": 0,
            "total_batches": 0,
            "loss": 0.0,
            "metrics": {},
        }

        logger.info(f"Heartbeat sender initialized: interval={heartbeat_interval}s")

    def set_heartbeat_callback(self, callback: Callable) -> None:
        """Set callback function for sending heartbeats

        Args:
            callback: Function to call for sending heartbeat
                      Should accept (worker_id, status) and return bool
        """
        self._heartbeat_callback = callback
        logger.debug("Heartbeat callback set")

    def update_status(self, **kwargs) -> None:
        """Update worker status

        Args:
            **kwargs: Status fields to update (state, epoch, batch, loss, metrics)
        """
        self._status.update(kwargs)
        logger.debug(f"Status updated: {kwargs}")

    def start(self) -> None:
        """Start sending heartbeats"""
        if self._running:
            logger.warning("Heartbeat sender already running")
            return

        if self._heartbeat_callback is None:
            logger.warning("No heartbeat callback set, heartbeats will not be sent")

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

        logger.info("Heartbeat sender started")

    def stop(self) -> None:
        """Stop sending heartbeats"""
        if not self._running:
            return

        self._running = False

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("Heartbeat sender stopped")

    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop (runs in separate thread)"""
        while self._running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(self.heartbeat_interval)

    def _send_heartbeat(self) -> None:
        """Send a single heartbeat"""
        if self._heartbeat_callback is None:
            return

        for attempt in range(self.max_retries):
            try:
                # Prepare heartbeat data
                heartbeat_data = {
                    "worker_id": self.worker_id,
                    "timestamp": time.time(),
                    "status": self._status.copy(),
                }

                # Call callback
                success = self._heartbeat_callback(heartbeat_data)

                if success:
                    self._last_heartbeat = time.time()
                    self._failed_attempts = 0
                    logger.debug(f"Heartbeat sent successfully")
                    return
                else:
                    logger.warning(f"Heartbeat failed (attempt {attempt + 1}/{self.max_retries})")

            except Exception as e:
                logger.warning(f"Heartbeat error (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff

        # All retries failed
        self._failed_attempts += 1
        logger.error(
            f"Heartbeat failed after {self.max_retries} attempts "
            f"(total failures: {self._failed_attempts})"
        )

    def get_last_heartbeat_time(self) -> Optional[float]:
        """Get timestamp of last successful heartbeat

        Returns:
            Timestamp or None if no heartbeat sent yet
        """
        return self._last_heartbeat

    def is_healthy(self) -> bool:
        """Check if heartbeat sender is healthy

        Returns:
            True if heartbeats are being sent successfully
        """
        if not self._running:
            return False

        if self._last_heartbeat is None:
            return False

        # Check if last heartbeat was recent
        time_since_last = time.time() - self._last_heartbeat
        max_delay = self.heartbeat_interval * 3  # Allow 3 intervals

        if time_since_last > max_delay:
            logger.warning(f"Heartbeat unhealthy: last heartbeat {time_since_last:.0f}s ago")
            return False

        return True

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


def create_heartbeat_sender(worker_id: str, interval: int = 30) -> HeartbeatSender:
    """Create heartbeat sender instance

    Args:
        worker_id: Worker identifier
        interval: Heartbeat interval in seconds

    Returns:
        HeartbeatSender instance
    """
    return HeartbeatSender(worker_id=worker_id, heartbeat_interval=interval)
