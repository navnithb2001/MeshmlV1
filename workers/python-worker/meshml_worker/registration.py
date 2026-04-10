"""
Worker Registration and Group Management

Handles:
- Worker registration with Leader service
- Group discovery and joining
- Invitation acceptance
- Worker authentication
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class WorkerRegistration:
    """Handle worker registration and group management

    Features:
    - Register worker with Leader service
    - Join groups via invitation code
    - Discover available groups
    - Update worker capabilities
    """

    def __init__(self, config: Any):
        """Initialize registration manager

        Args:
            config: Worker configuration
        """
        self.config = config
        self.api_base_url = getattr(config, "api_base_url", "http://localhost:8000")
        self.worker_id = getattr(config.worker, "id", None) or getattr(
            config.worker, "worker_id", None
        )
        if not self.worker_id:
            self.worker_id = "worker-temp"
        self.user_email = getattr(config.worker, "user_email", None)
        self.registered = False
        self.current_group_id: Optional[str] = None
        self.auth_token: Optional[str] = None

        # Try to load saved auth token
        saved_token = self._load_auth_token()
        if saved_token:
            self.auth_token = saved_token
            logger.info(f"Loaded saved authentication for {self.user_email}")

        logger.info(f"Initialized registration manager for worker {self.worker_id}")

    def register_worker(self, capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register worker with Leader service

        Args:
            capabilities: Worker capabilities (GPU, RAM, CPU, etc.)

        Returns:
            Registration response with worker ID and auth token

        Raises:
            RuntimeError: If registration fails
        """
        if capabilities is None:
            capabilities = self._detect_capabilities()

        logger.info(f"Registering worker {self.worker_id}...")

        payload = {
            "worker_id": self.worker_id,
            "user_email": self.user_email,
            "capabilities": capabilities,
            "status": "idle",
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/workers/register", json=payload, timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.auth_token = data.get("auth_token")
            self.registered = True

            logger.info(f"Successfully registered worker: {data.get('worker_id')}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register worker: {e}")
            raise RuntimeError(f"Worker registration failed: {e}")

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and get JWT token

        Args:
            email: User email
            password: User password

        Returns:
            Login response with user info and auth token

        Raises:
            RuntimeError: If login fails
        """
        logger.info(f"Logging in user: {email}...")

        try:
            response = requests.post(
                f"{self.api_base_url}/api/auth/login",
                json={"email": email, "password": password},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            self.auth_token = data.get("access_token")
            self.user_email = email

            # Save auth token to config
            if self.auth_token:
                self._save_auth_token(self.auth_token)

            logger.info(f"Successfully logged in as: {email}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to login: {e}")
            raise RuntimeError(f"Login failed: {e}")

    def join_group_by_invitation(self, invitation_code: str) -> Dict[str, Any]:
        """Join a group using invitation code

        Args:
            invitation_code: Invitation code or link

        Returns:
            Group information

        Raises:
            RuntimeError: If joining fails
        """
        logger.info(f"Joining group with invitation: {invitation_code[:8]}...")

        payload = {"worker_id": self.worker_id, "invitation_code": invitation_code}

        headers = self._get_auth_headers()

        try:
            response = requests.post(
                f"{self.api_base_url}/api/invitations/accept",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            self.current_group_id = data.get("group_id")

            # Save group info to config
            self._save_group_info(data)

            logger.info(f"Successfully joined group: {data.get('group_name')}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to join group: {e}")
            raise RuntimeError(f"Group joining failed: {e}")

    def discover_public_groups(self) -> List[Dict[str, Any]]:
        """Discover public groups available to join

        Returns:
            List of public groups
        """
        logger.info("Discovering public groups...")

        headers = self._get_auth_headers()

        try:
            response = requests.get(
                f"{self.api_base_url}/api/groups/public", headers=headers, timeout=10
            )
            response.raise_for_status()

            groups = response.json().get("groups", [])
            logger.info(f"Found {len(groups)} public groups")
            return groups

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to discover groups: {e}")
            return []

    def join_public_group(self, group_id: str) -> Dict[str, Any]:
        """Join a public group (no invitation required)

        Args:
            group_id: Group ID to join

        Returns:
            Group information

        Raises:
            RuntimeError: If joining fails
        """
        logger.info(f"Joining public group {group_id}...")

        payload = {"worker_id": self.worker_id}

        headers = self._get_auth_headers()

        try:
            response = requests.post(
                f"{self.api_base_url}/api/groups/{group_id}/join",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            self.current_group_id = group_id

            # Save group info to config
            self._save_group_info(data)

            logger.info(f"Successfully joined group: {data.get('group_name')}")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to join public group: {e}")
            raise RuntimeError(f"Group joining failed: {e}")

    def get_group_jobs(self, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available jobs for current group

        Args:
            group_id: Group ID (defaults to current group)

        Returns:
            List of available jobs
        """
        group_id = group_id or self.current_group_id
        if not group_id:
            logger.warning("No group assigned")
            return []

        logger.info(f"Fetching jobs for group {group_id}...")

        headers = self._get_auth_headers()

        try:
            response = requests.get(
                f"{self.api_base_url}/api/groups/{group_id}/jobs", headers=headers, timeout=10
            )
            response.raise_for_status()

            jobs = response.json().get("jobs", [])
            logger.info(f"Found {len(jobs)} jobs")
            return jobs

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch jobs: {e}")
            return []

    def request_job_assignment(self, group_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Request job assignment from orchestrator

        Args:
            group_id: Group ID (defaults to current group)

        Returns:
            Assigned job or None if no jobs available
        """
        group_id = group_id or self.current_group_id
        if not group_id:
            raise RuntimeError("No group assigned. Join a group first.")

        logger.info(f"Requesting job assignment from group {group_id}...")

        payload = {
            "worker_id": self.worker_id,
            "group_id": group_id,
            "capabilities": self._detect_capabilities(),
        }

        headers = self._get_auth_headers()

        try:
            response = requests.post(
                f"{self.api_base_url}/api/orchestrator/assign-job",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            if data.get("job_assigned"):
                logger.info(f"Assigned to job: {data.get('job_id')}")
                return data
            else:
                logger.info("No jobs available")
                return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to request job: {e}")
            return None

    def update_capabilities(self, capabilities: Dict[str, Any]) -> None:
        """Update worker capabilities

        Args:
            capabilities: New capabilities
        """
        logger.info("Updating worker capabilities...")

        payload = {"worker_id": self.worker_id, "capabilities": capabilities}

        headers = self._get_auth_headers()

        try:
            response = requests.put(
                f"{self.api_base_url}/api/workers/{self.worker_id}/capabilities",
                json=payload,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Capabilities updated successfully")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to update capabilities: {e}")

    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect worker capabilities

        Returns:
            Capabilities dictionary
        """
        import psutil
        import torch

        capabilities = {
            "device": "cpu",
            "cuda_available": False,
            "mps_available": False,
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "storage_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
        }

        # Check CUDA
        if torch.cuda.is_available():
            capabilities["device"] = "cuda"
            capabilities["cuda_available"] = True
            capabilities["gpu_name"] = torch.cuda.get_device_name(0)
            capabilities["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            capabilities["device"] = "mps"
            capabilities["mps_available"] = True

        return capabilities

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers

        Returns:
            Headers dictionary
        """
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _save_auth_token(self, token: str) -> None:
        """Save authentication token to config

        Args:
            token: JWT authentication token
        """
        config_dir = Path.home() / ".meshml"
        config_dir.mkdir(exist_ok=True)

        auth_file = config_dir / "auth.json"
        with open(auth_file, "w") as f:
            json.dump({"token": token, "email": self.user_email}, f, indent=2)

        logger.debug("Saved authentication token")

    def _load_auth_token(self) -> Optional[str]:
        """Load authentication token from config

        Returns:
            JWT token if available
        """
        config_dir = Path.home() / ".meshml"
        auth_file = config_dir / "auth.json"

        if not auth_file.exists():
            return None

        try:
            with open(auth_file, "r") as f:
                data = json.load(f)
                self.user_email = data.get("email")
                return data.get("token")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load auth token: {e}")
            return None

    def get_user_id_from_token(self) -> Optional[str]:
        """Extract user ID from JWT token

        Returns:
            User ID if token is valid and contains user info
        """
        if not self.auth_token:
            return None

        try:
            # JWT tokens have 3 parts: header.payload.signature
            # We decode the payload (middle part) which contains user info
            import base64

            parts = self.auth_token.split(".")
            if len(parts) != 3:
                logger.warning("Invalid JWT token format")
                return None

            # Decode payload (add padding if needed)
            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            payload_data = json.loads(decoded)

            # JWT tokens typically have 'sub' (subject) field with user ID
            # or 'user_id' field
            user_id = payload_data.get("sub") or payload_data.get("user_id")

            if user_id:
                logger.debug(f"Extracted user ID from token: {user_id}")
                return str(user_id)
            else:
                logger.warning("No user ID found in JWT token payload")
                return None

        except Exception as e:
            logger.error(f"Failed to decode JWT token: {e}")
            return None

    def _save_group_info(self, group_data: Dict[str, Any]) -> None:
        """Save group information to config

        Args:
            group_data: Group information
        """
        config_dir = Path.home() / ".meshml"
        config_dir.mkdir(exist_ok=True)

        group_file = config_dir / "current_group.json"
        with open(group_file, "w") as f:
            json.dump(group_data, f, indent=2)

        logger.debug(f"Saved group info to {group_file}")

    def load_saved_group(self) -> Optional[Dict[str, Any]]:
        """Load previously saved group information

        Returns:
            Group data or None if not found
        """
        group_file = Path.home() / ".meshml" / "current_group.json"
        if group_file.exists():
            with open(group_file, "r") as f:
                data = json.load(f)
                self.current_group_id = data.get("group_id")
                return data
        return None


def interactive_registration(config: Any) -> WorkerRegistration:
    """Interactive worker registration and group joining

    Args:
        config: Worker configuration

    Returns:
        Configured WorkerRegistration instance
    """
    registration = WorkerRegistration(config)

    print("\n" + "=" * 50)
    print("   MeshML Worker Registration")
    print("=" * 50 + "\n")

    # Register worker
    print("Registering worker...")
    try:
        registration.register_worker()
        print(f"✓ Worker registered: {registration.worker_id}\n")
    except Exception as e:
        print(f"✗ Registration failed: {e}")
        return registration

    # Check for saved group
    saved_group = registration.load_saved_group()
    if saved_group:
        print(f"Found saved group: {saved_group.get('group_name')}")
        use_saved = input("Continue with this group? [Y/n]: ").strip().lower()
        if use_saved != "n":
            print(f"✓ Using group: {saved_group.get('group_name')}\n")
            return registration

    # Group joining options
    print("How would you like to join a group?\n")
    print("1. Enter invitation code")
    print("2. Browse public groups")
    print("3. Skip (configure later)")
    print()

    choice = input("Choose option [1-3]: ").strip()

    if choice == "1":
        # Join by invitation
        invitation_code = input("Enter invitation code: ").strip()
        try:
            group = registration.join_group_by_invitation(invitation_code)
            print(f"\n✓ Joined group: {group.get('group_name')}")
        except Exception as e:
            print(f"\n✗ Failed to join group: {e}")

    elif choice == "2":
        # Browse public groups
        groups = registration.discover_public_groups()
        if groups:
            print("\nAvailable groups:\n")
            for i, group in enumerate(groups, 1):
                print(f"{i}. {group.get('name')} - {group.get('description', 'No description')}")
            print()

            group_choice = input(f"Select group [1-{len(groups)}]: ").strip()
            try:
                idx = int(group_choice) - 1
                if 0 <= idx < len(groups):
                    selected_group = groups[idx]
                    registration.join_public_group(selected_group["id"])
                    print(f"\n✓ Joined group: {selected_group.get('name')}")
            except (ValueError, IndexError):
                print("\n✗ Invalid selection")
        else:
            print("\nNo public groups found")

    print()
    return registration
