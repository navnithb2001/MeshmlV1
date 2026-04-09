"""
Custom exceptions for MeshML API Gateway.
"""


class MeshMLException(Exception):
    """Base exception for all MeshML errors."""

    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class AuthenticationError(MeshMLException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", details: dict = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(MeshMLException):
    """Raised when user lacks permissions."""

    def __init__(self, message: str = "Insufficient permissions", details: dict = None):
        super().__init__(message, status_code=403, details=details)


class NotFoundError(MeshMLException):
    """Raised when resource is not found."""

    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, status_code=404)


class ConflictError(MeshMLException):
    """Raised when resource already exists."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=409, details=details)


class ValidationError(MeshMLException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)


class RateLimitError(MeshMLException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, status_code=429, details=details)


class ServiceUnavailableError(MeshMLException):
    """Raised when a dependent service is unavailable."""

    def __init__(self, service: str, details: dict = None):
        message = f"Service unavailable: {service}"
        super().__init__(message, status_code=503, details=details)
