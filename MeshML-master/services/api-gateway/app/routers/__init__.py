"""Routers package"""

from . import (
    auth,
    datasets,
    groups,
    invitations,
    jobs,
    models,
    monitoring,
    parameters,
    stats_ws,
    workers,
)

__all__ = [
    "auth",
    "groups",
    "invitations",
    "jobs",
    "workers",
    "monitoring",
    "datasets",
    "parameters",
    "models",
    "stats_ws",
]
