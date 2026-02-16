"""Backward-compatible alias for legacy ``import OpenERF`` usage."""

from openerf import compute_erf, compute_erf_metrics, get_supported_models, save_erf
from openerf import __version__

compute_ERF = compute_erf
save_ERF = save_erf

__all__ = [
    "compute_erf",
    "save_erf",
    "compute_ERF",
    "save_ERF",
    "compute_erf_metrics",
    "get_supported_models",
]
