"""OpenERF public API."""

from .api import compute_erf, save_erf
from .metrics import compute_erf_metrics
from .model_zoo import get_supported_models

# Backward-compatible aliases.
compute_ERF = compute_erf
save_ERF = save_erf

__all__ = ["compute_erf", "save_erf", "compute_erf_metrics", "get_supported_models"]
__version__ = "0.6.1"
