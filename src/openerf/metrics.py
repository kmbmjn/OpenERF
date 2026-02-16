"""Metrics utilities for ERF maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def compute_erf_metrics(
    erf_map: np.ndarray,
    radii: Iterable[int] = (5, 10, 20, 30, 40, 60, 80),
) -> dict[str, Any]:
    """
    Compute compact ERF diagnostics.

    Metrics are based on the normalized ERF map (sum == 1).
    """
    array = np.asarray(erf_map, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"erf_map must be 2D, got shape={array.shape}")

    total = float(array.sum())
    if total <= 0:
        raise ValueError("Cannot compute metrics because ERF map sum is non-positive.")
    normalized = array / total

    height, width = normalized.shape
    center_y = height // 2
    center_x = width // 2

    y_indices, x_indices = np.indices((height, width))
    com_x = float((x_indices * normalized).sum())
    com_y = float((y_indices * normalized).sum())

    distances = np.sqrt(
        (x_indices - float(center_x)) ** 2 + (y_indices - float(center_y)) ** 2
    )
    radial_cumulative_mass: dict[str, float] = {}
    for radius in radii:
        if radius <= 0:
            raise ValueError("All radii must be positive integers.")
        key = str(int(radius))
        radial_cumulative_mass[key] = float(normalized[distances <= float(radius)].sum())

    peak_y, peak_x = np.unravel_index(np.argmax(normalized), normalized.shape)
    return {
        "shape": [int(height), int(width)],
        "sum": float(normalized.sum()),
        "min": float(normalized.min()),
        "max": float(normalized.max()),
        "argmax_yx": [int(peak_y), int(peak_x)],
        "center_yx": [int(center_y), int(center_x)],
        "center_value": float(normalized[center_y, center_x]),
        "com_x": com_x,
        "com_y": com_y,
        "radial_cumulative_mass": radial_cumulative_mass,
    }


def save_metrics_json(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Save metrics as a JSON file."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    return output_path
