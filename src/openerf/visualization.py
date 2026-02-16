"""Visualization helpers for ERF maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8_colormap_image(erf_map: np.ndarray, colormap: str = "plasma") -> np.ndarray:
    """Convert a 2D ERF map to an RGB uint8 image with a matplotlib colormap."""
    try:
        from matplotlib import colormaps
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for ERF PNG export. "
            "Install it with `pip install matplotlib`."
        ) from error

    normalized = np.asarray(erf_map, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError(f"erf_map must be 2D, got shape={normalized.shape}")

    normalized = normalized - float(normalized.min())
    max_value = float(normalized.max())
    if max_value > 0:
        normalized = normalized / max_value

    try:
        rgb_float = colormaps[colormap](normalized)[..., :3]
    except KeyError as error:
        raise ValueError(f"Unknown matplotlib colormap: '{colormap}'") from error

    return np.asarray(rgb_float * 255.0, dtype=np.uint8)


def save_erf_png(
    erf_map: np.ndarray,
    save_path: str | Path,
    colormap: str = "plasma",
) -> Path:
    """Save an ERF map as a colored PNG."""
    output_path = Path(save_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(to_uint8_colormap_image(erf_map, colormap=colormap), mode="RGB")
    image.save(output_path)
    return output_path
