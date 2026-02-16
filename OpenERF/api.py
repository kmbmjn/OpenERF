"""Public API for OpenERF."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import Module

from .erf import ERFResult, compute_erf
from .metrics import compute_erf_metrics, save_metrics_json
from .model_zoo import get_model_family, infer_target_layer
from .visualization import save_erf_png


def _resolve_save_path(model_name: str, save_dir: str | Path | None) -> Path:
    default_path = Path("./results") / f"OpenERF_{model_name}.png"
    if save_dir is None:
        return default_path

    candidate = Path(save_dir).expanduser()
    if candidate.suffix.lower() == ".png":
        return candidate
    return candidate / f"OpenERF_{model_name}.png"


def _artifact_base_name(png_path: Path) -> str:
    name = png_path.name
    if name.lower().endswith(".png"):
        return name[:-4]
    return png_path.stem


def _resolve_artifact_path(base_name: str, suffix: str, artifact_dir: str | Path | None, fallback_dir: Path) -> Path:
    directory = Path(artifact_dir).expanduser() if artifact_dir is not None else fallback_dir
    return directory / (base_name + suffix)


def compute_ERF(
    model: Module,
    model_name: str | None = None,
    image_dir: str | Path = "./imagenet_val_1000",
    max_images: int | None = None,
    target_layer: str | None = None,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    show_progress: bool = True,
    fit_gaussian: bool = False,
) -> ERFResult:
    """Compute ERF map for timm models."""
    resolved_target_layer = infer_target_layer(model_name=model_name, target_layer=target_layer)
    return compute_erf(
        model=model,
        image_dir=image_dir,
        max_images=max_images,
        target_layer=resolved_target_layer,
        device=device,
        num_workers=num_workers,
        show_progress=show_progress,
        fit_gaussian=fit_gaussian,
    )


def save_ERF(
    model: Module,
    model_name: str = "model",
    source_model_name: str | None = None,
    image_dir: str | Path = "./imagenet_val_1000",
    save_dir: str | Path | None = None,
    npy_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    max_images: int | None = None,
    target_layer: str | None = None,
    device: str | torch.device | None = None,
    num_workers: int = 0,
    show_progress: bool = True,
    fit_gaussian: bool = False,
    colormap: str = "plasma",
    save_numpy: bool = False,
    save_metrics: bool = False,
    metrics_radii: tuple[int, ...] = (5, 10, 20, 30, 40, 60, 80),
) -> dict[str, Any]:
    """
    Compute ERF and save PNG to disk.

    Default output paths:

    - PNG:     ``./results/OpenERF_<model_name>.png``
    - npy:     ``./results_npy/OpenERF_<model_name>.npy``  (when *save_numpy=True*)
    - metrics: ``./results_metrics/OpenERF_<model_name>_metrics.json``  (when *save_metrics=True*)
    """
    resolved_source_model_name = source_model_name or model_name

    result = compute_ERF(
        model=model,
        model_name=resolved_source_model_name,
        image_dir=image_dir,
        max_images=max_images,
        target_layer=target_layer,
        device=device,
        num_workers=num_workers,
        show_progress=show_progress,
        fit_gaussian=fit_gaussian,
    )

    output_path = _resolve_save_path(model_name=model_name, save_dir=save_dir)
    save_erf_png(result.erf_map, output_path, colormap=colormap)
    base_name = _artifact_base_name(output_path)

    payload: dict[str, Any] = {
        "model_name": model_name,
        "source_model_name": resolved_source_model_name,
        "save_path": str(output_path),
        "num_images": result.num_images,
        "target_layer": result.resolved_target_layer,
        "family": get_model_family(resolved_source_model_name),
        "data_config": result.data_config,
        "colormap": colormap,
    }
    if result.gaussian_fit is not None:
        payload["gaussian_fit"] = asdict(result.gaussian_fit)

    if save_numpy:
        default_npy_dir = Path("./results_npy")
        npy_path = _resolve_artifact_path(base_name, ".npy", npy_dir, default_npy_dir)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, result.erf_map)
        payload["npy_path"] = str(npy_path)

    if save_metrics:
        metrics = compute_erf_metrics(result.erf_map, radii=metrics_radii)
        metrics["data_config"] = result.data_config
        if result.gaussian_fit is not None:
            metrics["gaussian_fit"] = asdict(result.gaussian_fit)
        default_metrics_dir = Path("./results_metrics")
        metrics_path = _resolve_artifact_path(base_name, "_metrics.json", metrics_dir, default_metrics_dir)
        save_metrics_json(metrics, metrics_path)
        payload["metrics_path"] = str(metrics_path)

    return payload
