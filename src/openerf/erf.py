"""ERF computation core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .data import FlatImageDataset, build_timm_eval_transform
from .feature_ops import center_activation, extract_feature_tensor
from .fit import GaussianFitResult, fit_2d_gaussian


@dataclass(frozen=True)
class ERFResult:
    """Computation outputs for an ERF run."""

    erf_map: np.ndarray
    num_images: int
    data_config: dict[str, Any]
    resolved_target_layer: str
    gaussian_fit: GaussianFitResult | None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_child_module(module: Module, part: str) -> Module:
    if part in module._modules:
        return module._modules[part]

    if hasattr(module, part):
        child = getattr(module, part)
        if isinstance(child, Module):
            return child
        raise ValueError(f"Resolved object '{part}' is not a torch module.")

    if part.isdigit():
        index = int(part)
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            try:
                return module[index]
            except IndexError as error:
                raise ValueError(
                    f"Layer index '{index}' is out of range for '{type(module).__name__}'."
                ) from error

    raise ValueError(
        f"Layer component '{part}' was not found in '{type(module).__name__}'."
    )


def _get_module_by_name(model: Module, module_name: str) -> Module:
    module: Module = model
    for part in module_name.split("."):
        module = _resolve_child_module(module, part)
    return module


def _normalize_erf_map(erf_map: np.ndarray) -> np.ndarray:
    total = float(erf_map.sum())
    if total <= 0:
        return np.zeros_like(erf_map, dtype=np.float64)
    return erf_map / total


def _compute_positive_input_gradients_from_layer(
    model: Module,
    inputs: Tensor,
    target_module: Module,
    source_name: str,
) -> Tensor:
    feature_store: dict[str, Tensor] = {}

    def _forward_hook(
        _: Module,
        __: tuple[Any, ...],
        output: Tensor | tuple[Tensor, ...] | list[Tensor],
    ) -> None:
        feature_store["feature"] = extract_feature_tensor(output)

    hook_handle = target_module.register_forward_hook(_forward_hook)
    try:
        _ = model(inputs)
        if "feature" not in feature_store:
            raise RuntimeError(
                f"Feature source '{source_name}' did not produce a stored feature map."
            )
        activation = center_activation(
            feature_store["feature"],
            source_name=source_name,
            model=model,
        )
        gradients = torch.autograd.grad(
            outputs=activation,
            inputs=inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    finally:
        hook_handle.remove()

    return gradients.detach().mean(dim=(0, 1)).clamp(min=0)


def _compute_positive_input_gradients_from_forward_features(
    model: Module,
    inputs: Tensor,
    source_name: str,
) -> Tensor:
    if not hasattr(model, "forward_features"):
        raise ValueError(
            f"Model '{type(model).__name__}' does not expose forward_features(). "
            "Set target_layer to a valid module path."
        )
    output = model.forward_features(inputs)
    feature_map = extract_feature_tensor(output)
    activation = center_activation(feature_map, source_name=source_name, model=model)
    gradients = torch.autograd.grad(
        outputs=activation,
        inputs=inputs,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    return gradients.detach().mean(dim=(0, 1)).clamp(min=0)


def _build_progress_iterator(
    dataloader: DataLoader[tuple[Tensor, str]],
    show_progress: bool,
):
    iterator: Any = dataloader
    if not show_progress:
        return iterator

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc="OpenERF",
            unit="img",
        )
    except Exception:
        iterator = dataloader
    return iterator


def compute_erf(
    model: Module,
    image_dir: str | Path = "./imagenet_val_1000",
    max_images: int | None = None,
    target_layer: str = "forward_features",
    device: str | torch.device | None = None,
    num_workers: int = 0,
    show_progress: bool = True,
    fit_gaussian: bool = False,
) -> ERFResult:
    """
    Compute ERF for a timm model.

    ERF is accumulated from positive gradients of the center activation
    in the selected feature source, with batch size fixed to 1.
    """
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    previous_training_mode = model.training
    model.eval()

    transform, data_config = build_timm_eval_transform(model)
    dataset = FlatImageDataset(
        image_dir=image_dir,
        transform=transform,
        max_images=max_images,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
    )

    use_forward_features = target_layer == "forward_features"
    target_module = None if use_forward_features else _get_module_by_name(model, target_layer)

    erf_accumulator: Tensor | None = None
    num_processed = 0

    try:
        iterator = _build_progress_iterator(dataloader=dataloader, show_progress=show_progress)
        with torch.enable_grad():
            for batch_inputs, _ in iterator:
                if batch_inputs.shape[0] != 1:
                    raise ValueError("ERF computation requires batch_size=1.")
                inputs = batch_inputs.to(resolved_device, non_blocking=True)
                inputs.requires_grad_(True)

                if use_forward_features:
                    grad_map = _compute_positive_input_gradients_from_forward_features(
                        model=model,
                        inputs=inputs,
                        source_name="forward_features",
                    )
                else:
                    assert target_module is not None
                    grad_map = _compute_positive_input_gradients_from_layer(
                        model=model,
                        inputs=inputs,
                        target_module=target_module,
                        source_name=target_layer,
                    )

                if erf_accumulator is None:
                    erf_accumulator = torch.zeros_like(
                        grad_map,
                        dtype=torch.float32,
                        device=grad_map.device,
                    )
                erf_accumulator += grad_map.to(dtype=torch.float32)
                num_processed += 1
    finally:
        model.train(previous_training_mode)

    if num_processed == 0 or erf_accumulator is None:
        raise RuntimeError("No images were processed. Check dataset path and image files.")

    erf_array = erf_accumulator.detach().cpu().numpy().astype(np.float64)
    normalized_erf = _normalize_erf_map(erf_array)
    fit_result = fit_2d_gaussian(normalized_erf) if fit_gaussian else None
    return ERFResult(
        erf_map=normalized_erf,
        num_images=num_processed,
        data_config=data_config,
        resolved_target_layer=target_layer,
        gaussian_fit=fit_result,
    )

