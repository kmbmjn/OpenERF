"""Feature extraction helpers for ERF center-activation selection."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module


def extract_feature_tensor(output: Tensor | tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    """Normalize hook/forward output into a single tensor."""
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("Target layer output is empty.")
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError(f"Target layer output must be torch.Tensor, got {type(output)!r}.")
    return output


def _center_spatial_token_index(num_spatial_tokens: int) -> int:
    """
    Match the center token rule from the official BMVC2023 ERF code.

    For even grids (e.g. 14x14), this selects the upper-left token among the
    2x2 central patches.
    """
    if num_spatial_tokens <= 0:
        raise ValueError("num_spatial_tokens must be positive.")
    grid_size = int(math.sqrt(num_spatial_tokens))
    if grid_size * grid_size != num_spatial_tokens:
        raise ValueError(
            "Number of spatial tokens must form a square grid for ViT ERF. "
            f"Got {num_spatial_tokens}."
        )
    center_index = int(num_spatial_tokens / 2) - ((num_spatial_tokens + 1) % 2) * (
        int(grid_size / 2) + 1
    )
    return int(center_index)


def _infer_num_prefix_tokens(model: Module, num_tokens: int) -> int:
    """
    Infer prefix-token count for transformer families.

    This follows the BMVC2023 ERF protocol where class/distill/prefix tokens
    are excluded when selecting the spatial center token.
    """
    explicit = getattr(model, "num_prefix_tokens", None)
    if explicit is not None:
        prefix = int(explicit)
    else:
        prefix = 0
        cls_token = getattr(model, "cls_token", None)
        if cls_token is not None:
            prefix += 1
        dist_token = getattr(model, "dist_token", None)
        if dist_token is not None:
            prefix += 1
        num_reg_tokens = getattr(model, "num_reg_tokens", None)
        if num_reg_tokens is not None:
            prefix += int(num_reg_tokens)

    if prefix < 0 or prefix >= num_tokens:
        raise ValueError(
            f"Invalid inferred prefix token count={prefix} for token count={num_tokens}."
        )
    return prefix


def center_activation(feature_map: Tensor, source_name: str, model: Module) -> Tensor:
    """Select the center activation scalar used for ERF gradient extraction."""
    if feature_map.ndim == 4:
        # Some transformer families (e.g., Swin) expose NHWC or NCHW 4D
        # features from forward_features. We convert to token order and apply
        # the same center-token rule used in the BMVC2023 reference code.
        if source_name == "forward_features" and hasattr(model, "patch_embed"):
            if (
                getattr(model, "output_fmt", None) == "NHWC"
                or (
                    feature_map.shape[1] <= 16
                    and feature_map.shape[2] <= 16
                    and feature_map.shape[3] > 16
                )
            ):
                _, feat_h, feat_w, feat_c = feature_map.shape
                token_map = feature_map.reshape(1, feat_h * feat_w, feat_c)
            else:
                _, feat_c, feat_h, feat_w = feature_map.shape
                token_map = feature_map.permute(0, 2, 3, 1).reshape(
                    1, feat_h * feat_w, feat_c
                )

            center_token_idx = _center_spatial_token_index(feat_h * feat_w)
            return token_map[0, center_token_idx, :].mean()

        _, _, feat_h, feat_w = feature_map.shape
        center_y = feat_h // 2
        center_x = feat_w // 2
        return feature_map[0, :, center_y, center_x].mean()

    if feature_map.ndim == 3:
        _, num_tokens, _ = feature_map.shape
        num_prefix_tokens = _infer_num_prefix_tokens(model=model, num_tokens=num_tokens)
        num_spatial_tokens = num_tokens - num_prefix_tokens
        center_spatial_idx = _center_spatial_token_index(num_spatial_tokens)
        center_token_idx = num_prefix_tokens + center_spatial_idx
        return feature_map[0, center_token_idx, :].mean()

    raise ValueError(
        "This implementation expects either 4D CNN features [N, C, H, W] "
        "or 3D token features [N, T, C]. "
        f"Got shape={tuple(feature_map.shape)} for source '{source_name}'."
    )
