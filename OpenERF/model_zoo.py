"""Supported timm model presets for OpenERF."""

from __future__ import annotations

from dataclasses import dataclass


ORDERED_FAMILIES: tuple[str, ...] = (
    "resnet",
    "resnext",
    "densenet",
    "efficientnet",
    "vit",
    "deit",
    "cait",
    "xcit",
    "beit",
    "swin",
    "swinv2",
)


SUPPORTED_MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    "resnet": (
        "resnet18.a1_in1k",
        "resnet34.a1_in1k",
        "resnet50.a1_in1k",
    ),
    "resnext": (
        "resnext50_32x4d.a1_in1k",
        "resnext50_32x4d.ra_in1k",
        "resnext101_32x8d.tv_in1k",
    ),
    "densenet": (
        "densenet121.ra_in1k",
        "densenet169.tv_in1k",
        "densenet201.tv_in1k",
    ),
    "efficientnet": (
        "efficientnet_b0.ra_in1k",
        "efficientnet_b2.ra_in1k",
        "tf_efficientnet_b4.ns_jft_in1k",
    ),
    "vit": (
        "vit_small_patch16_224.augreg_in1k",
        "vit_base_patch16_224.augreg_in1k",
        "vit_base_patch32_224.augreg_in1k",
    ),
    "deit": (
        "deit_tiny_patch16_224.fb_in1k",
        "deit_small_patch16_224.fb_in1k",
        "deit_base_patch16_224.fb_in1k",
    ),
    "cait": (
        "cait_xxs24_224.fb_dist_in1k",
        "cait_xxs36_224.fb_dist_in1k",
        "cait_s24_224.fb_dist_in1k",
    ),
    "xcit": (
        "xcit_tiny_12_p16_224.fb_in1k",
        "xcit_small_12_p16_224.fb_in1k",
        "xcit_medium_24_p16_224.fb_in1k",
    ),
    "beit": (
        "beit_base_patch16_224.in22k_ft_in22k",
        "beit_base_patch16_224.in22k_ft_in22k_in1k",
        "beit_base_patch16_384.in22k_ft_in22k_in1k",
    ),
    "swin": (
        "swin_tiny_patch4_window7_224.ms_in1k",
        "swin_small_patch4_window7_224.ms_in1k",
        "swin_base_patch4_window7_224.ms_in1k",
    ),
    "swinv2": (
        "swinv2_cr_tiny_ns_224.sw_in1k",
        "swinv2_cr_small_224.sw_in1k",
        "swinv2_cr_small_ns_224.sw_in1k",
    ),
}

DEFAULT_TARGET_LAYER_BY_FAMILY: dict[str, str] = {
    "resnet": "layer4",
    "resnext": "layer4",
    "densenet": "features",
    "efficientnet": "bn2",
    "vit": "norm",
    "deit": "norm",
    "cait": "forward_features",
    "xcit": "forward_features",
    "beit": "forward_features",
    "swin": "forward_features",
    "swinv2": "forward_features",
}


@dataclass(frozen=True)
class ModelSpec:
    """OpenERF preset metadata for a model."""

    model_name: str
    family: str
    target_layer: str


SUPPORTED_MODEL_SPECS: dict[str, ModelSpec] = {
    model_name: ModelSpec(
        model_name=model_name,
        family=family,
        target_layer=DEFAULT_TARGET_LAYER_BY_FAMILY[family],
    )
    for family, model_names in SUPPORTED_MODEL_GROUPS.items()
    for model_name in model_names
}


def get_supported_models(family: str | None = None) -> list[str]:
    """Return supported model names for all families or a specific family."""
    if family is None:
        models: list[str] = []
        for family_name in ORDERED_FAMILIES:
            models.extend(SUPPORTED_MODEL_GROUPS[family_name])
        return models

    if family not in SUPPORTED_MODEL_GROUPS:
        raise ValueError(
            f"Unknown family '{family}'. Expected one of: "
            + ", ".join(sorted(SUPPORTED_MODEL_GROUPS))
        )
    return list(SUPPORTED_MODEL_GROUPS[family])


def resolve_supported_model_name(model_name: str | None) -> str | None:
    """
    Resolve a model name to a supported preset identifier when possible.

    In addition to exact matches, this accepts artifact-suffixed names such as
    ``resnet50.a1_in1k_run1`` and resolves them to ``resnet50.a1_in1k``.
    """
    if model_name is None:
        return None
    if model_name in SUPPORTED_MODEL_SPECS:
        return model_name

    # Prefer the longest matching prefix to avoid accidental short matches.
    candidates = [
        supported_name
        for supported_name in SUPPORTED_MODEL_SPECS
        if model_name.startswith(supported_name)
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def infer_target_layer(model_name: str | None, target_layer: str | None) -> str:
    """
    Resolve which feature source to use for ERF.

    If a target layer is explicitly provided (and not ``auto``), it is used as-is.
    Otherwise, known presets are used and unknown models fall back to
    ``forward_features``.
    """
    if target_layer is not None and target_layer != "auto":
        return target_layer
    canonical_name = resolve_supported_model_name(model_name)
    if canonical_name is not None:
        return SUPPORTED_MODEL_SPECS[canonical_name].target_layer
    return "forward_features"


def get_model_family(model_name: str | None) -> str | None:
    """Return supported family name if the model is in presets."""
    canonical_name = resolve_supported_model_name(model_name)
    if canonical_name is None:
        return None
    spec = SUPPORTED_MODEL_SPECS.get(canonical_name)
    return None if spec is None else spec.family
