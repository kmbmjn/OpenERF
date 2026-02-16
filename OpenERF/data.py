"""Data helpers for OpenERF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def list_image_paths(image_dir: str | Path) -> list[Path]:
    """Return sorted image paths from a flat or nested image directory."""
    root = Path(image_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No supported images found in: {root}")
    return image_paths


class FlatImageDataset(Dataset[tuple[Tensor, str]]):
    """Dataset for directories that only contain images (labels are not required)."""

    def __init__(
        self,
        image_dir: str | Path,
        transform: Callable[[Image.Image], Tensor],
        max_images: int | None = None,
    ) -> None:
        self.image_paths = list_image_paths(image_dir)
        if max_images is not None:
            if max_images <= 0:
                raise ValueError("max_images must be a positive integer")
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        return self.transform(image), str(image_path)


def build_timm_eval_transform(model: Module) -> tuple[Callable[[Image.Image], Tensor], dict[str, Any]]:
    """
    Build evaluation transform from timm model metadata.

    This preserves model-specific normalization, interpolation, and crop settings.
    """
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return transform, data_config

