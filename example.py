"""OpenERF example: run ERF extraction on preset timm model families."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Iterable

import timm
import torch

import OpenERF
from OpenERF.data import list_image_paths
from OpenERF.model_zoo import ORDERED_FAMILIES, SUPPORTED_MODEL_GROUPS


DEFAULT_FAMILIES = ORDERED_FAMILIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=list(DEFAULT_FAMILIES),
        help="families to run when --model-names is not provided",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=None,
        help="explicit timm model names to process",
    )
    parser.add_argument("--image-dir", type=str, default="./imagenet_val_1000")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--npy-dir", type=str, default="./results_npy")
    parser.add_argument("--metrics-dir", type=str, default="./results_metrics")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument(
        "--target-layer",
        type=str,
        default=None,
        help="optional override layer name (defaults to family preset)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fit-gaussian", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="optional suffix appended to saved model_name (e.g. _1000img)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip models whose PNG output already exists in save_dir",
    )
    parser.add_argument(
        "--save-numpy",
        action="store_true",
        default=True,
        help="save ERF map as .npy (default: enabled)",
    )
    parser.add_argument(
        "--no-save-numpy",
        dest="save_numpy",
        action="store_false",
        help="disable .npy export",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="save ERF metrics as JSON (default: enabled)",
    )
    parser.add_argument(
        "--no-save-metrics",
        dest="save_metrics",
        action="store_false",
        help="disable metrics JSON export",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop immediately when a model fails",
    )
    return parser.parse_args()


def _validate_explicit_pretrained_names(model_names: Iterable[str]) -> None:
    invalid = [name for name in model_names if "." not in name]
    if invalid:
        raise ValueError(
            "Model names must explicitly include the pretrained checkpoint tag "
            "(e.g. 'resnet50.a1_in1k'). Invalid: "
            + ", ".join(invalid)
        )


def _resolve_model_names(
    families: Iterable[str],
    model_names: Iterable[str] | None,
) -> list[str]:
    if model_names is not None:
        selected = list(model_names)
    else:
        selected = []
        for family in families:
            if family not in SUPPORTED_MODEL_GROUPS:
                raise ValueError(
                    f"Unknown family '{family}'. Available: "
                    + ", ".join(sorted(SUPPORTED_MODEL_GROUPS))
                )
            selected.extend(SUPPORTED_MODEL_GROUPS[family])
    _validate_explicit_pretrained_names(selected)
    return selected


def _print_supported_models() -> None:
    for family in ORDERED_FAMILIES:
        print(f"{family}:")
        for model_name in SUPPORTED_MODEL_GROUPS[family]:
            print(f"  - {model_name}")


def _clear_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_models(
    families: Iterable[str],
    model_names: Iterable[str] | None,
    image_dir: str,
    save_dir: str,
    npy_dir: str,
    metrics_dir: str,
    max_images: int | None,
    target_layer: str | None,
    num_workers: int,
    device: str | None,
    fit_gaussian: bool,
    show_progress: bool,
    output_suffix: str,
    skip_existing: bool,
    save_numpy: bool,
    save_metrics: bool,
    stop_on_error: bool,
) -> None:
    selected_model_names = _resolve_model_names(families=families, model_names=model_names)

    image_count = len(list_image_paths(image_dir))
    if max_images is not None:
        image_count = min(image_count, max_images)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if save_numpy:
        Path(npy_dir).mkdir(parents=True, exist_ok=True)
    if save_metrics:
        Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    print("OpenERF batch run started")
    print(f"  image_dir    : {image_dir}")
    print(f"  max_images   : {max_images} (effective: {image_count})")
    print(f"  target_layer : {target_layer or 'auto (family preset)'}")
    print(f"  save_numpy   : {save_numpy}")
    print(f"  save_metrics : {save_metrics}")
    print(f"  models       : {', '.join(selected_model_names)}")

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for index, model_name in enumerate(selected_model_names, start=1):
        print(f"\n[{index}/{len(selected_model_names)}] Processing {model_name}")
        output_model_name = f"{model_name}{output_suffix}"
        output_png_path = Path(save_dir) / f"OpenERF_{output_model_name}.png"
        if skip_existing and output_png_path.exists():
            print(f"  skipped     : existing output found at {output_png_path}")
            skipped.append(model_name)
            continue

        model = None
        try:
            model = timm.create_model(model_name, pretrained=True)
            result = OpenERF.save_ERF(
                model=model,
                model_name=output_model_name,
                source_model_name=model_name,
                image_dir=image_dir,
                save_dir=save_dir,
                npy_dir=npy_dir,
                metrics_dir=metrics_dir,
                max_images=max_images,
                target_layer=target_layer,
                device=device,
                num_workers=num_workers,
                show_progress=show_progress,
                fit_gaussian=fit_gaussian,
                colormap="plasma",
                save_numpy=save_numpy,
                save_metrics=save_metrics,
            )
        except Exception as error:
            failed.append((model_name, str(error)))
            print(f"  failed      : {error}")
            if stop_on_error:
                raise
            continue
        finally:
            if model is not None:
                del model
            gc.collect()
            _clear_torch_cache()

        cfg = result["data_config"]
        print(f"  family      : {result['family']}")
        print(f"  target_layer: {result['target_layer']}")
        print(f"  input_size  : {cfg.get('input_size')}")
        print(f"  interpolation: {cfg.get('interpolation')}, crop_pct: {cfg.get('crop_pct')}")
        print(f"  mean/std    : {cfg.get('mean')} / {cfg.get('std')}")
        print(f"  num_images  : {result['num_images']}")
        print(f"  png         : {result['save_path']}")
        if save_numpy:
            print(f"  npy         : {result['npy_path']}")
        if save_metrics:
            print(f"  metrics     : {result['metrics_path']}")
        if "gaussian_fit" in result:
            sigma_x = result["gaussian_fit"]["sigma_x"]
            sigma_y = result["gaussian_fit"]["sigma_y"]
            print(f"  sigma_x/y   : {sigma_x:.3f} / {sigma_y:.3f}")
        succeeded.append(model_name)

    print("\nOpenERF batch run finished")
    print(f"  succeeded   : {len(succeeded)}")
    print(f"  skipped     : {len(skipped)}")
    print(f"  failed      : {len(failed)}")
    if failed:
        print("  failed_models:")
        for model_name, reason in failed:
            print(f"    - {model_name}: {reason}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        _print_supported_models()
        return

    run_models(
        families=args.families,
        model_names=args.model_names,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        npy_dir=args.npy_dir,
        metrics_dir=args.metrics_dir,
        max_images=args.max_images,
        target_layer=args.target_layer,
        num_workers=args.num_workers,
        device=args.device,
        fit_gaussian=args.fit_gaussian,
        show_progress=not args.no_progress,
        output_suffix=args.output_suffix,
        skip_existing=args.skip_existing,
        save_numpy=args.save_numpy,
        save_metrics=args.save_metrics,
        stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    main()
