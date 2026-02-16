# OpenERF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![timm](https://img.shields.io/badge/Built%20on-timm-orange.svg)](https://github.com/huggingface/pytorch-image-models)

> One-line **Effective Receptive Field (ERF)** extraction for pretrained `timm` vision models.

ERF visualizes how input pixels contribute to model responses — useful for interpretability and trustworthy AI workflows.

```python
import openerf
import timm

model_name = "resnet50.a1_in1k"
model = timm.create_model(model_name, pretrained=True)

openerf.save_erf(model, model_name=model_name)
# -> ./results/OpenERF_resnet50.a1_in1k.png
```

Legacy top-level import is still available: `import OpenERF`.

```python
from openerf import compute_erf, save_erf
```

---

## Visual Gallery

All images below are ERF maps generated from the ImageNet validation subset in `./imagenet_val_1000`.
The full set is available in [`./results/`](./results/).

<table>
  <tr>
    <th align="center">ResNet-34</th>
    <th align="center">ResNet-50</th>
  </tr>
  <tr>
    <td align="center"><img src="results/OpenERF_resnet34.a1_in1k.png" alt="ERF ResNet-34"></td>
    <td align="center"><img src="results/OpenERF_resnet50.a1_in1k.png" alt="ERF ResNet-50"></td>
  </tr>
  <tr>
    <td align="center"><code>resnet34.a1_in1k</code></td>
    <td align="center"><code>resnet50.a1_in1k</code></td>
  </tr>
  <tr>
    <th align="center">ResNeXt-101</th>
    <th align="center">DenseNet-201</th>
  </tr>
  <tr>
    <td align="center"><img src="results/OpenERF_resnext101_32x8d.tv_in1k.png" alt="ERF ResNeXt-101"></td>
    <td align="center"><img src="results/OpenERF_densenet201.tv_in1k.png" alt="ERF DenseNet-201"></td>
  </tr>
  <tr>
    <td align="center"><code>resnext101_32x8d.tv_in1k</code></td>
    <td align="center"><code>densenet201.tv_in1k</code></td>
  </tr>
  <tr>
    <th align="center">ViT-B/16</th>
    <th align="center">DeiT-B/16</th>
  </tr>
  <tr>
    <td align="center"><img src="results/OpenERF_vit_base_patch16_224.augreg_in1k.png" alt="ERF ViT-B16"></td>
    <td align="center"><img src="results/OpenERF_deit_base_patch16_224.fb_in1k.png" alt="ERF DeiT-B16"></td>
  </tr>
  <tr>
    <td align="center"><code>vit_base_patch16_224.augreg_in1k</code></td>
    <td align="center"><code>deit_base_patch16_224.fb_in1k</code></td>
  </tr>
  <tr>
    <th align="center">CaiT-S24</th>
    <th align="center">XCiT-Medium</th>
  </tr>
  <tr>
    <td align="center"><img src="results/OpenERF_cait_s24_224.fb_dist_in1k.png" alt="ERF CaiT-S24"></td>
    <td align="center"><img src="results/OpenERF_xcit_medium_24_p16_224.fb_in1k.png" alt="ERF XCiT-Medium"></td>
  </tr>
  <tr>
    <td align="center"><code>cait_s24_224.fb_dist_in1k</code></td>
    <td align="center"><code>xcit_medium_24_p16_224.fb_in1k</code></td>
  </tr>
  <tr>
    <th align="center">BEiT-B/16 (224)</th>
    <th align="center">SwinV2-Small</th>
  </tr>
  <tr>
    <td align="center"><img src="results/OpenERF_beit_base_patch16_224.in22k_ft_in22k_in1k.png" alt="ERF BEiT-B16-224"></td>
    <td align="center"><img src="results/OpenERF_swinv2_cr_small_224.sw_in1k.png" alt="ERF SwinV2-Small"></td>
  </tr>
  <tr>
    <td align="center"><code>beit_base_patch16_224.in22k_ft_in22k_in1k</code></td>
    <td align="center"><code>swinv2_cr_small_224.sw_in1k</code></td>
  </tr>
</table>

---

## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Options](#cli-options)
- [Output Structure](#output-structure)
- [Distribution Package](#distribution-package)
- [API Reference](#api-reference)
- [Supported Models](#supported-models)
- [Dataset](#dataset)
- [Citing](#citing)
- [License](#license)

---

## Features

- **One-line API** — `openerf.save_erf` for end-to-end ERF extraction
- **Automatic preprocessing** — `mean/std`, interpolation, `crop_pct`, input size via `timm`
- **Unified workflow** — CNN and Transformer families with the same interface
- **Publication-ready output** — colored ERF maps (`plasma` colormap by default)
- **Optional exports** — Gaussian fitting (`lmfit`), `.npy` arrays, `.json` metrics

## Repository Layout

```text
OpenERF/
  src/
    openerf/
      __init__.py
      api.py
      cli.py
      data.py
      erf.py
      feature_ops.py
      fit.py
      metrics.py
      model_zoo.py
      visualization.py
      py.typed
  tests/
    test_smoke.py
  examples/
    example.py
  README.md
  LICENSE
  pyproject.toml
  requirements.txt
```

## Installation

```bash
conda activate OpenERF
pip install -r requirements.txt
pip install -e .
# optional gaussian fit extras
pip install -e ".[gaussian]"
```

## Quick Start

### Python API — single model

```python
import openerf
import timm

model_name = "resnet50.a1_in1k"
model = timm.create_model(model_name, pretrained=True)

result = openerf.save_erf(
    model=model,
    model_name=model_name,
    image_dir="./imagenet_val_1000",
    max_images=1000,
    colormap="plasma",
    save_numpy=True,
    save_metrics=True,
)

print(result["save_path"])
print(result.get("npy_path"))
print(result.get("metrics_path"))
```

### CLI — batch run (preset models)

```bash
openerf --help
```

```bash
openerf --image-dir ./imagenet_val_1000 --max-images 1000
```

```bash
# show all preset models
openerf --list-models

# resume long runs
openerf --image-dir ./imagenet_val_1000 --max-images 1000 --skip-existing

# run selected families only
openerf --families vit deit cait xcit beit swin swinv2 \
    --image-dir ./imagenet_val_1000 --max-images 1000
```

```bash
# module form is also supported
python -m openerf.cli --list-models
```

## CLI Options

`openerf` (or `python -m openerf.cli`) supports the following options:

| Option | Description | Default |
| --- | --- | --- |
| `--families` | Model families to run (when `--model-names` is not set) | all preset families |
| `--model-names` | Explicit `timm` model names (must include pretrained tag) | `None` |
| `--image-dir` | Input image directory | `./imagenet_val_1000` |
| `--save-dir` | PNG output directory | `./results` |
| `--npy-dir` | NumPy output directory | `./results_npy` |
| `--metrics-dir` | Metrics output directory | `./results_metrics` |
| `--max-images` | Maximum images to process | `1000` |
| `--target-layer` | Override feature source layer | auto (family preset) |
| `--fit-gaussian` | Enable 2D Gaussian fit (`sigma_x`, `sigma_y`) | disabled |
| `--skip-existing` | Skip if output PNG already exists | disabled |
| `--no-save-numpy` | Disable `.npy` export | enabled |
| `--no-save-metrics` | Disable `.json` export | enabled |
| `--stop-on-error` | Stop immediately on model failure | disabled |

## Output Structure

```
./results/                OpenERF_<model_name>.png          # ERF heatmap
./results_npy/            OpenERF_<model_name>.npy          # ERF array
./results_metrics/        OpenERF_<model_name>_metrics.json # ERF metrics
./results_debug/          (smoke/check/intermediate artifacts)
```

## Distribution Package

Source distributions are configured to include package source and core metadata while excluding large experiment artifacts.

- Included: `src/`, `README.md`, `LICENSE`, `pyproject.toml`
- Excluded: `results*/`, `reference/`, `examples/`, `tests/`, `OpenERF.egg-info/`

## API Reference

### `openerf.compute_erf(...)`

Computes ERF in memory. Returns `ERFResult`:

| Field | Description |
| --- | --- |
| `erf_map` | 2D ERF array |
| `num_images` | Number of images processed |
| `data_config` | `timm` data config used |
| `resolved_target_layer` | Actual target layer name |
| `gaussian_fit` | *(optional)* Gaussian fit parameters |

### `openerf.save_erf(...)`

Computes ERF, saves PNG, and optionally exports `.npy` / `.json`.

**Defaults:** `save_numpy=False`, `save_metrics=False`, `colormap="plasma"`

| Return key | Description |
| --- | --- |
| `model_name` | Model identifier |
| `source_model_name` | Original `timm` model name |
| `save_path` | Path to saved PNG |
| `num_images` | Number of images processed |
| `target_layer` | Target layer name |
| `family` | Model family |
| `data_config` | `timm` data config used |
| `gaussian_fit` | *(optional)* Gaussian fit result |
| `npy_path` | *(optional)* Path to `.npy` file |
| `metrics_path` | *(optional)* Path to `.json` file |

## Supported Models

11 families &times; 3 variants = **33 preset models**

| Family | Models |
| --- | --- |
| `resnet` | `resnet18.a1_in1k`, `resnet34.a1_in1k`, `resnet50.a1_in1k` |
| `resnext` | `resnext50_32x4d.a1_in1k`, `resnext50_32x4d.ra_in1k`, `resnext101_32x8d.tv_in1k` |
| `densenet` | `densenet121.ra_in1k`, `densenet169.tv_in1k`, `densenet201.tv_in1k` |
| `efficientnet` | `efficientnet_b0.ra_in1k`, `efficientnet_b2.ra_in1k`, `tf_efficientnet_b4.ns_jft_in1k` |
| `vit` | `vit_small_patch16_224.augreg_in1k`, `vit_base_patch16_224.augreg_in1k`, `vit_base_patch32_224.augreg_in1k` |
| `deit` | `deit_tiny_patch16_224.fb_in1k`, `deit_small_patch16_224.fb_in1k`, `deit_base_patch16_224.fb_in1k` |
| `cait` | `cait_xxs24_224.fb_dist_in1k`, `cait_xxs36_224.fb_dist_in1k`, `cait_s24_224.fb_dist_in1k` |
| `xcit` | `xcit_tiny_12_p16_224.fb_in1k`, `xcit_small_12_p16_224.fb_in1k`, `xcit_medium_24_p16_224.fb_in1k` |
| `beit` | `beit_base_patch16_224.in22k_ft_in22k`, `beit_base_patch16_224.in22k_ft_in22k_in1k`, `beit_base_patch16_384.in22k_ft_in22k_in1k` |
| `swin` | `swin_tiny_patch4_window7_224.ms_in1k`, `swin_small_patch4_window7_224.ms_in1k`, `swin_base_patch4_window7_224.ms_in1k` |
| `swinv2` | `swinv2_cr_tiny_ns_224.sw_in1k`, `swinv2_cr_small_224.sw_in1k`, `swinv2_cr_small_ns_224.sw_in1k` |

## Dataset

OpenERF uses the ImageNet validation set (or a subset).

- Default subset index: `./imagenet_val_1000.txt`
- Image folder: `./imagenet_val_1000/`
- Default scope: `--image-dir ./imagenet_val_1000 --max-images 1000`

## Citing

If OpenERF is useful in your research, please cite:

```bibtex
@inproceedings{KimCJK23,
  author       = {Bum Jun Kim and
                  Hyeyeon Choi and
                  Hyeonah Jang and
                  Sang Woo Kim},
  title        = {Understanding Gaussian Attention Bias of Vision Transformers Using
                  Effective Receptive Fields},
  booktitle    = {{BMVC}},
  pages        = {214},
  publisher    = {{BMVA} Press},
  year         = {2023}
}
```

```bibtex
@article{KimCJLJK23,
  author       = {Bum Jun Kim and
                  Hyeyeon Choi and
                  Hyeonah Jang and
                  Dong Gu Lee and
                  Wonseok Jeong and
                  Sang Woo Kim},
  title        = {Dead pixel test using effective receptive field},
  journal      = {Pattern Recognit. Lett.},
  volume       = {167},
  pages        = {149--156},
  year         = {2023}
}
```

## License

This project is released under the [MIT License](./LICENSE).
