"""Optional 2D Gaussian fitting for ERF maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from lmfit import Model
except ImportError:  # pragma: no cover
    Model = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GaussianFitResult:
    """Parameters from 2D Gaussian fit on a normalized ERF map."""

    amp: float
    xc: float
    yc: float
    sigma_x: float
    sigma_y: float
    r_squared: float


def fit_2d_gaussian(erf_map: np.ndarray) -> GaussianFitResult:
    """
    Fit a 2D Gaussian to the ERF map.

    The model and initialization follow the protocol used in the referenced papers.
    """
    if Model is None:
        raise ImportError(
            "lmfit is not installed. Install optional dependencies with "
            "`pip install lmfit scipy` to enable Gaussian fitting."
        )

    if erf_map.ndim != 2:
        raise ValueError(f"erf_map must be 2D, got shape={erf_map.shape}")

    noise = np.asarray(erf_map, dtype=np.float64)
    noise_sum = float(noise.sum())
    if noise_sum <= 0:
        raise ValueError("Cannot fit Gaussian because ERF map sum is non-positive.")
    noise = noise / noise_sum

    height, width = noise.shape
    x = np.arange(width)
    y = np.arange(height)
    xy_mesh = np.meshgrid(x, y)

    def gaussian_2d(
        mesh: tuple[np.ndarray, np.ndarray],
        amp: float,
        xc: float,
        yc: float,
        sigma_x: float,
        sigma_y: float,
    ) -> np.ndarray:
        x_mesh, y_mesh = mesh
        gauss = amp * np.exp(
            -(
                (x_mesh - xc) ** 2 / (2 * sigma_x**2)
                + (y_mesh - yc) ** 2 / (2 * sigma_y**2)
            )
        ) / (2 * np.pi * sigma_x * sigma_y)
        return np.ravel(gauss)

    amp = 1.0
    xc, yc = np.median(x), np.median(y)
    sigma_x, sigma_y = max(width / 10, 1e-3), max(height / 6, 1e-3)
    guess_vals = [amp * 2, xc * 0.8, yc * 0.8, sigma_x / 1.5, sigma_y / 1.5]

    lmfit_model = Model(gaussian_2d)
    params = lmfit_model.make_params(
        amp=guess_vals[0],
        xc=guess_vals[1],
        yc=guess_vals[2],
        sigma_x=guess_vals[3],
        sigma_y=guess_vals[4],
    )
    params["amp"].set(min=0.0)
    params["xc"].set(min=0.0, max=float(width - 1))
    params["yc"].set(min=0.0, max=float(height - 1))
    params["sigma_x"].set(min=1e-6)
    params["sigma_y"].set(min=1e-6)

    lmfit_result = lmfit_model.fit(
        np.ravel(noise),
        mesh=xy_mesh,
        params=params,
    )

    noise_var = float(np.var(noise))
    r_squared = (
        float("nan")
        if noise_var == 0
        else 1.0 - float(lmfit_result.residual.var()) / noise_var
    )

    values = lmfit_result.best_values
    return GaussianFitResult(
        amp=float(values["amp"]),
        xc=float(values["xc"]),
        yc=float(values["yc"]),
        sigma_x=float(values["sigma_x"]),
        sigma_y=float(values["sigma_y"]),
        r_squared=r_squared,
    )
