"""
FujiShader.shader.ao_ray
========================

Multi-ray ambient occlusion (terrain AO) – Vectorized implementation
--------------------------------------------------------------------

* Shoots *n_rays* rays from each pixel to calculate maximum horizon angle
* Inner loops are vectorized with NumPy array operations → several times faster than pure Python
* `stride` option allows coarser sampling intervals for further speed improvement
  - stride=1 for exact pixel-by-pixel sampling (equivalent quality to traditional approach)
  - stride>1 sacrifices some accuracy for speed priority
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from FujiShader.core.progress import ProgressReporter, NullProgress

__all__ = ["ambient_occlusion"]


def _bilinear(dem: NDArray[np.float32], y: NDArray, x: NDArray) -> NDArray[np.float32]:
    """
    Bilinear interpolation of DEM at floating-point coordinates (y,x).
    Returns NaN for out-of-bounds coordinates or when any corner contains NaN.
    """
    h, w = dem.shape
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)

    # Mark out-of-bounds pixels
    invalid = (x0 < 0) | (x0 >= w - 1) | (y0 < 0) | (y0 >= h - 1)

    # Clip coordinates to safe bounds to prevent array access errors
    y0_safe = np.clip(y0, 0, h - 2)
    x0_safe = np.clip(x0, 0, w - 2)

    # Interpolation weights
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)

    # Sample the four corners
    z00 = dem[y0_safe,     x0_safe]
    z10 = dem[y0_safe,     x0_safe + 1]
    z01 = dem[y0_safe + 1, x0_safe]
    z11 = dem[y0_safe + 1, x0_safe + 1]

    # Mark pixels with NaN in any corner as invalid
    invalid |= np.isnan(z00) | np.isnan(z10) | np.isnan(z01) | np.isnan(z11)

    # Bilinear interpolation
    z = (
        z00 * (1 - tx) * (1 - ty)
        + z10 * tx * (1 - ty)
        + z01 * (1 - tx) * ty
        + z11 * tx * ty
    ).astype(np.float32, copy=False)

    # Set invalid pixels to NaN
    z[invalid] = np.nan
    return z


def ambient_occlusion(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    max_radius: float = 200.0,
    n_rays: int = 64,
    stride: int = 1,
    progress: ProgressReporter | None = None,
    _stream_state=None,           # For stream_tiles compatibility (unused)
) -> NDArray[np.float32]:
    """
    Calculate cosine-weighted ambient occlusion (0–1).

    Parameters
    ----------
    dem        : 2-D float32 array (NaN = NoData)
    cellsize   : Pixel resolution (single value or (dx,dy) tuple)
    max_radius : Maximum ray distance in ground units
    n_rays     : Number of rays (directions)
    stride     : Sampling interval in pixels – larger values (2,4) provide speed improvement
    progress   : ProgressReporter implementation (optional)

    Returns
    -------
    ao : 2-D float32 array, same shape as *dem*, 0 = fully occluded, 1 = no occlusion
    """
    # Input validation
    if dem.ndim != 2:
        raise ValueError("Input DEM must be 2-D array")
    if not (1 <= n_rays <= 360):
        raise ValueError("n_rays must be between 1 and 360")
    if stride < 1:
        raise ValueError("stride must be ≥1 pixel")
    if max_radius <= 0:
        raise ValueError("max_radius must be positive")

    # Ensure float32 dtype
    if dem.dtype != np.float32:
        dem = dem.astype(np.float32)

    progress = progress or NullProgress()

    h, w = dem.shape
    nan_mask = np.isnan(dem)

    # Pixel resolution
    if isinstance(cellsize, (int, float)):
        dx = dy = float(cellsize)
    else:
        dx, dy = map(float, cellsize)

    # Unit vectors for ray directions
    thetas = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    ux = np.cos(thetas)
    uy = np.sin(thetas)

    # Sampling step array (1, 1+stride, 1+2*stride, …)
    max_steps = int(np.ceil(max_radius / min(dx, dy)))
    step_indices = np.arange(1, max_steps + 1, stride, dtype=np.int32)

    # Pixel grid for broadcasting
    I = np.arange(h, dtype=np.float32)[:, None]  # shape (h,1)
    J = np.arange(w, dtype=np.float32)[None, :]  # shape (1,w)

    # AO output array (accumulates occlusion values)
    ao_map = np.zeros((h, w), dtype=np.float32)

    # プログレス設定：総ステップ数はn_rays
    progress.set_range(n_rays)

    for k in range(n_rays):
        vx, vy = ux[k], uy[k]

        # プログレス表示用テキスト
        progress_text = f"計算中: レイ {k + 1}/{n_rays} (方向角: {np.degrees(thetas[k]):.1f}°)"

        # Track maximum horizon angle for each pixel along this ray direction
        max_angle = np.full((h, w), -np.inf, dtype=np.float32)

        for step in step_indices:
            # Sample coordinates along the ray
            x_f = J + vx * step
            y_f = I + vy * step

            # Sample elevation at these coordinates
            z_sample = _bilinear(dem, y_f, x_f)
            valid = ~np.isnan(z_sample)

            if not np.any(valid):
                # All samples invalid at this step, subsequent steps will also be invalid
                break

            # Calculate ground distance for this step
            dist = np.sqrt((vx * step * dx)**2 + (vy * step * dy)**2)

            # Calculate elevation angle from current pixel to sampled point
            angle = np.full_like(z_sample, -np.inf, dtype=np.float32)
            angle[valid] = np.arctan((z_sample[valid] - dem[valid]) / dist)

            # Update maximum angle encountered so far
            np.maximum(max_angle, angle, out=max_angle, where=valid)

            # Early termination: if all pixels reach near-vertical angle
            if np.all((max_angle >= (np.pi/2 - 1e-6)) | nan_mask):
                break

        # Handle pixels where no valid samples were found
        max_angle[max_angle == -np.inf] = 0.0

        # Calculate occlusion: (π/2 - max_horizon_angle) / (π/2)
        # Higher horizon angle = more occlusion = lower AO value
        occl = (np.pi/2 - max_angle) / (np.pi/2)
        occl[nan_mask] = np.nan

        ao_map += occl
        
        # プログレス更新：1ステップ進捗、説明テキスト付き
        progress.advance(step=1, text=progress_text)

    progress.done()

    # Average across all rays
    ao_map /= float(n_rays)

    # Ensure valid range [0,1] and preserve NaN values
    ao_map = np.clip(ao_map, 0.0, 1.0, out=ao_map)
    ao_map[nan_mask] = np.nan

    return ao_map.astype(np.float32, copy=False)