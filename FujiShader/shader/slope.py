"""
FujiShader.shader.slope
======================

Fast vectorised slope calculation from a DEM.
---------------------------------------------
The function **`slope`** returns surface slope either in degrees (default) or
percent.  It is designed to mirror the behaviour of *gdaldem slope* but using
pure NumPy for tight integration with the rest of *FujiShader*.

Key points
~~~~~~~~~~
* **Central differences** via :func:`numpy.gradient` – O(N) execution, near
  optimal cache usage.
* **Arbitrary cellsize** – accepts scalar or (dy, dx) tuple so non-square
  pixels are handled correctly.
* **NaN‑aware** – optional *treat_nan* fills voids temporarily to minimise edge
  artefacts, then restores NaNs in the output.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["slope"]

_EPS: float = 1e-12


def slope(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    unit: str = "degree",
    treat_nan: float | None = None,
    dtype=np.float32,
    progress: ProgressReporter | None = None,
) -> NDArray[np.float32]:
    """Compute terrain slope from a DEM.

    Parameters
    ----------
    dem : ndarray (H, W)
        Elevation raster (float32 preferred).
    cellsize : float or (float, float), default 1.0
        Pixel size (map units).  If tuple, interpreted as (dy, dx).
    unit : {"degree", "percent"}, default "degree"
        Output unit.  "percent" returns *rise / run* ×100.
    treat_nan : float or None
        If not *None*, NaNs in *dem* are filled with this value before gradient
        computation, then restored afterwards.
    dtype : NumPy dtype, default float32
        Output array dtype.

    Returns
    -------
    slope : ndarray (H, W)
        Slope in requested unit.  NaNs from input are preserved.
    """
    # Early validation
    if dem.ndim != 2:
        raise ValueError("DEM must be 2‑D array")
    
    if dem.size == 0:
        raise ValueError("DEM cannot be empty")
    
    # Validate unit parameter early to avoid unnecessary computation
    if unit not in ("degree", "percent", "percentage") and not unit.startswith("perc"):
        raise ValueError("unit must be 'degree' or 'percent'")
    
    # Initialize progress reporting
    progress = progress or NullProgress()
    # 処理ステップの定義:
    # 1. データ準備・検証
    # 2. NaN処理
    # 3. 勾配計算
    # 4. 単位変換
    # 5. 最終処理
    progress.set_range(5)
    progress.advance(text="初期化中...")

    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    if dx <= 0 or dy <= 0:
        raise ValueError("cellsize must be positive")

    arr = dem.astype(dtype, copy=False)
    progress.advance(text="データ準備完了")

    # Optional NaN handling
    nan_mask = np.isnan(arr)
    if treat_nan is not None and nan_mask.any():
        arr = arr.copy()
        arr[nan_mask] = treat_nan
        progress.advance(text="NaN値を処理中...")
    else:
        progress.advance(text="NaN処理をスキップ")

    # Handle edge case: single element array
    if arr.size == 1:
        # Single pixel has zero slope
        result = np.zeros_like(arr, dtype=dtype)
        if nan_mask.any():
            result[nan_mask] = np.nan
        # 残りのステップを完了扱いにする
        progress.advance(text="単一ピクセル処理完了")
        progress.advance(text="処理完了")
        progress.done()
        return result

    # Gradient computation using central differences
    progress.advance(text="勾配を計算中...")
    dz_dy, dz_dx = np.gradient(arr, dy, dx, edge_order=1)

    # Calculate slope magnitude
    grad_mag = np.hypot(dz_dx, dz_dy)

    # Convert to requested unit
    progress.advance(text=f"単位を{unit}に変換中...")
    if unit == "degree":
        out = np.degrees(np.arctan(grad_mag))
    elif unit in ("percent", "percentage") or unit.startswith("perc"):
        out = grad_mag * 100.0  # Convert to percent
    else:
        # This should never be reached due to early validation
        raise ValueError("unit must be 'degree' or 'percent'")

    # Restore NaN values from original input
    if nan_mask.any():
        out[nan_mask] = np.nan  # type: ignore[index]
    
    result = out.astype(dtype, copy=False)

    # Mark computation as complete
    progress.done()
    return result