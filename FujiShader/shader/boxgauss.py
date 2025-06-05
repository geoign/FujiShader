"""
FujiShader.shader.topo_boxgauss
==================================

Gaussian approximation USM implementation using stacked-box filters.
--------------------------------------------------------------------
* Approximates Gaussian with σ≈r/2 using 3-pass 1-D box filters for fast O(N) computation
  with low memory usage. Returns `DEM - Blur` for unsharp masking effect.
* Pure NumPy implementation without SciPy dependency.
* NaN handling via mask integration with division maintains quality of original implementation.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.progress import ProgressReporter

__all__ = [
    "topo_boxgauss",
    "multi_scale_boxgauss",
]

# ----------------------------------------------------------------------------
# Box filter utilities
# ----------------------------------------------------------------------------

def _box_filter_sum(
    arr: NDArray[np.float32], 
    r: int, 
    axis: int
) -> NDArray[np.float32]:
    """Apply box filter sum of width (2r+1) along specified axis (pure NumPy)."""
    if r < 1:
        return arr
    
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    a = np.pad(arr, pad, mode="constant", constant_values=0)
    cs = np.cumsum(a, axis=axis, dtype=np.float64)
    
    slice_hi = [slice(None)] * arr.ndim
    slice_lo = [slice(None)] * arr.ndim
    slice_hi[axis] = slice(2 * r + 1, None)
    slice_lo[axis] = slice(None, -2 * r - 1)
    
    return (cs[tuple(slice_hi)] - cs[tuple(slice_lo)]).astype(arr.dtype, copy=False)


def _box_blur_nan(
    arr: NDArray[np.float32], 
    r: int, 
    passes: int = 3,
    progress: Optional["ProgressReporter"] = None
) -> NDArray[np.float32]:
    """NaN-aware multi-pass box blur for Gaussian approximation."""
    if r < 1:
        return arr

    nan_mask = np.isnan(arr)
    val = np.where(nan_mask, 0.0, arr).astype(np.float32, copy=False)
    wgt = (~nan_mask).astype(np.float32)

    for pass_idx in range(passes):
        if progress:
            progress.advance(1, f"Box blur pass {pass_idx + 1}/{passes} - horizontal")
        
        # Horizontal pass
        val = _box_filter_sum(val, r, axis=1)
        wgt = _box_filter_sum(wgt, r, axis=1)
        
        if progress:
            progress.advance(1, f"Box blur pass {pass_idx + 1}/{passes} - vertical")
        
        # Vertical pass
        val = _box_filter_sum(val, r, axis=0)
        wgt = _box_filter_sum(wgt, r, axis=0)

    out = np.empty_like(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(val, wgt, out=out, where=wgt > 0)
    out[wgt == 0] = np.nan
    return out


def _box_blur_simple(
    arr: NDArray[np.float32], 
    r: int, 
    passes: int = 3,
    progress: Optional["ProgressReporter"] = None
) -> NDArray[np.float32]:
    """Simple multi-pass box blur for arrays without NaN values."""
    if r < 1:
        return arr

    result = arr.copy()
    filter_size = 2 * r + 1

    for pass_idx in range(passes):
        if progress:
            progress.advance(1, f"Box blur pass {pass_idx + 1}/{passes} - horizontal")
        
        # Horizontal pass
        result = _box_filter_sum(result, r, axis=1) / filter_size
        
        if progress:
            progress.advance(1, f"Box blur pass {pass_idx + 1}/{passes} - vertical")
        
        # Vertical pass
        result = _box_filter_sum(result, r, axis=0) / filter_size

    return result

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def topo_boxgauss(
    dem: NDArray[np.float32],
    radius: int = 8,
    *,
    passes: int = 3,
    treat_nan: Union[float, None] = None,
    dtype: type = np.float32,
    progress: Optional["ProgressReporter"] = None,
) -> NDArray[np.float32]:
    """
    Topographic unsharp masking using box filter Gaussian approximation.
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        Input 2D digital elevation model
    radius : int, default=8
        Filter radius in pixels (σ ≈ radius/2 for Gaussian approximation)
    passes : int, default=3
        Number of box filter passes for Gaussian approximation
    treat_nan : float or None, default=None
        Value to replace NaN with. If None, NaN values are preserved in output
    dtype : type, default=np.float32
        Output data type
    progress : ProgressReporter or None, default=None
        Progress reporter for tracking computation progress
        
    Returns
    -------
    NDArray[np.float32]
        Unsharp mask result (original - blurred)
        
    Raises
    ------
    ValueError
        If input DEM is not 2D or radius is less than 1
    """
    if dem.ndim != 2:
        raise ValueError("Input DEM must be 2D array")
    if radius < 1:
        raise ValueError("Radius must be ≥1 pixel")
    if passes < 1:
        raise ValueError("Number of passes must be ≥1")

    # プログレス初期化：前処理(1) + ブラー処理(passes*2) + 後処理(1)
    total_steps = 1 + passes * 2 + 1
    if progress:
        progress.set_range(total_steps)
        progress.advance(1, "Preparing input data")

    arr_orig = dem.astype(dtype, copy=False)

    # Box radius for σ ≈ radius/2 approximation
    r = radius

    if treat_nan is not None:
        arr_base = np.where(np.isnan(arr_orig), treat_nan, arr_orig)
        ignore_nan = False
    else:
        arr_base = arr_orig
        ignore_nan = True

    # Use pure NumPy implementation
    if ignore_nan:
        blur = _box_blur_nan(arr_base, r, passes=passes, progress=progress)
    else:
        blur = _box_blur_simple(arr_base, r, passes=passes, progress=progress)

    if progress:
        progress.advance(1, "Computing unsharp mask")

    usm = arr_orig - blur

    if ignore_nan:
        usm[np.isnan(arr_orig)] = np.nan

    if progress:
        progress.done()

    return usm.astype(dtype, copy=False)


def multi_scale_boxgauss(
    dem: NDArray[np.float32],
    radii: Union[Sequence[int], int] = (4, 16, 64, 256),
    weights: Union[Sequence[float], None] = None,
    *,
    passes: int = 3,
    normalize: bool = True,
    treat_nan: Union[float, None] = None,
    dtype: type = np.float32,
    progress: Optional["ProgressReporter"] = None,
) -> Tuple[NDArray[np.float32], List[NDArray[np.float32]]]:
    """
    Multi-scale topographic unsharp masking using stacked box filter layers.
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        Input 2D digital elevation model
    radii : Sequence[int] or int, default=(4, 16, 64, 256)
        Filter radii for each scale
    weights : Sequence[float] or None, default=None
        Weights for combining scales. If None, uses radii as weights
    passes : int, default=3
        Number of box filter passes for Gaussian approximation
    normalize : bool, default=True
        Whether to normalize output to [-1, 1] range using 99th percentile
    treat_nan : float or None, default=None
        Value to replace NaN with. If None, NaN values are preserved
    dtype : type, default=np.float32
        Output data type
    progress : ProgressReporter or None, default=None
        Progress reporter for tracking computation progress
        
    Returns
    -------
    Tuple[NDArray[np.float32], List[NDArray[np.float32]]]
        Combined multi-scale result and list of individual scale layers
        
    Raises
    ------
    ValueError
        If any radius is less than 1 or weights/radii length mismatch
    """
    if isinstance(radii, int):
        radii = (radii,)
    radii = list(radii)
    if any(r < 1 for r in radii):
        raise ValueError("All radii must be ≥1 pixel")

    if weights is None:
        weights = list(radii)
    if len(weights) != len(radii):
        raise ValueError("Weights and radii must have same length")

    # プログレス初期化：各スケール処理 + 合成処理(1) + 正規化処理(1)
    total_steps = len(radii) + 1 + (1 if normalize else 0)
    if progress:
        progress.set_range(total_steps)

    layers: List[NDArray[np.float32]] = []
    
    # 各スケールの処理時に個別のプログレスは使用せず、完了時に1ステップ進める
    for i, r in enumerate(radii):
        if progress:
            progress.advance(0, f"Processing scale {i + 1}/{len(radii)} (radius={r})")
        
        layer = topo_boxgauss(
            dem,
            radius=r,
            passes=passes,
            treat_nan=treat_nan,
            dtype=dtype,
            progress=None,  # 個別プログレスは無効化
        )
        layers.append(layer)
        
        if progress:
            progress.advance(1, f"Completed scale {i + 1}/{len(radii)}")

    if progress:
        progress.advance(1, "Combining multi-scale layers")

    comp = np.zeros_like(layers[0], dtype=np.float64)
    for w, lay in zip(weights, layers):
        comp += w * lay.astype(np.float64)
    comp /= max(weights)

    if normalize:
        if progress:
            progress.advance(0, "Normalizing output")
        abs99 = float(np.nanpercentile(np.abs(comp), 99.0)) or 1e-9
        comp = np.clip(comp / abs99, -1.0, 1.0)
        if progress:
            progress.advance(1, "Normalization completed")

    if progress:
        progress.done()

    return comp.astype(dtype, copy=False), layers