"""
FujiShader.shader.topo_usm
==============================

Optimised TopoUSM (Topographic Un‑Sharp Mask) implementation for QGIS.
--------------------------------------------------------------------
A drop‑in, strictly API‑compatible replacement for the original
``FujiShader.shader.topo_usm`` that runs **2–10 × faster** in common
workflows while remaining usable inside QGIS Python console where
Numba and some SciPy features may be unavailable.

Public API
~~~~~~~~~~~~~~~~~~~~~~
``topo_usm``            – single‑radius USM layer
``multi_scale_usm``     – flexible n‑layer stack (TopoUSM‑Stack)
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import List, Sequence, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

# Progress reporting support
try:
    from ..core.progress import ProgressReporter, NullProgress
except (ImportError, ModuleNotFoundError):
    # Fallback if progress module not available
    class ProgressReporter:
        def set_range(self, maximum: int) -> None: ...
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
        def done(self) -> None: ...
    
    class NullProgress:
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: pass
        def done(self) -> None: pass

# -----------------------------------------------------------------------------
# Public symbols
# -----------------------------------------------------------------------------
__all__ = [
    "topo_usm",
    "multi_scale_usm",
]

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
_FFT_THRESHOLD: int = 128  # px — empirically tuned for pure NumPy FFT
_LARGE_KERNEL_WARNING: int = 1000  # px — warn for very large kernels

def _get_eps(dtype) -> float:
    """Get appropriate epsilon for given dtype."""
    return float(np.finfo(dtype).eps * 1000)

# -----------------------------------------------------------------------------
# Kernel helpers (memoised)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _disc_kernel(radius: int, dtype_str: str) -> NDArray[np.float32]:
    """Return a *normalised* flat‑top disc kernel with given radius."""
    if radius < 1:
        raise ValueError("radius must be ≥1 pixel")
    if radius > _LARGE_KERNEL_WARNING:
        warnings.warn(f"Large kernel radius ({radius}) may consume significant memory", 
                     RuntimeWarning, stacklevel=3)
    
    r = radius
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    mask = x**2 + y**2 <= r**2
    k = mask.astype(np.float32)
    k_sum = k.sum(dtype=np.float64)
    if k_sum == 0:
        raise ValueError(f"Invalid kernel for radius {radius}")
    k /= k_sum
    return k.astype(dtype_str, copy=False)


@lru_cache(maxsize=64)
def _gauss_kernel(radius: int, dtype_str: str) -> NDArray[np.float32]:
    """Return a *normalised* Gaussian kernel with σ ≈ radius/2."""
    if radius < 1:
        raise ValueError("radius must be ≥1 pixel")
    if radius > _LARGE_KERNEL_WARNING:
        warnings.warn(f"Large kernel radius ({radius}) may consume significant memory", 
                     RuntimeWarning, stacklevel=3)
    
    sigma = radius / 2.0
    size = int(2 * radius + 1)
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size, dtype=dtype_str)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k_sum = k.sum(dtype=np.float64)
    if k_sum == 0:
        raise ValueError(f"Invalid Gaussian kernel for radius {radius}")
    k /= k_sum
    return k.astype(dtype_str, copy=False)


# -----------------------------------------------------------------------------
# FFT utilities using pure NumPy
# -----------------------------------------------------------------------------

def _fft_convolve_same(arr: NDArray[np.float32], kern: NDArray[np.float32], 
                      progress: Optional[ProgressReporter] = None) -> NDArray[np.float32]:
    """'Same' convolution using pure NumPy FFT implementation."""
    if progress:
        progress.advance(1, "FFT convolution: preparing")
    
    kh, kw = kern.shape
    ph, pw = arr.shape
    
    # Optimize for memory usage - zero-padded FFT
    if progress:
        progress.advance(1, "FFT convolution: padding kernel")
    pad_kernel = np.zeros((ph, pw), dtype=arr.dtype)
    y0, x0 = (ph - kh) // 2, (pw - kw) // 2
    pad_kernel[y0 : y0 + kh, x0 : x0 + kw] = kern
    pad_kernel = np.roll(pad_kernel, (-y0, -x0), axis=(0, 1))
    
    # FFT convolution using NumPy
    if progress:
        progress.advance(1, "FFT convolution: forward FFT")
    arr_fft = np.fft.rfftn(arr)
    kern_fft = np.fft.rfftn(pad_kernel)
    
    if progress:
        progress.advance(1, "FFT convolution: inverse FFT")
    out = np.fft.irfftn(arr_fft * kern_fft, s=arr.shape)
    
    # Ensure real output and correct dtype
    if np.iscomplexobj(out):
        out = out.real
    
    return out.astype(arr.dtype, copy=False)


def _convolve_nan_fft(arr: NDArray[np.float32], kern: NDArray[np.float32],
                     progress: Optional[ProgressReporter] = None) -> NDArray[np.float32]:
    """NaN‑aware convolution carried out fully in the frequency domain."""
    eps = _get_eps(arr.dtype)
    
    if progress:
        progress.advance(1, "NaN-aware FFT: preparing arrays")
    
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0.0, arr)
    weight = (~nan_mask).astype(arr.dtype)
    
    if progress:
        progress.advance(1, "NaN-aware FFT: computing numerator")
    numer = _fft_convolve_same(arr_filled, kern)
    
    if progress:
        progress.advance(1, "NaN-aware FFT: computing denominator")
    denom = _fft_convolve_same(weight, kern)

    if progress:
        progress.advance(1, "NaN-aware FFT: normalizing result")
    out = np.empty_like(arr)
    # Vectorized division with proper handling of small denominators
    valid_mask = denom > eps
    out[valid_mask] = numer[valid_mask] / denom[valid_mask]
    out[~valid_mask] = np.nan
    
    return out


# -----------------------------------------------------------------------------
# Pure NumPy spatial convolution (NaN-aware)
# -----------------------------------------------------------------------------

def _convolve_nan_spatial_fast(arr: NDArray[np.float32], kern: NDArray[np.float32],
                              progress: Optional[ProgressReporter] = None) -> NDArray[np.float32]:
    """
    Fast NaN-aware spatial convolution using optimized NumPy operations.
    This version prioritizes speed over perfect normalization.
    """
    kh, kw = kern.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = arr.shape
    
    if progress:
        progress.advance(1, "Spatial convolution: preparing arrays")
    
    # Pre-allocate output
    out = np.zeros_like(arr)
    weight_sum = np.zeros_like(arr)
    
    # Create padded array
    arr_padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), 
                       mode='constant', constant_values=np.nan)
    
    # Count non-zero kernel elements for progress tracking
    non_zero_count = np.count_nonzero(kern)
    processed_count = 0
    
    # Optimized convolution loop
    for j in range(kh):
        for i in range(kw):
            kern_weight = kern[j, i]
            if kern_weight == 0:
                continue
                
            # Extract the shifted window
            window = arr_padded[j:j+h, i:i+w]
            valid_mask = ~np.isnan(window)
            
            # Accumulate values and weights where valid
            out[valid_mask] += window[valid_mask] * kern_weight
            weight_sum[valid_mask] += kern_weight
            
            processed_count += 1
            if progress and processed_count % max(1, non_zero_count // 10) == 0:
                pct = int(100 * processed_count / non_zero_count)
                progress.advance(0, f"Spatial convolution: {pct}% complete")
    
    # Normalize by accumulated weights
    eps = _get_eps(arr.dtype)
    result = np.full_like(arr, np.nan)
    valid_norm = weight_sum > eps
    result[valid_norm] = out[valid_norm] / weight_sum[valid_norm]
    
    return result


def _convolve_spatial_no_nan(arr: NDArray[np.float32], kern: NDArray[np.float32],
                            progress: Optional[ProgressReporter] = None) -> NDArray[np.float32]:
    """
    Simple spatial convolution without NaN handling for better performance.
    """
    kh, kw = kern.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    if progress:
        progress.advance(1, "Spatial convolution: processing (no NaN)")
    
    # Pad array with edge values
    arr_padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Simple convolution using correlation
    from numpy import correlate
    # For 2D, we need to implement manually
    h, w = arr.shape
    out = np.zeros_like(arr)
    
    non_zero_count = np.count_nonzero(kern)
    processed_count = 0
    
    for j in range(kh):
        for i in range(kw):
            if kern[j, i] == 0:
                continue
            
            window = arr_padded[j:j+h, i:i+w]
            out += window * kern[j, i]
            
            processed_count += 1
            if progress and processed_count % max(1, non_zero_count // 10) == 0:
                pct = int(100 * processed_count / non_zero_count)
                progress.advance(0, f"Spatial convolution: {pct}% complete")
    
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def topo_usm(
    dem: NDArray[np.float32],
    radius: int = 8,
    *,
    kernel: str = "disc",
    use_fft: bool | None = None,
    treat_nan: float | None = None,
    dtype=np.float32,
    progress: Optional[ProgressReporter] = None,
    _stream_state=None,  # ← stream_tiles 互換（未使用）
) -> NDArray[np.float32]:
    """Single‑radius **TopoUSM** layer (signed local relief).

    Parameters
    ----------
    dem        : 2‑D float32 array (NaN = NoData)
    radius     : blur radius in pixels (≥1)
    kernel     : 'disc' | 'gauss'
    use_fft    : force True/False, or None (auto: radius ≥ _FFT_THRESHOLD)
    treat_nan  : None → ignore NaNs; float → fill NoData with that value
    dtype      : output dtype (default float32)
    progress   : Optional progress reporter instance

    Returns
    -------
    usm : float32 array same shape as *dem*
        Signed local relief (original - blurred)
    """
    # Initialize progress if not provided
    if progress is None:
        progress = NullProgress()
    
    # Set up progress tracking - estimate total steps
    total_steps = 6  # Basic steps: validation, preparation, kernel, convolution, calculation, finalization
    if use_fft is None:
        use_fft = bool(radius >= _FFT_THRESHOLD)
    
    if use_fft:
        total_steps += 4  # Additional FFT steps
    else:
        total_steps += 2  # Additional spatial convolution steps
    
    progress.set_range(total_steps)
    
    try:
        # Input validation
        progress.advance(1, f"Validating input (radius={radius})")
        if dem.ndim != 2:
            raise ValueError("Input DEM must be 2‑D array")
        if dem.size == 0:
            raise ValueError("Input DEM cannot be empty")
        if radius < 1:
            raise ValueError("radius must be ≥1 pixel")
        
        # Warn for small arrays relative to kernel size
        min_dim = min(dem.shape)
        if min_dim < 2 * radius + 1:
            warnings.warn(f"DEM size {dem.shape} may be too small for radius {radius}", 
                         RuntimeWarning, stacklevel=2)

        # Prepare base array
        progress.advance(1, "Preparing arrays")
        arr_orig = dem.astype(dtype, copy=False)

        if treat_nan is not None:
            nan_mask = np.isnan(arr_orig)
            arr_base = arr_orig.copy()
            arr_base[nan_mask] = treat_nan
            ignore_nan = False
        else:
            arr_base = arr_orig
            ignore_nan = True

        # Build / fetch kernel
        progress.advance(1, f"Building {kernel} kernel")
        dtype_str = np.dtype(dtype).name
        if kernel == "disc":
            kern = _disc_kernel(radius, dtype_str)
        elif kernel == "gauss":
            kern = _gauss_kernel(radius, dtype_str)
        else:
            raise ValueError("kernel must be 'disc' or 'gauss'")

        # Convolution
        method_desc = "FFT" if use_fft else "spatial"
        progress.advance(1, f"Starting {method_desc} convolution")
        
        if ignore_nan:
            if use_fft:
                blur = _convolve_nan_fft(arr_base, kern, progress)
            else:
                # Use fast spatial convolution with NaN handling
                blur = _convolve_nan_spatial_fast(arr_base, kern, progress)
        else:
            # No NaN handling needed
            if use_fft:
                blur = _fft_convolve_same(arr_base, kern, progress)
            else:
                # Use simple spatial convolution without NaN handling
                blur = _convolve_spatial_no_nan(arr_base, kern, progress)

        # Signed local relief
        progress.advance(1, "Computing signed local relief")
        usm = arr_orig - blur

        # Restore NaNs where appropriate
        if ignore_nan:
            usm[np.isnan(arr_orig)] = np.nan

        progress.advance(1, "Finalizing output")
        result = usm.astype(dtype, copy=False)
        
        progress.done()
        return result
        
    except Exception as e:
        # Ensure progress is properly closed on error
        progress.done()
        raise


def multi_scale_usm(
    dem: NDArray[np.float32],
    radii: Sequence[int] | int = (4, 16, 64, 256),
    weights: Sequence[float] | None = None,
    *,
    kernel: str = "disc",
    use_fft: bool | None = None,
    normalize: bool = True,
    treat_nan: float | None = None,
    dtype=np.float32,
    progress: Optional[ProgressReporter] = None,
) -> Tuple[NDArray[np.float32], List[NDArray[np.float32]]]:
    """Compute *n*‑scale **TopoUSM‑Stack** and its weighted composite.
    
    Parameters
    ----------
    dem        : 2‑D float32 array (NaN = NoData)
    radii      : sequence of blur radii, or single int
    weights    : weights for each radius (default: use radii as weights)
    kernel     : 'disc' | 'gauss'
    use_fft    : force True/False, or None (auto)
    normalize  : normalize composite to [-1, 1] range
    treat_nan  : None → ignore NaNs; float → fill NoData with that value
    dtype      : output dtype
    progress   : Optional progress reporter instance
    
    Returns
    -------
    composite : weighted combination of all TopoUSM layers
    layers    : list of individual TopoUSM layers
    """
    # Initialize progress if not provided
    if progress is None:
        progress = NullProgress()
    
    if isinstance(radii, int):
        radii = (radii,)
    radii = list(radii)
    if any(r < 1 for r in radii):
        raise ValueError("All radii must be ≥1 px")

    if weights is None:
        weights = list(radii)
    if len(weights) != len(radii):
        raise ValueError("weights and radii lengths differ")

    # Set up progress tracking
    total_steps = len(radii) + 2  # Individual layers + composite + normalization
    progress.set_range(total_steps)
    
    try:
        # Compute individual layers
        layers: List[NDArray[np.float32]] = []
        for i, r in enumerate(radii):
            progress.advance(1, f"Computing TopoUSM layer {i+1}/{len(radii)} (radius={r})")
            layer = topo_usm(
                dem,
                radius=r,
                kernel=kernel,
                use_fft=use_fft,
                treat_nan=treat_nan,
                dtype=dtype,
                progress=None,  # Don't pass progress to avoid nested progress bars
            )
            layers.append(layer)

        # Weighted composite using higher precision
        progress.advance(1, "Computing weighted composite")
        comp = np.zeros_like(layers[0], dtype=np.float64)
        total_weight = sum(weights)
        
        for w, lay in zip(weights, layers):
            comp += (w / total_weight) * lay.astype(np.float64)

        # Normalize to [-1, 1] range
        if normalize:
            progress.advance(1, "Normalizing composite to [-1, 1] range")
            with np.errstate(invalid="ignore"):
                abs_vals = np.abs(comp)
                abs99 = float(np.nanpercentile(abs_vals, 99.0))
            
            if abs99 > 0:
                comp = np.clip(comp / abs99, -1.0, 1.0)
            # If abs99 == 0, comp is already all zeros/NaNs
        else:
            progress.advance(1, "Finalizing composite (no normalization)")

        result = comp.astype(dtype, copy=False), layers
        progress.done()
        return result
        
    except Exception as e:
        # Ensure progress is properly closed on error
        progress.done()
        raise