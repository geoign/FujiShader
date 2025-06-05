"""
FujiShader.shader.integral
==============================

超大 DEM 向け **Summed‑Area Table (積分画像) ベース** 
----------------------------------------------------------------------
* FFT を一切使わず、2 回の累積和と差分だけで **O(N)** にスケール。
* 円盤平均の完全再現ではなく **正方窓** (square box) 平均を採用
  ⇒ 尾根/谷の符号表現は同じで、計算は桁違いに高速。
* SciPy も Numba も不要。NumPy さえあれば QGIS コンソールで動作。
* 元 ``topo_usm`` / ``multi_scale_usm`` とシグネチャ互換。

制限事項
~~~~~~~~
* 正方窓なので半径 r に対応する *円盤* 平均とスペクトルが若干異なります
  （視覚的にはほぼ問題ないことを確認）。
* 複数スケール合成を行う際に、円盤版と全く同じウェイトを使うと
  コントラストがわずかに変わることがあります。必要ならウェイトを微調整してください。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union
import warnings

import numpy as np
from numpy.typing import NDArray

# プログレス管理のインポート
from ..core.progress import ProgressReporter, NullProgress

__all__ = [
    "topo_integral",
    "multi_scale_integral",
    "integral_image",
]

# Typing aliases for clarity
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

def _validate_input_array(arr: np.ndarray, name: str = "array") -> None:
    """Validate input array properties."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {arr.ndim}D")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")


def _moving_sum_axis(
    arr: Float32Array, 
    r: int, 
    axis: int,
    progress: ProgressReporter = None
) -> Float32Array:
    """(2 r + 1)‑wide running **sum** along *axis* using an integral image.

    The DEM is zero‑padded by *r* cells on both sides, then a cumulative sum
    is computed. By *prepending a zero slice* to that prefix sum we can take
    window sums as `csum[i + w] − csum[i]`, which yields an output array of
    **exactly the same shape as the original input** (no off‑by‑one).
    
    Args:
        arr: Input array
        r: Radius (must be >= 0)
        axis: Axis along which to compute moving sum (0 or 1)
        progress: Progress reporter instance
        
    Returns:
        Array of same shape as input with moving sums
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"Radius must be >= 0, got {r}")
    if axis not in (0, 1):
        raise ValueError(f"Axis must be 0 or 1, got {axis}")
    if r == 0:
        # No blur requested - return copy
        progress.advance(1, f"No blur needed (r=0) for axis {axis}")
        return arr.astype(np.float32, copy=True)

    # Validate array shape for given axis
    if arr.shape[axis] == 0:
        raise ValueError("Cannot apply moving sum to zero-sized axis")

    axis_name = "Y" if axis == 0 else "X"
    progress.advance(1, f"Padding array for {axis_name}-axis blur (r={r})")

    # Create padding specification
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[axis] = (r, r)
    
    try:
        padded = np.pad(arr, pad_spec, mode="constant", constant_values=0.0)
    except Exception as e:
        raise ValueError(f"Failed to pad array: {e}")

    # Compute prefix sum with float64 for numerical stability
    progress.advance(1, f"Computing cumulative sum along {axis_name}-axis")
    try:
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Failed to compute cumulative sum: {e}")

    # Prepend a single zero slice along the integration axis
    progress.advance(1, f"Preparing window extraction for {axis_name}-axis")
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    try:
        zero_slice = np.zeros(zero_shape, dtype=np.float64)
        csum = np.concatenate((zero_slice, csum), axis=axis)
    except Exception as e:
        raise ValueError(f"Failed to concatenate zero slice: {e}")

    # Extract window sums
    n = arr.shape[axis]
    w = 2 * r + 1  # window width

    # Create slice objects for difference operation
    sl_hi = [slice(None)] * arr.ndim
    sl_lo = [slice(None)] * arr.ndim
    sl_hi[axis] = slice(w, w + n)
    sl_lo[axis] = slice(0, n)

    progress.advance(1, f"Extracting {axis_name}-axis window sums")
    try:
        result = csum[tuple(sl_hi)] - csum[tuple(sl_lo)]
        return result.astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"Failed to compute window differences: {e}")


def _square_sum(
    arr: Float32Array, 
    r: int, 
    progress: ProgressReporter = None
) -> Float32Array:
    """Return the (2 r + 1)×(2 r + 1) **sum** for every pixel.
    
    Args:
        arr: Input array (NaNs should be pre-handled)
        r: Radius in pixels
        progress: Progress reporter instance
        
    Returns:
        Array of same shape with square window sums
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"Radius must be >= 0, got {r}")
    if r == 0:
        progress.advance(2, "No square blur needed (r=0)")
        return arr.astype(np.float32, copy=True)
        
    # Apply moving sum in both dimensions
    try:
        tmp = _moving_sum_axis(arr, r, axis=1, progress=progress)  # X pass
        return _moving_sum_axis(tmp, r, axis=0, progress=progress)  # Y pass
    except Exception as e:
        raise ValueError(f"Failed to compute square sum: {e}")


def _square_blur_nan(
    arr: Float32Array, 
    r: int, 
    progress: ProgressReporter = None
) -> Float32Array:
    """NaN‑aware square‑window **mean** of radius *r*.
    
    Args:
        arr: Input array
        r: Radius in pixels
        progress: Progress reporter instance
        
    Returns:
        Array of same shape with NaN-aware square window means
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"Radius must be >= 0, got {r}")
    
    # Handle trivial case
    if r == 0:
        progress.advance(1, "No blur needed (r=0)")
        return arr.astype(np.float32, copy=True)
    
    # Separate NaN handling
    progress.advance(1, f"Analyzing NaN distribution for blur (r={r})")
    nan_mask = np.isnan(arr)
    has_nan = np.any(nan_mask)
    
    if not has_nan:
        # No NaNs - simple case
        progress.advance(1, "Computing simple square blur (no NaNs)")
        window_area = (2 * r + 1) ** 2
        total_sum = _square_sum(arr, r, progress)
        return total_sum / window_area
    
    # NaN-aware computation
    progress.advance(1, "Preparing NaN-aware blur computation")
    arr_filled = np.where(nan_mask, 0.0, arr).astype(np.float32, copy=False)
    weight = (~nan_mask).astype(np.float32, copy=False)

    try:
        progress.advance(1, "Computing numerator (filled values)")
        numerator = _square_sum(arr_filled, r, progress)
        progress.advance(1, "Computing denominator (valid weights)")
        denominator = _square_sum(weight, r, progress)
    except Exception as e:
        raise ValueError(f"Failed to compute NaN-aware sums: {e}")

    # Compute mean with proper NaN handling
    progress.advance(1, "Computing final NaN-aware mean")
    result = np.empty_like(arr, dtype=np.float32)
    
    # Use numpy's error handling for division
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = denominator > 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        result[~valid_mask] = np.nan
    
    return result


def integral_image(
    img: np.ndarray, 
    progress: ProgressReporter = None
) -> Float32Array:
    """Compute the integral image (summed-area table) of the input.
    
    Args:
        img: Input 2D array
        progress: Progress reporter instance
        
    Returns:
        Integral image of same shape as input
        
    Note:
        This is a simplified version focusing on the core functionality.
        Streaming capabilities have been removed for clarity and maintainability.
    """
    if progress is None:
        progress = NullProgress()
        
    _validate_input_array(img, "img")
    
    progress.set_range(3)
    progress.advance(1, "Converting to float64 for computation")
    
    try:
        # Convert to float64 for computation, then back to float32
        img_f64 = img.astype(np.float64, copy=False)
        progress.advance(1, "Computing cumulative sum (axis 0)")
        result = img_f64.cumsum(axis=0)
        progress.advance(1, "Computing cumulative sum (axis 1)")
        result = result.cumsum(axis=1)
        return result.astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"Failed to compute integral image: {e}")
    finally:
        progress.done()


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def topo_integral(
    dem: Union[np.ndarray, Float32Array],
    radius: int = 8,
    *,
    kernel: str = "square",  # kept for signature compatibility
    use_fft: bool | None = None,  # ignored – SAT is always spatial
    treat_nan: float | None = None,
    dtype=np.float32,
    progress: ProgressReporter = None,
) -> Float32Array:
    """Compute a signed local‑relief layer using a **square window** SAT blur.
    
    Args:
        dem: Input DEM as 2D array
        radius: Blur radius in pixels (must be >= 1)
        kernel: Kernel type (only "square" supported, kept for compatibility)
        use_fft: Ignored (kept for compatibility)
        treat_nan: If provided, replace NaN values with this value
        dtype: Output data type (should be float32 or compatible)
        progress: Progress reporter instance
        
    Returns:
        Unsharp mask result (original - blurred)
        
    Raises:
        ValueError: For invalid inputs
        TypeError: For wrong input types
    """
    if progress is None:
        progress = NullProgress()
    
    # Set up progress tracking - estimate steps needed
    total_steps = 10  # Base steps for setup, validation, blur, and finalization
    if np.any(np.isnan(dem)) and treat_nan is None:
        total_steps += 8  # Additional steps for NaN-aware processing
    
    progress.set_range(total_steps)
    
    try:
        # Input validation
        progress.advance(1, f"Validating input DEM (shape: {dem.shape})")
        _validate_input_array(dem, "dem")
        
        if radius < 1:
            raise ValueError(f"Radius must be >= 1 pixel, got {radius}")
        
        if kernel != "square":
            warnings.warn(f"Only 'square' kernel is supported, got '{kernel}'. Using 'square'.")
        
        # Convert to working dtype
        progress.advance(1, f"Converting to working dtype ({dtype})")
        try:
            arr_orig = dem.astype(dtype, copy=False)
        except Exception as e:
            raise ValueError(f"Failed to convert DEM to dtype {dtype}: {e}")

        # Handle NaN treatment
        progress.advance(1, "Processing NaN values")
        if treat_nan is not None:
            if not np.isfinite(treat_nan):
                raise ValueError("treat_nan must be a finite number")
            arr_base = np.where(np.isnan(arr_orig), treat_nan, arr_orig)
            preserve_nan = False
        else:
            arr_base = arr_orig
            preserve_nan = True

        # Compute blur and unsharp mask
        progress.advance(1, f"Starting square blur computation (radius={radius})")
        try:
            blur = _square_blur_nan(arr_base, radius, progress)
            progress.advance(1, "Computing unsharp mask (original - blurred)")
            usm = arr_orig - blur
        except Exception as e:
            raise ValueError(f"Failed to compute topographic unsharp mask: {e}")

        # Preserve original NaN locations if requested
        if preserve_nan:
            progress.advance(1, "Preserving original NaN locations")
            nan_mask = np.isnan(arr_orig)
            if np.any(nan_mask):
                usm[nan_mask] = np.nan
        else:
            progress.advance(1, "Finalizing result")

        progress.advance(1, f"Converting result to output dtype ({dtype})")
        result = usm.astype(dtype, copy=False)
        
        progress.advance(1, "Topographic integral computation complete")
        return result
        
    except Exception as e:
        progress.advance(1, f"Error occurred: {str(e)}")
        raise
    finally:
        progress.done()


def multi_scale_integral(
    dem: Union[np.ndarray, Float32Array],
    radii: Union[Sequence[int], int] = (4, 16, 64, 256),
    weights: Sequence[float] | None = None,
    *,
    kernel: str = "square",
    use_fft: bool | None = None,
    normalize: bool = True,
    treat_nan: float | None = None,
    dtype=np.float32,
    progress: ProgressReporter = None,
) -> Tuple[Float32Array, List[Float32Array]]:
    """Stack multiple SAT‑based USM layers into a composite relief image.
    
    Args:
        dem: Input DEM as 2D array
        radii: Sequence of blur radii, or single radius
        weights: Weights for combining layers (default: use radii as weights)
        kernel: Kernel type (only "square" supported)
        use_fft: Ignored (kept for compatibility)
        normalize: Whether to normalize output to [-1, 1] range
        treat_nan: If provided, replace NaN values with this value
        dtype: Output data type
        progress: Progress reporter instance
        
    Returns:
        Tuple of (composite_result, individual_layers)
        
    Raises:
        ValueError: For invalid inputs
    """
    if progress is None:
        progress = NullProgress()
    
    # Input validation
    _validate_input_array(dem, "dem")
    
    # Handle radii parameter
    if isinstance(radii, int):
        if radii < 1:
            raise ValueError(f"Radius must be >= 1, got {radii}")
        radii = (radii,)
    else:
        radii = list(radii)
        if not radii:
            raise ValueError("radii cannot be empty")
        if any(r < 1 for r in radii):
            raise ValueError(f"All radii must be >= 1 px, got {radii}")

    # Handle weights parameter
    if weights is None:
        weights = list(radii)  # Use radii as default weights
    else:
        weights = list(weights)
        if len(weights) != len(radii):
            raise ValueError(
                f"Length mismatch: weights ({len(weights)}) vs radii ({len(radii)})"
            )
        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive")

    if kernel != "square":
        warnings.warn(f"Only 'square' kernel is supported, got '{kernel}'. Using 'square'.")

    # Set up progress tracking
    num_layers = len(radii)
    total_steps = num_layers + 5  # layers + setup + combination + normalization + finalization
    progress.set_range(total_steps)
    
    try:
        progress.advance(1, f"Starting multi-scale computation with {num_layers} layers")
        progress.advance(1, f"Radii: {radii}, Weights: {weights}")

        # Compute individual layers
        layers: List[Float32Array] = []
        for i, r in enumerate(radii):
            try:
                progress.advance(1, f"Computing layer {i+1}/{num_layers} (radius={r})")
                
                # Create a sub-progress reporter for individual layer computation
                # Note: This is a simplified approach - in a more sophisticated implementation,
                # we might allocate progress proportionally to expected computation time
                layer = topo_integral(
                    dem,
                    radius=r,
                    treat_nan=treat_nan,
                    dtype=dtype,
                    progress=None,  # Use None to avoid nested progress reporting
                )
                layers.append(layer)
            except Exception as e:
                raise ValueError(f"Failed to compute layer {i} (radius={r}): {e}")

        # Combine layers with weights
        progress.advance(1, "Combining layers with weights")
        try:
            # Use float64 for accumulation to avoid precision loss
            composite = np.zeros_like(layers[0], dtype=np.float64)
            weight_sum = 0.0
            
            for w, layer in zip(weights, layers):
                composite += w * layer.astype(np.float64, copy=False)
                weight_sum += w
            
            # Normalize by weight sum
            if weight_sum > 0:
                composite /= weight_sum
            else:
                raise ValueError("Total weight sum is zero")
                
        except Exception as e:
            raise ValueError(f"Failed to combine layers: {e}")

        # Optional output normalization
        if normalize:
            progress.advance(1, "Normalizing output to [-1, 1] range")
            try:
                # Use 99th percentile for robust normalization
                abs_values = np.abs(composite)
                abs_values_finite = abs_values[np.isfinite(abs_values)]
                
                if len(abs_values_finite) > 0:
                    abs99 = float(np.percentile(abs_values_finite, 99.0))
                    abs99 = max(abs99, 1e-9)  # Avoid division by zero
                    composite = np.clip(composite / abs99, -1.0, 1.0)
                else:
                    # All values are non-finite
                    composite.fill(0.0)
                    
            except Exception as e:
                raise ValueError(f"Failed to normalize output: {e}")
        else:
            progress.advance(1, "Skipping normalization")

        # Convert to output dtype
        progress.advance(1, f"Converting result to output dtype ({dtype})")
        try:
            composite_result = composite.astype(dtype, copy=False)
        except Exception as e:
            raise ValueError(f"Failed to convert result to dtype {dtype}: {e}")

        progress.advance(1, "Multi-scale integral computation complete")
        return composite_result, layers
        
    except Exception as e:
        progress.advance(1, f"Error occurred: {str(e)}")
        raise
    finally:
        progress.done()