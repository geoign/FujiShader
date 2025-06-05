"""
FujiShader.shader.tpi
=====================

Terrain Position Index (TPI) & Local Relief Model (LRM)
------------------------------------------------------
* **TPI**  = elevation − mean elevation in surrounding window (±r px or metres)
* **LRM**  = positive TPI (ridges) + |negative TPI| (valleys)  → always ≥0

Both are azimuth‐independent and highlight subtle geomorphic features.  TPI is
signed; LRM is unsigned for shading pipelines.

QGIS Compatible Version - Does not use scipy to avoid import issues.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

# プログレス関連のimport（progress.pyから）
try:
    from ..core.progress import ProgressReporter, NullProgress
except ImportError:
    # スタンドアロン実行時やimportエラー時の代替
    from typing import Protocol
    
    class ProgressReporter(Protocol):
        def set_range(self, maximum: int) -> None: ...
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
        def done(self) -> None: ...
    
    class NullProgress:
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: pass
        def done(self) -> None: pass

__all__ = ["tpi", "local_relief"]


def _radius_to_window(radius_m: float, cellsize: float) -> int:
    """Convert radius in meters to window half-size in pixels."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if cellsize <= 0:
        raise ValueError("cellsize must be positive")
    return max(1, int(np.ceil(radius_m / cellsize)))


def _uniform_filter_numpy(
    arr: NDArray[np.float32], 
    window_size: int,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """
    Pure numpy implementation of uniform filter using convolution.
    Replaces scipy.ndimage.uniform_filter for QGIS compatibility.
    
    Args:
        arr: Input array
        window_size: Size of the uniform filter window (must be odd)
        progress: Progress reporter for tracking computation
        
    Returns:
        Filtered array with same shape as input
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    # Create uniform kernel
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)
    
    # Pad array to handle edges (nearest mode)
    pad_width = window_size // 2
    padded = np.pad(arr, pad_width, mode='edge')
    
    # Initialize output array
    filtered = np.zeros_like(arr)
    
    # Progress setup
    if progress:
        total_pixels = arr.shape[0] * arr.shape[1]
        progress.set_range(total_pixels)
        processed = 0
        update_interval = max(1, total_pixels // 100)  # Update every 1% of pixels
    
    # Apply convolution manually for better control
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Extract window from padded array
            window = padded[i:i+window_size, j:j+window_size]
            # Compute mean
            filtered[i, j] = np.sum(window * kernel)
            
            # Update progress
            if progress:
                processed += 1
                if processed % update_interval == 0 or processed == total_pixels:
                    progress.advance(update_interval if processed % update_interval == 0 else processed % update_interval,
                                   f"Filtering: {processed}/{total_pixels} pixels")
    
    if progress:
        progress.done()
    
    return filtered


def _uniform_filter_optimized(
    arr: NDArray[np.float32], 
    window_size: int,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """
    Optimized numpy implementation using integral images for better performance.
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    # Progress setup
    if progress:
        # 処理ステップ: パディング(10%) + 積分画像計算(20%) + フィルタリング(70%)
        progress.set_range(100)
        progress.advance(0, "Starting optimized filtering...")
    
    # Pad array to handle edges
    pad_width = window_size // 2
    padded = np.pad(arr, pad_width, mode='edge')
    
    if progress:
        progress.advance(10, "Array padded")
    
    # Create integral image for fast sum computation
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    if progress:
        progress.advance(20, "Integral image computed")
    
    # Initialize output
    filtered = np.zeros_like(arr)
    
    # Progress tracking for filtering
    total_pixels = arr.shape[0] * arr.shape[1]
    processed = 0
    update_interval = max(1, total_pixels // 70)  # 70% of progress for filtering
    
    # Compute window sums using integral image
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Calculate window boundaries in integral image coordinates
            y1, y2 = i, i + window_size
            x1, x2 = j, j + window_size
            
            # Compute sum using integral image
            window_sum = (integral[y2, x2] - 
                         (integral[y1, x2] if y1 > 0 else 0) -
                         (integral[y2, x1] if x1 > 0 else 0) +
                         (integral[y1, x1] if y1 > 0 and x1 > 0 else 0))
            
            # Compute mean
            filtered[i, j] = window_sum / (window_size * window_size)
            
            # Update progress
            if progress:
                processed += 1
                if processed % update_interval == 0 or processed == total_pixels:
                    current_progress = 30 + int((processed / total_pixels) * 70)
                    progress.advance(1 if processed % update_interval == 0 else 0,
                                   f"Optimized filtering: {processed}/{total_pixels} pixels")
    
    if progress:
        progress.done()
    
    return filtered


def tpi(
    dem: NDArray[np.float32],
    *,
    radius_m: float = 100.0,
    cellsize: float = 1.0,
    use_optimized: bool = True,
    progress: Optional[ProgressReporter] = None,
) -> NDArray[np.float32]:
    """
    Signed Terrain Position Index.
    
    Args:
        dem: Digital elevation model as 2D array
        radius_m: Radius of analysis window in meters (default: 100.0)
        cellsize: Size of each cell in meters (default: 1.0)
        use_optimized: Use optimized integral image algorithm (default: True)
        progress: Progress reporter for tracking computation
        
    Returns:
        TPI values as 2D array with same shape as input DEM
        
    Raises:
        ValueError: If radius_m or cellsize are not positive, or if dem is empty
    """
    # デフォルトのプログレスレポーターを設定
    if progress is None:
        progress = NullProgress()
    
    # Input validation
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    if cellsize <= 0:
        raise ValueError("cellsize must be positive")
    if dem.size == 0:
        raise ValueError("dem array is empty")
    if dem.ndim != 2:
        raise ValueError("dem must be a 2D array")
    
    progress.advance(0, f"Starting TPI calculation (radius: {radius_m}m)")
    
    # Convert radius to window size
    w = _radius_to_window(radius_m, cellsize)
    window_size = w * 2 + 1
    
    progress.advance(0, f"Window size: {window_size}x{window_size} pixels")
    
    # Ensure input is float32 (avoid unnecessary copying if already correct type)
    dem_f32 = dem.astype(np.float32, copy=False)
    
    # Compute mean elevation in surrounding window
    if use_optimized and dem.shape[0] * dem.shape[1] > 10000:
        # Use optimized version for large arrays
        progress.advance(0, "Using optimized algorithm for large DEM")
        mean_elevation = _uniform_filter_optimized(dem_f32, window_size, progress)
    else:
        # Use simple version for small arrays or when optimization is disabled
        progress.advance(0, "Using standard algorithm")
        mean_elevation = _uniform_filter_numpy(dem_f32, window_size, progress)
    
    progress.advance(0, "Computing final TPI values...")
    
    # TPI = elevation - mean elevation
    tpi_result = dem_f32 - mean_elevation
    
    progress.advance(0, "TPI calculation completed")
    progress.done()
    
    return tpi_result


def local_relief(
    dem: NDArray[np.float32],
    *,
    radius_m: float = 100.0,
    cellsize: float = 1.0,
    use_optimized: bool = True,
    progress: Optional[ProgressReporter] = None,
) -> NDArray[np.float32]:
    """
    Unsigned Local Relief Model (≥0).
    
    Args:
        dem: Digital elevation model as 2D array
        radius_m: Radius of analysis window in meters (default: 100.0)
        cellsize: Size of each cell in meters (default: 1.0)
        use_optimized: Use optimized integral image algorithm (default: True)
        progress: Progress reporter for tracking computation
        
    Returns:
        Local relief values as 2D array (always ≥ 0)
    """
    # デフォルトのプログレスレポーターを設定
    if progress is None:
        progress = NullProgress()
    
    progress.advance(0, "Starting Local Relief Model calculation")
    
    # TPIを計算（プログレスは内部で処理される）
    tpi_values = tpi(dem, radius_m=radius_m, cellsize=cellsize, 
                     use_optimized=use_optimized, progress=progress)
    
    progress.advance(0, "Converting to absolute values...")
    
    # 絶対値を取る
    relief_result = np.abs(tpi_values).astype(np.float32)
    
    progress.advance(0, "Local Relief Model calculation completed")
    progress.done()
    
    return relief_result


# Convenience function for testing/debugging
def _test_functions():
    """Test function to verify implementation correctness."""
    # Create simple test DEM
    test_dem = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 5, 4, 2],
        [3, 5, 6, 5, 3],
        [2, 4, 5, 4, 2],
        [1, 2, 3, 2, 1]
    ], dtype=np.float32)
    
    print("Test DEM:")
    print(test_dem)
    
    # CLI環境でのテスト用簡易プログレスレポーター
    class SimpleProgress:
        def __init__(self):
            self.max_val = 100
            self.current = 0
            
        def set_range(self, maximum: int) -> None:
            self.max_val = maximum
            self.current = 0
            
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            self.current += step
            if text:
                progress_pct = min(100, (self.current * 100 / self.max_val))
                print(f"[{progress_pct:5.1f}%] {text}")
                
        def done(self) -> None:
            print("[100.0%] Completed!")
    
    progress_reporter = SimpleProgress()
    
    # Test TPI
    print("\n=== Testing TPI ===")
    tpi_result = tpi(test_dem, radius_m=1.5, cellsize=1.0, progress=progress_reporter)
    print("\nTPI result:")
    print(tpi_result)
    
    # Test Local Relief
    print("\n=== Testing Local Relief ===")
    lr_result = local_relief(test_dem, radius_m=1.5, cellsize=1.0, progress=progress_reporter)
    print("\nLocal Relief result:")
    print(lr_result)
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    _test_functions()