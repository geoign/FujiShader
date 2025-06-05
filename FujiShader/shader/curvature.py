"""
FujiShader.shader.curvature
===========================

Profile, plan and total curvature (second‐order derivatives)
-----------------------------------------------------------
Implements the algorithm outlined in Zevenbergen & Thorne (1987) using 3×3
window coefficients. Returns curvature in 1/m; sign convention follows GDAL
(+ values = convex for profile curvature).

The implementation uses finite difference methods to compute first and second
partial derivatives of elevation data, then applies the curvature formulas
according to Zevenbergen & Thorne (1987).

Public functions
~~~~~~~~~~~~~~~~
* `profile_curvature` - Curvature in the direction of maximum slope
* `plan_curvature` - Curvature perpendicular to the direction of maximum slope  
* `total_curvature` - Mean curvature (Laplacian)

References
~~~~~~~~~~
Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land 
surface topography. Earth surface processes and landforms, 12(1), 47-56.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional, Protocol
import numpy as np
from numpy.typing import NDArray

# Define progress reporter protocol for type hints
class ProgressReporterProtocol(Protocol):
    """Protocol for progress reporting."""
    def set_range(self, maximum: int) -> None: ...
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
    def done(self) -> None: ...


# Import progress reporting - adjust path as needed for your module structure
try:
    from ..core.progress import ProgressReporter, NullProgress
except ImportError:
    # Fallback for standalone usage
    class NullProgress:
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: pass
        def done(self) -> None: pass

__all__ = [
    "profile_curvature",
    "plan_curvature", 
    "total_curvature",
]


def _validate_dem(dem: NDArray[np.float32]) -> None:
    """Validate DEM input array.
    
    Args:
        dem: Digital elevation model array
        
    Raises:
        ValueError: If DEM is too small or has invalid shape
    """
    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")
    
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        raise ValueError("DEM must be at least 3x3 pixels for curvature calculation")


def _gradients_2nd(
    dem: NDArray[np.float32], 
    dy: float, 
    dx: float,
    progress: Optional[ProgressReporterProtocol] = None
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], 
    NDArray[np.float32], NDArray[np.float32]
]:
    """Compute first and second partial derivatives using finite differences.
    
    Uses central difference approximations for first derivatives and 
    second-order finite differences for second derivatives.
    
    Args:
        dem: Digital elevation model array
        dy: Cell size in y direction (typically negative for north-up grids)
        dx: Cell size in x direction
        progress: Optional progress reporter
        
    Returns:
        Tuple of (dz/dx, dz/dy, d²z/dx², d²z/dy², d²z/dxdy)
        All arrays have shape (height-2, width-2)
    """
    if progress is None:
        progress = NullProgress()
    
    z = dem.astype(np.float32, copy=False)
    
    # First partial derivatives using central differences
    # All resulting arrays will be (height-2, width-2)
    progress.advance(1, "第1次偏微分を計算中 (dz/dx)")
    dzdx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2.0 * dx)
    
    progress.advance(1, "第1次偏微分を計算中 (dz/dy)")
    dzdy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2.0 * dy)
    
    # Second partial derivatives using finite differences
    progress.advance(1, "第2次偏微分を計算中 (d²z/dx²)")
    d2zdx2 = (z[1:-1, 2:] - 2.0 * z[1:-1, 1:-1] + z[1:-1, :-2]) / (dx * dx)
    
    progress.advance(1, "第2次偏微分を計算中 (d²z/dy²)")
    d2zdy2 = (z[2:, 1:-1] - 2.0 * z[1:-1, 1:-1] + z[:-2, 1:-1]) / (dy * dy)
    
    # Mixed partial derivative (cross-derivative)
    progress.advance(1, "交差偏微分を計算中 (d²z/dxdy)")
    d2zdxdy = (
        z[2:, 2:] + z[:-2, :-2] - z[2:, :-2] - z[:-2, 2:]
    ) / (4.0 * dx * dy)
    
    return dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy


def _curvature_impl(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    kind: str,
    progress: Optional[ProgressReporterProtocol] = None,
) -> NDArray[np.float32]:
    """Internal implementation for curvature calculations.
    
    Args:
        dem: Digital elevation model array
        cellsize: Cell size as (dy, dx) tuple or single value for square cells
        kind: Type of curvature - 'profile', 'plan', or 'total'
        progress: Optional progress reporter
        
    Returns:
        Curvature array with same shape as input DEM, NaN at borders
        
    Raises:
        ValueError: If kind is not recognized or DEM is invalid
    """
    _validate_dem(dem)
    
    # Handle cellsize parameter
    if np.isscalar(cellsize):
        dy, dx = float(cellsize), float(cellsize)
    else:
        dy, dx = float(cellsize[0]), float(cellsize[1])
    
    # Initialize progress reporting
    if progress is None:
        progress = NullProgress()
    
    # プログレス範囲を設定（偏微分計算5ステップ + 曲率計算1ステップ + 出力配列作成1ステップ）
    progress.set_range(7)
    
    progress.advance(1, f"{kind}曲率の計算を開始")
    
    # Compute partial derivatives
    dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = _gradients_2nd(dem, dy, dx, progress)
    
    # Assign common variable names following Zevenbergen & Thorne notation
    p = dzdx  # First derivative in x direction
    q = dzdy  # First derivative in y direction  
    a = d2zdx2  # Second derivative in x direction
    b = d2zdxdy  # Mixed second derivative
    c = d2zdy2  # Second derivative in y direction
    
    # Compute curvature based on type
    progress.advance(1, f"{kind}曲率を計算中")
    
    if kind == "profile":
        # Profile curvature: curvature in direction of maximum slope
        # Formula: -(a*p² + 2*b*p*q + c*q²) / (1 + p² + q²)^(3/2)
        denom = (1.0 + p**2 + q**2) ** 1.5
        # Add small epsilon to prevent division by zero in flat areas
        denom = np.maximum(denom, 1e-10)
        curv = -((a * p**2 + 2.0 * b * p * q + c * q**2) / denom)
        
    elif kind == "plan":
        # Plan curvature: curvature perpendicular to direction of maximum slope
        # Formula: -(a*q² - 2*b*p*q + c*p²) / (1 + p² + q²)^(1/2)
        denom = (1.0 + p**2 + q**2) ** 0.5
        # Add small epsilon to prevent division by zero in flat areas
        denom = np.maximum(denom, 1e-10)
        curv = -((a * q**2 - 2.0 * b * p * q + c * p**2) / denom)
        
    elif kind == "total":
        # Total curvature (mean curvature): Laplacian of elevation
        # Formula: a + c = d²z/dx² + d²z/dy²
        curv = a + c
        
    else:
        raise ValueError(f"Unknown curvature kind '{kind}'. Must be 'profile', 'plan', or 'total'")
    
    # Create output array with NaN borders
    progress.advance(1, "出力配列を作成中")
    out = np.full_like(dem, np.nan, dtype=np.float32)
    out[1:-1, 1:-1] = curv
    
    progress.done()
    
    return out


def profile_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """Compute profile curvature from a digital elevation model.
    
    Profile curvature measures the rate of change of slope in the direction
    of maximum slope. Positive values indicate convex areas (ridges),
    negative values indicate concave areas (valleys).
    
    Args:
        dem: Digital elevation model as 2D array (must be at least 3x3)
        cellsize: Cell size as (dy, dx) tuple or single value for square cells.
                 Units should match DEM elevation units.
        progress: Optional progress reporter for tracking computation progress
    
    Returns:
        Profile curvature array with same shape as input DEM.
        Border pixels are set to NaN. Units are 1/[length_unit].
        
    Raises:
        ValueError: If DEM is too small or has invalid dimensions
        
    Example:
        >>> import numpy as np
        >>> # Create a simple ridge
        >>> dem = np.array([[1, 2, 1], [1, 3, 1], [1, 2, 1]], dtype=np.float32)
        >>> prof_curv = profile_curvature(dem, cellsize=1.0)
        >>> # Central pixel should show positive curvature (convex ridge)
    """
    return _curvature_impl(dem, cellsize=cellsize, kind="profile", progress=progress)


def plan_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """Compute plan curvature from a digital elevation model.
    
    Plan curvature measures the rate of change of aspect (direction of slope).
    It represents curvature perpendicular to the direction of maximum slope.
    Positive values indicate divergent flow areas, negative values indicate
    convergent flow areas.
    
    Args:
        dem: Digital elevation model as 2D array (must be at least 3x3)
        cellsize: Cell size as (dy, dx) tuple or single value for square cells.
                 Units should match DEM elevation units.
        progress: Optional progress reporter for tracking computation progress
    
    Returns:
        Plan curvature array with same shape as input DEM.
        Border pixels are set to NaN. Units are 1/[length_unit].
        
    Raises:
        ValueError: If DEM is too small or has invalid dimensions
        
    Example:
        >>> import numpy as np
        >>> # Create a simple valley
        >>> dem = np.array([[2, 1, 2], [2, 1, 2], [2, 1, 2]], dtype=np.float32)
        >>> plan_curv = plan_curvature(dem, cellsize=1.0)
        >>> # Central pixel should show negative curvature (convergent)
    """
    return _curvature_impl(dem, cellsize=cellsize, kind="plan", progress=progress)


def total_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """Compute total curvature (mean curvature) from a digital elevation model.
    
    Total curvature is the Laplacian of the elevation surface and represents
    the mean curvature at each point. It is the sum of the principal curvatures
    and equals the sum of profile and plan curvature in most cases.
    
    Args:
        dem: Digital elevation model as 2D array (must be at least 3x3)
        cellsize: Cell size as (dy, dx) tuple or single value for square cells.
                 Units should match DEM elevation units.
        progress: Optional progress reporter for tracking computation progress
    
    Returns:
        Total curvature array with same shape as input DEM.
        Border pixels are set to NaN. Units are 1/[length_unit].
        
    Raises:
        ValueError: If DEM is too small or has invalid dimensions
        
    Example:
        >>> import numpy as np
        >>> # Create a simple dome
        >>> dem = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.float32)
        >>> total_curv = total_curvature(dem, cellsize=1.0)
        >>> # Central pixel should show positive curvature (convex dome)
    """
    return _curvature_impl(dem, cellsize=cellsize, kind="total", progress=progress)