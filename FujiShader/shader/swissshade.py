"""
FujiShader.shader.swissshade
===========================

Swiss‐style multi‐directional hillshade (diffuse soft shading)
--------------------------------------------------------------
Combines 3 or more low-intensity hillshades from different azimuths and a
moderate altitude to achieve a soft, directionally neutral relief—popularised
by Eduard Imhof.

This implementation provides a robust, memory-efficient approach suitable for
large DEMs in QGIS environments.
"""
from __future__ import annotations

from typing import Iterable, Union, Tuple
import warnings
import numpy as np
from numpy.typing import NDArray

from FujiShader.shader.sunsky import direct_light
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["swiss_shade", "estimate_swiss_memory"]


def swiss_shade(
    dem: NDArray[np.float32],
    *,
    azimuths_deg: Iterable[float] = (225, 315, 45, 135),
    altitude_deg: float = 45.0,
    cellsize: Union[float, Tuple[float, float]] = 1.0,
    weight: float = 1.0,
    cast_shadows: bool = False,
    max_shadow_radius: float = 100.0,
    memory_efficient: bool = True,
    normalize_by_count: bool = True,
    progress: ProgressReporter = None
) -> NDArray[np.float32]:
    """Return composite Swiss hillshade in [0,1].
    
    Parameters
    ----------
    dem : ndarray (H, W)
        Digital elevation model as 2D array.
    azimuths_deg : iterable of float, default (225, 315, 45, 135)
        Azimuth angles in degrees (0° = north, clockwise positive).
        Must contain at least one azimuth.
    altitude_deg : float, default 45.0
        Solar altitude angle in degrees [0, 90].
    cellsize : float or (dy, dx), default 1.0
        Pixel size. If tuple, (row_spacing, col_spacing).
    weight : float, default 1.0
        Global weight factor applied to final result.
    cast_shadows : bool, default False
        Whether to compute cast shadows. Generally disabled for Swiss shading
        to maintain soft, even illumination.
    max_shadow_radius : float, default 100.0
        Maximum search distance for shadow casting (if enabled).
    memory_efficient : bool, default True
        Use chunked processing for large DEMs.
    normalize_by_count : bool, default True
        Normalize result by number of directions to maintain [0,1] range
        regardless of azimuth count.
    progress : ProgressReporter, optional
        Progress reporting callback.
        
    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        Composite Swiss hillshade where 0=dark, 1=bright.
        
    Raises
    ------
    ValueError
        If DEM is not 2D, no azimuths provided, or altitude out of range.
    TypeError
        If DEM is not numeric array.
        
    Notes
    -----
    Swiss shading typically uses 3-4 cardinal/intercardinal directions
    (N, NE, E, SE or NW, NE, SE, SW) at moderate altitude (30-60°).
    Cast shadows are usually disabled to preserve the soft, even lighting
    characteristic of this technique.
    
    Examples
    --------
    >>> dem = np.random.rand(100, 100).astype(np.float32)
    >>> # Basic Swiss shading with 4 directions
    >>> shade = swiss_shade(dem)
    >>> # Custom directions with shadows
    >>> shade = swiss_shade(dem, azimuths_deg=[0, 90, 180, 270], 
    ...                    cast_shadows=True)
    """
    # Initialize progress reporter
    if progress is None:
        progress = NullProgress()
        
    # Setup progress tracking
    # 処理ステップ: 検証(1) + 各方位での計算(n) + 合成(1) + 最終化(1)
    azimuth_list = list(azimuths_deg)
    if not azimuth_list:
        raise ValueError("At least one azimuth must be provided")
    
    total_steps = 1 + len(azimuth_list) + 1 + 1  # validation + hillshades + combine + finalize
    progress.set_range(total_steps)
    
    # Input validation
    progress.advance(1, "Validating inputs...")
    
    if not isinstance(dem, np.ndarray):
        raise TypeError("DEM must be a numpy array")
    
    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")
        
    if not np.issubdtype(dem.dtype, np.number):
        raise TypeError("DEM must contain numeric values")
    
    if not (0 <= altitude_deg <= 90):
        raise ValueError(f"Altitude must be in range [0, 90] degrees, got {altitude_deg}")
    
    if weight < 0:
        warnings.warn("Negative weight may produce unexpected results", UserWarning)
    
    # Ensure DEM is float32 for consistency
    dem_f32 = dem.astype(np.float32, copy=False)
    
    # Compute individual hillshades
    shades = []
    total_azimuths = len(azimuth_list)
    
    for i, az in enumerate(azimuth_list):
        step_text = f"Computing hillshade {i+1}/{total_azimuths} (azimuth {az}°)"
        
        try:
            # Create a sub-progress for this hillshade computation
            # direct_light may also use progress, so we pass it through
            shade = direct_light(
                dem_f32,
                azimuth_deg=float(az),
                altitude_deg=altitude_deg,
                cellsize=cellsize,
                cast_shadows=cast_shadows,
                max_shadow_radius=max_shadow_radius,
                memory_efficient=memory_efficient,
                progress=None  # Let direct_light handle its own progress if needed
            )
            shades.append(shade)
            
            # Advance progress after completing this hillshade
            progress.advance(1, step_text)
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute hillshade for azimuth {az}°: {e}") from e
    
    # Combine shades using mean
    progress.advance(1, "Combining hillshades...")
    
    if len(shades) == 1:
        composite = shades[0]
    else:
        # Stack arrays for efficient mean computation
        shade_stack = np.stack(shades, axis=0)
        composite = np.mean(shade_stack, axis=0)
    
    # Apply weight
    if weight != 1.0:
        composite = composite * weight
    
    # Optional normalization
    if normalize_by_count and len(shades) > 1:
        # Already normalized by np.mean, but ensure we maintain proper scaling
        pass
    
    # Finalize result
    progress.advance(1, "Finalizing result...")
    
    # Ensure output is in [0,1] range and float32
    result = np.clip(composite, 0.0, 1.0).astype(np.float32, copy=False)
    
    # Mark completion
    progress.done()
    
    return result


def estimate_swiss_memory(
    dem_shape: Tuple[int, int],
    n_azimuths: int = 4,
    cast_shadows: bool = False,
    max_shadow_radius: float = 100.0,
    cellsize: float = 1.0
) -> dict:
    """Estimate memory usage for Swiss shading computation.
    
    Parameters
    ----------
    dem_shape : tuple of int
        Shape of the DEM (height, width).
    n_azimuths : int, default 4
        Number of azimuth directions.
    cast_shadows : bool, default False
        Whether cast shadows will be computed.
    max_shadow_radius : float, default 100.0
        Shadow search radius (if applicable).
    cellsize : float, default 1.0
        Cell size for shadow computation.
        
    Returns
    -------
    dict
        Memory usage estimates in MB.
        
    Examples
    --------
    >>> mem_info = estimate_swiss_memory((1000, 1000), n_azimuths=4)
    >>> print(f"Estimated peak memory: {mem_info['peak_memory_mb']:.1f} MB")
    """
    h, w = dem_shape
    total_pixels = h * w
    
    # Base memory for DEM
    dem_memory = total_pixels * 4 / 1024 / 1024  # float32
    
    # Memory for individual shades (stored temporarily)
    shade_memory = total_pixels * 4 / 1024 / 1024  # float32
    
    # Memory for stacking operation
    stack_memory = n_azimuths * shade_memory
    
    # Shadow computation memory (if enabled)
    shadow_memory = 0
    if cast_shadows:
        from FujiShader.shader.sunsky import estimate_memory_usage
        shadow_info = estimate_memory_usage(dem_shape, max_shadow_radius, cellsize)
        shadow_memory = shadow_info['chunked_method_peak_mb']
    
    # Peak memory occurs during stacking operation
    peak_memory = dem_memory + stack_memory + shadow_memory
    
    return {
        'dem_memory_mb': dem_memory,
        'single_shade_mb': shade_memory,
        'stack_memory_mb': stack_memory,
        'shadow_memory_mb': shadow_memory,
        'peak_memory_mb': peak_memory,
        'n_azimuths': n_azimuths,
        'total_pixels': total_pixels,
        'recommended_chunked': peak_memory > 500  # Recommend chunking if >500MB
    }


# Utility function for common Swiss shading configurations
def swiss_shade_classic(
    dem: NDArray[np.float32],
    *,
    style: str = "imhof",
    cellsize: Union[float, Tuple[float, float]] = 1.0,
    intensity: float = 1.0,
    progress: ProgressReporter = None
) -> NDArray[np.float32]:
    """Classic Swiss shading configurations popularized by cartographers.
    
    Parameters
    ----------
    dem : ndarray (H, W)
        Digital elevation model.
    style : str, default "imhof"
        Predefined style. Options:
        - "imhof": Eduard Imhof's classic 4-direction setup
        - "jenny": Bernhard Jenny's 3-direction variant
        - "cardinal": Simple N-E-S-W configuration
    cellsize : float or tuple, default 1.0
        Pixel size.
    intensity : float, default 1.0
        Overall intensity multiplier.
    progress : ProgressReporter, optional
        Progress callback.
        
    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        Swiss hillshade using the specified classic configuration.
    """
    # Initialize progress if not provided
    if progress is None:
        progress = NullProgress()
    
    style_configs = {
        "imhof": {
            "azimuths_deg": (315, 45, 135, 225),  # NW, NE, SE, SW
            "altitude_deg": 45.0
        },
        "jenny": {
            "azimuths_deg": (315, 45, 180),  # NW, NE, S
            "altitude_deg": 50.0
        },
        "cardinal": {
            "azimuths_deg": (0, 90, 180, 270),  # N, E, S, W
            "altitude_deg": 40.0
        }
    }
    
    if style not in style_configs:
        available = ", ".join(style_configs.keys())
        raise ValueError(f"Unknown style '{style}'. Available: {available}")
    
    config = style_configs[style]
    
    # Set up progress for classic style
    progress.set_range(1)
    progress.advance(1, f"Applying {style} Swiss shading style...")
    
    result = swiss_shade(
        dem,
        azimuths_deg=config["azimuths_deg"],
        altitude_deg=config["altitude_deg"],
        cellsize=cellsize,
        weight=intensity,
        cast_shadows=False,  # Classic Swiss shading doesn't use shadows
        progress=progress
    )
    
    progress.done()
    return result