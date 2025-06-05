"""
FujiShader.shader.sunsky
========================

Physically‐inspired *directional* illumination models (QGIS optimized)
---------------------------------------------------------------------
This module introduces two complementary shading functions:

* **`direct_light`** – strict Lambertian hillshade with **optional cast
  shadows** for a single solar position (azimuth, altitude).
* **`sky_light`** – hemispherical skylight integral using a discrete set of
  azimuths; essentially *direction‐weighted SVF*.

Both return *irradiance weights* in the range **[0–1]** so they can be fed
into colour pipelines (e.g. warm–cool compositing) or multiplied with
reflectance maps.

Design choices
~~~~~~~~~~~~~~
* Same *cellsize / max_radius / n_directions* parameters as other FujiShader
  modules → consistent UX.
* Pure NumPy implementation optimized for QGIS environments.
* Cast‐shadow routine uses vectorized operations for better performance.
* Memory-efficient implementation suitable for large DEMs.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional
import warnings
import math

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["direct_light", "sky_light"]

# -----------------------------------------------------------------------------
# Helper: gradient → unit normal (optimized)
# -----------------------------------------------------------------------------

def _unit_normals(dem: NDArray[np.float32], dy: float, dx: float) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Return (nx, ny, nz) unit surface normals via central differences.
    
    Optimized for memory efficiency and speed in QGIS environments.
    """
    # Use numpy gradient with edge_order=1 for better edge handling
    dz_dy, dz_dx = np.gradient(dem, dy, dx, edge_order=1)
    
    # Compute magnitude once and reuse
    mag_sq = dz_dx * dz_dx + dz_dy * dz_dy + 1.0
    nz = np.reciprocal(np.sqrt(mag_sq))  # Slightly faster than 1.0 / sqrt
    
    # Compute normals
    nx = -dz_dx * nz
    ny = -dz_dy * nz
    
    return nx.astype(np.float32, copy=False), ny.astype(np.float32, copy=False), nz.astype(np.float32, copy=False)

# -----------------------------------------------------------------------------
# Vectorized cast‐shadow computation
# -----------------------------------------------------------------------------

def _compute_shadow_mask_vectorized(
    dem: NDArray[np.float32],
    max_steps: int,
    sin_az: float,
    cos_az: float,
    sin_alt: float,
    dy: float,
    dx: float,
    chunk_size: int = 1000,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.bool_]:
    """
    Vectorized shadow computation using chunked processing for memory efficiency.
    
    This approach processes pixels in chunks to balance memory usage and performance,
    making it suitable for large DEMs in QGIS.
    """
    h, w = dem.shape
    shadow_mask = np.zeros((h, w), dtype=np.bool_)
    
    if progress is None:
        progress = NullProgress()
    
    # Pre-compute step vectors for efficiency
    step_dx = -cos_az * np.arange(1, max_steps + 1)
    step_dy = -sin_az * np.arange(1, max_steps + 1)
    
    # Process in chunks to manage memory
    total_pixels = h * w
    total_chunks = (total_pixels + chunk_size - 1) // chunk_size
    
    # Set up progress reporting
    progress.set_range(total_chunks * max_steps)
    processed_operations = 0
    
    for chunk_idx, chunk_start in enumerate(range(0, total_pixels, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_pixels)
        chunk_indices = np.arange(chunk_start, chunk_end)
        
        # Convert flat indices to 2D coordinates
        y_coords = chunk_indices // w
        x_coords = chunk_indices % w
        
        # Get elevation at current positions
        z0 = dem[y_coords, x_coords]
        
        # Check each step for this chunk
        for step_idx in range(max_steps):
            dx_step = step_dx[step_idx]
            dy_step = step_dy[step_idx]
            
            # Calculate target positions
            xx = x_coords + dx_step
            yy = y_coords + dy_step
            
            # Check bounds
            valid_mask = (xx >= 0) & (yy >= 0) & (xx < w - 1) & (yy < h - 1)
            
            if np.any(valid_mask):
                # Get integer and fractional parts for interpolation
                xx_valid = xx[valid_mask]
                yy_valid = yy[valid_mask]
                ix = np.floor(xx_valid).astype(np.int32)
                iy = np.floor(yy_valid).astype(np.int32)
                fx = xx_valid - ix
                fy = yy_valid - iy
                
                # Ensure indices are within bounds
                ix = np.clip(ix, 0, w - 2)
                iy = np.clip(iy, 0, h - 2)
                
                # Bilinear interpolation
                z_interp = (
                    dem[iy, ix] * (1 - fx) * (1 - fy) +
                    dem[iy, ix + 1] * fx * (1 - fy) +
                    dem[iy + 1, ix] * (1 - fx) * fy +
                    dem[iy + 1, ix + 1] * fx * fy
                )
                
                # Calculate horizontal distance and elevation angle
                horiz_dist = np.sqrt(
                    (dx * (x_coords[valid_mask] - xx_valid)) ** 2 +
                    (dy * (y_coords[valid_mask] - yy_valid)) ** 2
                )
                
                # Avoid division by zero
                horiz_dist = np.maximum(horiz_dist, 1e-8)
                
                elevation_angle = np.arctan2(z_interp - z0[valid_mask], horiz_dist)
                sun_angle = np.arcsin(sin_alt)
                
                # Update shadow mask for pixels that are now in shadow
                shadow_pixels = chunk_indices[valid_mask][elevation_angle > sun_angle]
                if len(shadow_pixels) > 0:
                    shadow_y = shadow_pixels // w
                    shadow_x = shadow_pixels % w
                    shadow_mask[shadow_y, shadow_x] = True
            
            # Update progress
            processed_operations += 1
            if processed_operations % 100 == 0:  # Update every 100 operations to avoid overhead
                progress.advance(100, f"Shadow computation: chunk {chunk_idx + 1}/{total_chunks}, step {step_idx + 1}/{max_steps}")
    
    # Advance any remaining progress
    remaining = total_chunks * max_steps - processed_operations
    if remaining > 0:
        progress.advance(remaining)
    
    return shadow_mask

def _compute_shadow_mask_simple(
    dem: NDArray[np.float32],
    max_steps: int,
    sin_az: float,
    cos_az: float,
    sin_alt: float,
    dy: float,
    dx: float,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.bool_]:
    """
    Simplified shadow computation for smaller DEMs or when memory is not a concern.
    
    This version processes all pixels simultaneously for maximum vectorization.
    """
    h, w = dem.shape
    shadow_mask = np.zeros((h, w), dtype=np.bool_)
    
    if progress is None:
        progress = NullProgress()
    
    # Set up progress reporting
    progress.set_range(max_steps)
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    z0 = dem.copy()
    
    sun_angle = np.arcsin(sin_alt)
    
    # Check each step
    for step in range(1, max_steps + 1):
        # Calculate sample positions
        xx = x_grid - cos_az * step
        yy = y_grid - sin_az * step
        
        # Check bounds
        valid = (xx >= 0) & (yy >= 0) & (xx <= w - 1) & (yy <= h - 1)
        
        if not np.any(valid):
            # Update progress for remaining steps
            progress.advance(max_steps - step + 1, "Shadow computation completed early (no valid pixels)")
            break
            
        # Bilinear interpolation
        ix = np.clip(xx.astype(np.int32), 0, w - 2)
        iy = np.clip(yy.astype(np.int32), 0, h - 2)
        fx = np.clip(xx - ix, 0, 1)
        fy = np.clip(yy - iy, 0, 1)
        
        # Ensure indices are valid
        ix = np.where(valid, ix, 0)
        iy = np.where(valid, iy, 0)
        
        z_interp = (
            dem[iy, ix] * (1 - fx) * (1 - fy) +
            dem[iy, np.minimum(ix + 1, w - 1)] * fx * (1 - fy) +
            dem[np.minimum(iy + 1, h - 1), ix] * (1 - fx) * fy +
            dem[np.minimum(iy + 1, h - 1), np.minimum(ix + 1, w - 1)] * fx * fy
        )
        
        # Calculate elevation angle
        horiz_dist = np.sqrt((dx * (x_grid - xx)) ** 2 + (dy * (y_grid - yy)) ** 2)
        horiz_dist = np.maximum(horiz_dist, 1e-8)  # Avoid division by zero
        
        elevation_angle = np.arctan2(z_interp - z0, horiz_dist)
        
        # Update shadow mask
        shadow_mask |= valid & (elevation_angle > sun_angle)
        
        # Update progress
        progress.advance(1, f"Shadow computation: step {step}/{max_steps}")
    
    return shadow_mask

# -----------------------------------------------------------------------------
# Public functions
# -----------------------------------------------------------------------------

def direct_light(
    dem: NDArray[np.float32],
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    cast_shadows: bool = True,
    max_shadow_radius: float = 500.0,
    memory_efficient: bool = True,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """Lambertian hillshade with optional cast shadows (QGIS optimized).

    Parameters
    ----------
    dem : ndarray (H, W)
        Elevation raster.
    azimuth_deg, altitude_deg : float
        Solar position (0° = north, clockwise positive).
    cellsize : float or (dy, dx)
        Pixel size. Units must match *max_shadow_radius*.
    cast_shadows : bool
        Whether to compute cast‐shadow mask.
    max_shadow_radius : float
        Search distance for shadow casting.
    memory_efficient : bool, default True
        Use chunked processing for large DEMs. Set False for small DEMs
        where full vectorization is preferred.
    progress : ProgressReporter, optional
        Progress reporting callback.

    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        Hillshade values where 0=fully shadowed, 1=fully illuminated.
    """
    # Input validation
    if dem.ndim != 2:
        raise ValueError("DEM must be a 2D array")
    if not (0 <= altitude_deg <= 90):
        raise ValueError("Altitude must be in range [0, 90] degrees")
    
    if progress is None:
        progress = NullProgress()
    
    # Set up main progress steps
    total_steps = 3 if cast_shadows and max_shadow_radius > 0 else 2
    progress.set_range(total_steps)
    
    progress.advance(0, "Computing surface normals...")
    
    # Handle cellsize parameter
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    
    # Convert angles to radians
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    sin_alt = math.sin(alt)
    cos_alt = math.cos(alt)
    sin_az = math.sin(az)
    cos_az = math.cos(az)

    # Ensure dem is float32 for consistency
    dem_f32 = dem.astype(np.float32, copy=False)
    
    # Compute surface normals
    nx, ny, nz = _unit_normals(dem_f32, dy, dx)
    progress.advance(1, "Computing illumination...")
    
    # Cosine of incidence angle (Lambert's law)
    cos_i = (nx * cos_az + ny * sin_az) * cos_alt + nz * sin_alt
    cos_i = np.clip(cos_i, 0.0, 1.0)

    # Compute cast shadows if requested
    if cast_shadows and max_shadow_radius > 0:
        progress.advance(0, "Computing cast shadows...")
        
        max_steps = max(1, int(max_shadow_radius / max(dx, dy)))
        
        # Choose shadow computation method based on DEM size and user preference
        total_pixels = dem.shape[0] * dem.shape[1]
        use_chunked = memory_efficient or total_pixels > 1_000_000
        
        # Create sub-progress reporter for shadow computation
        class SubProgress:
            def __init__(self, parent_progress: ProgressReporter):
                self.parent = parent_progress
                self._max = 100
                self._current = 0
            
            def set_range(self, maximum: int) -> None:
                self._max = maximum
                self._current = 0
            
            def advance(self, step: int = 1, text: Optional[str] = None) -> None:
                self._current += step
                if text:
                    self.parent.advance(0, text)
        
        sub_progress = SubProgress(progress)
        
        if use_chunked:
            shadow_mask = _compute_shadow_mask_vectorized(
                dem_f32, max_steps, sin_az, cos_az, sin_alt, dy, dx, progress=sub_progress
            )
        else:
            shadow_mask = _compute_shadow_mask_simple(
                dem_f32, max_steps, sin_az, cos_az, sin_alt, dy, dx, progress=sub_progress
            )
        
        # Apply shadows
        cos_i[shadow_mask] = 0.0
        progress.advance(1, "Shadows applied")
    
    progress.advance(1, "Finalizing direct light computation...")
    
    return cos_i.astype(np.float32, copy=False)


def sky_light(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    weight_cos2: bool = True,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """Isotropic skylight integral (0–1) - QGIS optimized.

    Parameters
    ----------
    dem : ndarray (H,W)
        Elevation raster.
    cellsize : float or (dy,dx)
        Pixel size.
    max_radius : float
        Horizon scan distance.
    n_directions : int
        Number of azimuth sectors.
    weight_cos2 : bool, default True
        Weight each direction by ``cos^2(theta)`` (Oke‐style). If False uses
        simple SVF average.
    progress : ProgressReporter, optional
        Progress reporting callback.

    Returns
    -------
    sky : ndarray (H, W) in [0,1]
        Sky view factor where 0=completely obstructed, 1=full sky visibility.
    """
    if progress is None:
        progress = NullProgress()
    
    # Set up progress tracking
    progress.set_range(3)
    progress.advance(0, "Computing sky view factor...")
    
    # Local import to avoid circular dependencies
    from FujiShader.shader.skyview import skyview_factor as _svf

    # Create sub-progress reporter for SVF computation
    class SubProgress:
        def __init__(self, parent_progress: ProgressReporter):
            self.parent = parent_progress
        
        def set_range(self, maximum: int) -> None:
            pass  # Parent handles main progress
        
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text:
                self.parent.advance(0, f"SVF: {text}")
        
        def done(self) -> None:
            pass
    
    sub_progress = SubProgress(progress)
    svf = _svf(dem, cellsize=cellsize, max_radius=max_radius, 
               n_directions=n_directions, progress=sub_progress)
    
    progress.advance(1, "Applying weighting...")
    
    if weight_cos2:
        # Convert SVF to cosine-squared weighted sky illumination
        # SVF represents 1 - mean(cos^2(θ)), so we need 1 - SVF
        cos2_mean = 1.0 - svf
        sky = cos2_mean
    else:
        # Use raw SVF
        sky = svf
    
    progress.advance(1, "Finalizing sky light...")
    
    result = np.clip(sky, 0.0, 1.0).astype(np.float32, copy=False)
    progress.advance(1, "Sky light computation complete")
    
    return result


# -----------------------------------------------------------------------------
# Utility functions for QGIS integration
# -----------------------------------------------------------------------------

def estimate_memory_usage(dem_shape: Tuple[int, int], max_shadow_radius: float, 
                         cellsize: float) -> dict:
    """
    Estimate memory usage for shadow computation to help users choose appropriate settings.
    
    Parameters
    ----------
    dem_shape : tuple of int
        Shape of the DEM (height, width).
    max_shadow_radius : float
        Maximum shadow search radius.
    cellsize : float
        Cell size of the DEM.
    
    Returns
    -------
    dict
        Dictionary with memory estimates in MB.
    """
    h, w = dem_shape
    total_pixels = h * w
    max_steps = max(1, int(max_shadow_radius / cellsize))
    
    # Estimate memory for different approaches
    base_memory = total_pixels * 4 / 1024 / 1024  # DEM in float32
    shadow_mask_memory = total_pixels / 1024 / 1024 / 8  # boolean mask
    
    # Simple vectorized approach
    simple_temp_memory = total_pixels * 8 * 4 / 1024 / 1024  # coordinate grids in float64
    
    # Chunked approach (assuming 1000 pixel chunks)
    chunk_memory = min(1000, total_pixels) * 8 * 4 / 1024 / 1024
    
    return {
        'base_memory_mb': base_memory,
        'shadow_mask_mb': shadow_mask_memory,
        'simple_method_peak_mb': base_memory + simple_temp_memory + shadow_mask_memory,
        'chunked_method_peak_mb': base_memory + chunk_memory + shadow_mask_memory,
        'max_steps': max_steps,
        'recommended_method': 'chunked' if total_pixels > 500_000 else 'simple'
    }


def get_optimal_chunk_size(available_memory_mb: float, dem_shape: Tuple[int, int]) -> int:
    """
    Calculate optimal chunk size based on available memory.
    
    Parameters
    ----------
    available_memory_mb : float
        Available memory in megabytes.
    dem_shape : tuple of int
        Shape of the DEM.
    
    Returns
    -------
    int
        Optimal chunk size in pixels.
    """
    # Estimate memory per pixel for processing (roughly 32 bytes per pixel)
    memory_per_pixel = 32
    max_pixels = int(available_memory_mb * 1024 * 1024 / memory_per_pixel)
    
    # Ensure we don't exceed total pixels
    total_pixels = dem_shape[0] * dem_shape[1]
    chunk_size = min(max_pixels, total_pixels)
    
    # Ensure minimum chunk size for efficiency
    return max(100, chunk_size)


# -----------------------------------------------------------------------------
# Convenience functions for batch processing
# -----------------------------------------------------------------------------

def compute_sun_sky_composite(
    dem: NDArray[np.float32],
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    cast_shadows: bool = True,
    max_shadow_radius: float = 500.0,
    max_sky_radius: float = 100.0,
    n_directions: int = 16,
    sun_weight: float = 0.7,
    sky_weight: float = 0.3,
    memory_efficient: bool = True,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """
    Compute a weighted composite of direct sunlight and sky illumination.
    
    This is a convenience function that combines direct_light and sky_light
    with proper progress reporting for the entire operation.
    
    Parameters
    ----------
    dem : ndarray (H, W)
        Elevation raster.
    azimuth_deg, altitude_deg : float
        Solar position parameters.
    cellsize : float or (dy, dx)
        Pixel size.
    cast_shadows : bool
        Whether to compute cast shadows for direct light.
    max_shadow_radius : float
        Shadow search radius for direct light.
    max_sky_radius : float
        Sky view radius.
    n_directions : int
        Number of azimuth directions for sky computation.
    sun_weight, sky_weight : float
        Weights for combining direct and sky illumination.
    memory_efficient : bool
        Use memory-efficient processing.
    progress : ProgressReporter, optional
        Progress reporting callback.
    
    Returns
    -------
    composite : ndarray (H, W) in [0,1]
        Weighted composite illumination.
    """
    if progress is None:
        progress = NullProgress()
    
    progress.set_range(3)
    
    # Compute direct light
    progress.advance(0, "Computing direct sunlight...")
    
    class DirectLightProgress:
        def __init__(self, parent: ProgressReporter):
            self.parent = parent
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text: self.parent.advance(0, f"Direct: {text}")
        def done(self) -> None: pass
    
    direct = direct_light(
        dem, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        cellsize=cellsize, cast_shadows=cast_shadows,
        max_shadow_radius=max_shadow_radius, memory_efficient=memory_efficient,
        progress=DirectLightProgress(progress)
    )
    
    progress.advance(1, "Computing sky illumination...")
    
    # Compute sky light
    class SkyLightProgress:
        def __init__(self, parent: ProgressReporter):
            self.parent = parent
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text: self.parent.advance(0, f"Sky: {text}")
        def done(self) -> None: pass
    
    sky = sky_light(
        dem, cellsize=cellsize, max_radius=max_sky_radius,
        n_directions=n_directions, progress=SkyLightProgress(progress)
    )
    
    progress.advance(1, "Combining illumination...")
    
    # Normalize weights
    total_weight = sun_weight + sky_weight
    if total_weight <= 0:
        raise ValueError("Total weight must be positive")
    
    sun_weight_norm = sun_weight / total_weight
    sky_weight_norm = sky_weight / total_weight
    
    # Combine
    composite = sun_weight_norm * direct + sky_weight_norm * sky
    
    progress.advance(1, "Composite illumination complete")
    
    return np.clip(composite, 0.0, 1.0).astype(np.float32, copy=False)