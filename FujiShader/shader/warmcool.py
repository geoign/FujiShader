"""
FujiShader.shader.warmcool
==========================

Colourising terrain with a *warm-ridges / cool-valleys* palette that is fully
azimuth-independent and grounded in human visual science.

The function ``warmcool_map`` takes **TopoUSM** (signed local relief) and
optionally **slope** (surface roughness) and **sky-view factor** (environmental
occlusion).  It produces an ``(H, W, 3)`` RGB array in the range ``[0, 1]``.

The algorithm follows these design rules
----------------------------------------
* **Hue   – TopoUSM sign**: positive → warm (reddish–yellow), negative → cool
  (cyan-blue).
* **Value – Luminance base**: starts at ``1`` and is reduced by weighted slope
  and SVF to mimic darker steep or sheltered areas.
* **Chroma – |TopoUSM|**: larger local relief → higher colour saturation.

All maths is vectorised NumPy; no external dependencies beyond NumPy itself.
Writing/reading rasters is handled by :pymod:`FujiShader.core.io`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["warmcool_map"]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

_P99_EPS: float = 1e-6  # Prevent divide-by-zero when data are flat


def _normalize_signed(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Normalize signed data to •±1 by *pct*-percentile."""
    hi = float(np.nanpercentile(np.abs(arr), pct))
    if hi < _P99_EPS:
        hi = 1.0  # fall-back for nearly flat rasters
    return np.clip(arr / hi, -1.0, 1.0)


def _normalize_scalar(arr: np.ndarray, ref: float) -> np.ndarray:
    """Normalize positive scalar field to 0-1 relative to *ref*."""
    return np.clip(arr / ref, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def warmcool_map(
    usm: np.ndarray,
    *,
    slope: Optional[np.ndarray] = None,
    svf: Optional[np.ndarray] = None,
    slope_ref: float = 45.0,
    slope_weight: float = 0.4,
    svf_weight: float = 0.3,
    warm_gain: float = 0.5,
    cool_gain: float = 0.5,
    dtype=np.float32,
    progress: Optional[ProgressReporter] = None,
) -> np.ndarray:
    """Generate a warm–cool RGB relief image from terrain layers.

    Parameters
    ----------
    usm : ndarray (H, W)
        Signed local relief from TopoUSM.  Positive → ridges, negative → valleys.
    slope : ndarray (H, W), optional
        Slope in degrees or percent.  Darkens steep surfaces; leave ``None`` to
        skip.
    svf : ndarray (H, W), optional
        Sky-view factor [0–1].  Darkens sheltered terrain; leave ``None`` to skip.
    slope_ref : float, default 45.0
        Slope value corresponding to maximum darkening (degrees).
    slope_weight : float, default 0.4
        Contribution of *slope* to the luminance term (0–1).
    svf_weight : float, default 0.3
        Contribution of *svf* to the luminance term (0–1).
    warm_gain : float, default 0.5
        Colour gain applied to ridges (positive USM).
    cool_gain : float, default 0.5
        Colour gain applied to valleys (negative USM).
    dtype : numpy dtype, default ``np.float32``
        Output array data-type.
    progress : ProgressReporter, optional
        Progress reporter for tracking computation progress.

    Returns
    -------
    rgb : ndarray (H, W, 3)
        RGB image in range ``[0, 1]``.
    """
    if usm.ndim != 2:
        raise ValueError("usm must be a 2-D array (height × width)")

    # Initialize progress reporter
    if progress is None:
        progress = NullProgress()
    
    # Calculate total steps based on optional inputs
    total_steps = 7  # Base steps: normalize USM, init L, warm mask, cool mask, stack, clamp, done
    if slope is not None:
        total_steps += 1  # slope processing
    if svf is not None:
        total_steps += 1  # svf processing
    
    progress.set_range(total_steps)
    current_step = 0

    # ------------------------------------------------------------------
    # 1. Normalise input layers
    # ------------------------------------------------------------------
    progress.advance(1, "Normalizing USM data...")
    current_step += 1
    usm_n = _normalize_signed(usm.astype(dtype))  # ±1

    # Base luminance starts at 1 (white)
    progress.advance(1, "Initializing luminance...")
    current_step += 1
    L = np.ones_like(usm_n, dtype=dtype)

    # Slope contribution (optional)
    if slope is not None:
        progress.advance(1, "Processing slope data...")
        current_step += 1
        slope_n = _normalize_scalar(slope.astype(dtype), slope_ref)  # 0–1
        L = L - slope_weight * slope_n

    # Sky-view factor contribution (optional)
    if svf is not None:
        progress.advance(1, "Processing sky-view factor...")
        current_step += 1
        svf_n = np.clip(svf.astype(dtype), 0.0, 1.0)
        #L = L - svf_weight * svf_n
        L = L - svf_weight * (1.0 - svf_n)

    # Clamp luminance to [0,1]
    L = np.clip(L, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2. Build RGB channels
    # ------------------------------------------------------------------
    progress.advance(1, "Initializing RGB channels...")
    current_step += 1
    R = L.copy()
    G = L.copy()
    B = L.copy()

    # Warm ridges (USM > 0)
    progress.advance(1, "Processing warm ridges...")
    current_step += 1
    warm_mask = usm_n > 0
    if warm_mask.any():
        usm_pos = usm_n[warm_mask]
        R[warm_mask] += warm_gain * usm_pos
        G[warm_mask] += warm_gain * 0.5 * usm_pos
        B[warm_mask] -= warm_gain * 0.5 * usm_pos

    # Cool valleys (USM < 0)
    progress.advance(1, "Processing cool valleys...")
    current_step += 1
    cool_mask = usm_n < 0
    if cool_mask.any():
        usm_neg_abs = np.abs(usm_n[cool_mask])
        R[cool_mask] -= cool_gain * 0.5 * usm_neg_abs
        G[cool_mask] += cool_gain * 0.25 * usm_neg_abs
        B[cool_mask] += cool_gain * usm_neg_abs

    # ------------------------------------------------------------------
    # 3. Stack and clamp
    # ------------------------------------------------------------------
    progress.advance(1, "Finalizing RGB image...")
    current_step += 1
    rgb = np.clip(np.stack((R, G, B), axis=-1), 0.0, 1.0).astype(dtype)
    
    progress.done()
    return rgb