"""
FujiShader.shader.specular
==========================

Metallic / specular highlights for terrain
-----------------------------------------
A lightweight Blinn–Phong implementation that turns a DEM into a *specular
response* map – making ridges gleam like polished metal when combined with
diffuse shading.

Why Blinn–Phong?
~~~~~~~~~~~~~~~~
* Fast: a single `(N·H)^n` power term per pixel – fine for software or QGIS CPU.
* Physically plausible enough for cartography if you choose a low *n* (broad
  highlights) and mix with diffuse light.

Design notes
~~~~~~~~~~~~
* Assumes an **orthographic camera** (constant view vector) – suitable for
  map‑view renders.
* Compatible with `direct_light()`; you can combine like:

    ```python
    diff = fs.direct_light(dem, azimuth_deg=315, altitude_deg=45)
    spec = fs.metallic_shade(dem, azimuth_deg=315, altitude_deg=45,
                             shininess=32, view_alt_deg=60)
    rgb  = np.clip(diff[...,None] * base_rgb + spec[...,None]*[1,1,1], 0, 1)
    ```
* Numba optional; falls back to NumPy.
"""
from __future__ import annotations

import math
from typing import Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["metallic_shade"]

# -----------------------------------------------------------------------------
# Shared helper – surface normals (re‑use from sunsky but re‑implement locally)
# -----------------------------------------------------------------------------

def _unit_normals(arr: NDArray[np.float32], dy: float, dx: float) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Calculate unit surface normals from DEM using gradient."""
    dz_dy, dz_dx = np.gradient(arr, dy, dx, edge_order=1)
    nz = 1.0 / np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
    nx = -dz_dx * nz
    ny = -dz_dy * nz
    return nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)

# -----------------------------------------------------------------------------
# Core implementation – vectorised NumPy (fast enough; Numba not critical)
# -----------------------------------------------------------------------------

def metallic_shade(
    dem: NDArray[np.float32],
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    view_az_deg: Optional[float] = None,  # Python 3.9互換性のため変更
    view_alt_deg: float = 60.0,
    cellsize: Union[Tuple[float, float], float] = 1.0,  # Python 3.9互換性のため変更
    shininess: float = 32.0,
    specular_strength: float = 0.6,   # ← デフォルト弱め
    gamma: float = 2.2,               # ← sRGB 相当のガンマ
    progress: Optional[ProgressReporter] = None,
) -> NDArray[np.float32]:
    """Return Blinn–Phong specular term *S* in [0,1].

    Parameters
    ----------
    dem : ndarray (H, W)
        Elevation raster.
    azimuth_deg, altitude_deg : float
        Sun vector (0° = north; clockwise positive).
    view_az_deg : float or None, default None
        Camera azimuth. ``None`` = sun azimuth (top‑lit view).
    view_alt_deg : float, default 60
        Camera altitude (90° = nadir, 0° = horizon).
    cellsize : float or (dy,dx)
        Pixel size.
    shininess : float, default 32
        Phong exponent *n*; lower = broader highlights.
    specular_strength : float, default 0.6
        Linear multiplier (0–1 typically) to mix with diffuse.
    gamma : float, default 2.2
        Gamma correction factor for perceptual output (1.0 = linear).
        
    Returns
    -------
    ndarray (H, W)
        Specular response values in [0,1] range.
    """
    progress = progress or NullProgress()
    
    # プログレス範囲を設定（処理ステップ数に基づく）
    total_steps = 6  # 1. 準備, 2. 法線計算, 3. 光ベクトル, 4. 視点ベクトル, 5. スペキュラ計算, 6. ガンマ補正
    progress.set_range(total_steps)
    
    # ステップ1: 準備
    progress.advance(text="Preparing DEM data...")
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    
    # ステップ2: 表面法線の計算
    progress.advance(text="Computing surface normals...")
    nx, ny, nz = _unit_normals(dem.astype(np.float32, copy=False), dy, dx)

    # ステップ3: 光ベクトルの計算
    progress.advance(text="Setting up light vector...")
    azL = math.radians(azimuth_deg)
    altL = math.radians(altitude_deg)
    Lx = math.sin(azL) * math.cos(altL)
    Ly = math.cos(azL) * math.cos(altL)
    Lz = math.sin(altL)

    # ステップ4: 視点ベクトルとハーフベクトルの計算
    progress.advance(text="Computing view and half vectors...")
    if view_az_deg is None:
        view_az_deg = azimuth_deg  # look from sun direction by default
    azV = math.radians(view_az_deg)
    altV = math.radians(view_alt_deg)
    Vx = math.sin(azV) * math.cos(altV)
    Vy = math.cos(azV) * math.cos(altV)
    Vz = math.sin(altV)

    # Half vector H = normalize(L + V)
    Hx = Lx + Vx
    Hy = Ly + Vy
    Hz = Lz + Vz
    norm = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) + 1e-12
    Hx /= norm; Hy /= norm; Hz /= norm

    # ステップ5: スペキュラ項の計算
    progress.advance(text="Computing specular highlights...")
    # Dot product N·H (clamped >=0)
    ndh = np.clip(nx * Hx + ny * Hy + nz * Hz, 0.0, 1.0)
    # Specular term (linear)
    S_lin = specular_strength * (ndh ** shininess)

    # ステップ6: ガンマ補正と最終処理
    progress.advance(text="Applying gamma correction...")
    if gamma != 1.0:
        S_out = np.clip(S_lin, 0.0, 1.0) ** (1.0 / gamma)
    else:
        S_out = np.clip(S_lin, 0.0, 1.0)

    result = S_out.astype(np.float32, copy=False)

    # 処理完了
    progress.done()
    return result