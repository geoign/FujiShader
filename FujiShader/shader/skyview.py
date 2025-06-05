"""
FujiShader.shader.skyview
=========================

Sky‑View Factor (SVF) computation in pure Python/NumPy optimized for QGIS
environments. SVF expresses how much of the upper hemisphere is visible
from each DEM cell (0 = fully enclosed, 1 = unobstructed sky) and is widely
used as an azimuth‑independent proxy for ambient occlusion, cold‑air drainage
analysis, or 'warm–cool' colour mapping.

The classic definition is

    SVF = 1⁄n Σ cos² θᵢ            (Oke 1988)

where *θᵢ* is the elevation angle to the terrain horizon in direction *i*.
We approximate the integral with an evenly spaced set of azimuths (default
16).  Horizon angles are found by radial scanning out to a user‑defined
``max_radius`` (px or metres).

Key features
------------
* **Pure NumPy** – optimized for QGIS Python environments without external
  JIT compilers.
* **Vectorized operations** – efficient batch processing using NumPy's
  broadcasting and advanced indexing.
* **Metric‑aware** – accept pixel size in metres so the scan distance remains
  physically meaningful after reprojection.
* **Tiled processing** – no DEM is too large; tiles are merged seamlessly.
"""
from __future__ import annotations

from ..core.progress import ProgressReporter, NullProgress
from typing import Optional, Tuple

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["skyview_factor"]

# -----------------------------------------------------------------------------
# _horizon_scan: NumPy vectorized core
# -----------------------------------------------------------------------------

def _horizon_scan_vectorized(
    dem: NDArray[np.float32],
    max_steps: int,
    dx: float,
    dy: float,
    sin_az: NDArray[np.float32],
    cos_az: NDArray[np.float32],
    progress: ProgressReporter | None = None,
    progress_prefix: str = "",
) -> NDArray[np.float32]:
    """Return SVF for a tile using vectorized horizon scanning."""
    h, w = dem.shape
    n_dir = sin_az.size
    
    # 座標グリッドを事前作成
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    
    # 結果配列
    svf_sum = np.zeros((h, w), dtype=np.float32)
    
    # 進捗管理：方向数分の作業
    if progress:
        progress.set_range(n_dir)
    
    # 各方向について処理
    for d in range(n_dir):
        max_angles = np.full((h, w), -1e9, dtype=np.float32)
        
        # 各ステップについて処理（ベクトル化）
        for s in range(1, max_steps + 1):
            # 現在のステップでの座標
            xx = x_idx + cos_az[d] * s
            yy = y_idx + sin_az[d] * s
            
            # 境界チェック
            valid_mask = (
                (xx >= 0) & (yy >= 0) & 
                (xx < w - 1) & (yy < h - 1)
            )
            
            if not valid_mask.any():
                break  # この方向にはもう有効な点がない
            
            # 整数部と小数部
            ix = xx.astype(np.int32)
            iy = yy.astype(np.int32)
            fx = xx - ix
            fy = yy - iy
            
            # 双線形補間（有効な点のみ）
            z = np.zeros((h, w), dtype=np.float32)
            mask = valid_mask
            
            z[mask] = (
                dem[iy[mask], ix[mask]] * (1 - fx[mask]) * (1 - fy[mask]) +
                dem[iy[mask], ix[mask] + 1] * fx[mask] * (1 - fy[mask]) +
                dem[iy[mask] + 1, ix[mask]] * (1 - fx[mask]) * fy[mask] +
                dem[iy[mask] + 1, ix[mask] + 1] * fx[mask] * fy[mask]
            )
            
            # 距離と角度計算
            dz = z - dem
            dist_x = dx * (xx - x_idx)
            dist_y = dy * (yy - y_idx)
            dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
            
            # ゼロ除算対策
            dist = np.where(dist > 1e-10, dist, 1e-10)
            angles = np.arctan2(dz, dist)
            
            # 最大角度更新（有効な点のみ）
            update_mask = mask & (angles > max_angles)
            max_angles[update_mask] = angles[update_mask]
        
        # この方向のSVF寄与を加算
        svf_sum += np.cos(max_angles) ** 2
        
        # 進捗更新
        if progress:
            azimuth_deg = int(math.degrees(d * 2 * math.pi / n_dir))
            progress_text = f"{progress_prefix}方向 {azimuth_deg}° ({d+1}/{n_dir})"
            progress.advance(1, progress_text)
    
    return svf_sum / n_dir


def _horizon_scan_chunked(
    dem: NDArray[np.float32],
    max_steps: int,
    dx: float,
    dy: float,
    sin_az: NDArray[np.float32],
    cos_az: NDArray[np.float32],
    chunk_size: int = 1000,
    progress: ProgressReporter | None = None,
    progress_prefix: str = "",
) -> NDArray[np.float32]:
    """Memory-efficient version processing pixels in chunks."""
    h, w = dem.shape
    n_dir = sin_az.size
    result = np.zeros((h, w), dtype=np.float32)
    
    # ピクセルを線形インデックスでチャンク処理
    total_pixels = h * w
    total_work = math.ceil(total_pixels / chunk_size) * n_dir
    
    # 進捗管理：チャンク数 × 方向数
    if progress:
        progress.set_range(total_work)
    
    work_done = 0
    
    for chunk_start in range(0, total_pixels, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_pixels)
        
        # チャンク内のピクセル座標
        linear_idx = np.arange(chunk_start, chunk_end)
        y_chunk = linear_idx // w
        x_chunk = linear_idx % w
        chunk_len = len(linear_idx)
        
        svf_chunk = np.zeros(chunk_len, dtype=np.float32)
        chunk_number = chunk_start // chunk_size + 1
        total_chunks = math.ceil(total_pixels / chunk_size)
        
        # 各方向について処理
        for d in range(n_dir):
            max_angles = np.full(chunk_len, -1e9, dtype=np.float32)
            
            for s in range(1, max_steps + 1):
                xx = x_chunk + cos_az[d] * s
                yy = y_chunk + sin_az[d] * s
                
                # 境界チェック
                valid_mask = (
                    (xx >= 0) & (yy >= 0) & 
                    (xx < w - 1) & (yy < h - 1)
                )
                
                if not valid_mask.any():
                    break
                
                # 双線形補間
                ix = xx[valid_mask].astype(np.int32)
                iy = yy[valid_mask].astype(np.int32)
                fx = xx[valid_mask] - ix
                fy = yy[valid_mask] - iy
                
                z_interp = (
                    dem[iy, ix] * (1 - fx) * (1 - fy) +
                    dem[iy, ix + 1] * fx * (1 - fy) +
                    dem[iy + 1, ix] * (1 - fx) * fy +
                    dem[iy + 1, ix + 1] * fx * fy
                )
                
                # 角度計算
                z0_chunk = dem[y_chunk[valid_mask], x_chunk[valid_mask]]
                dz = z_interp - z0_chunk
                
                dist_x = dx * (xx[valid_mask] - x_chunk[valid_mask])
                dist_y = dy * (yy[valid_mask] - y_chunk[valid_mask])
                dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
                dist = np.where(dist > 1e-10, dist, 1e-10)
                
                angles = np.arctan2(dz, dist)
                
                # 最大角度更新
                update_indices = np.where(valid_mask)[0]
                angle_update_mask = angles > max_angles[update_indices]
                max_angles[update_indices[angle_update_mask]] = angles[angle_update_mask]
            
            svf_chunk += np.cos(max_angles) ** 2
            work_done += 1
            
            # 進捗更新
            if progress:
                azimuth_deg = int(math.degrees(d * 2 * math.pi / n_dir))
                progress_text = (f"{progress_prefix}チャンク {chunk_number}/{total_chunks}, "
                               f"方向 {azimuth_deg}° ({d+1}/{n_dir})")
                progress.advance(1, progress_text)
        
        # 結果を格納
        result[y_chunk, x_chunk] = svf_chunk / n_dir
    
    return result


# -----------------------------------------------------------------------------
# Public function
# -----------------------------------------------------------------------------

def skyview_factor(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    tile_size: int | None = None,
    memory_efficient: bool = False,
    progress: ProgressReporter | None = None,
) -> NDArray[np.float32]:
    """Compute sky‑view factor from a DEM.

    Parameters
    ----------
    dem : ndarray (H, W)
        Elevation raster (float32 preferred).
    cellsize : float or (float, float), default 1.0
        Pixel size in map units (e.g. metres).  If tuple, interpreted as
        ``(dy, dx)``.
    max_radius : float, default 100.0
        Scan distance in the same units as *cellsize*.
    n_directions : int, default 16
        Number of azimuth sectors (uniformly spaced from 0–360°).
    tile_size : int or None, default None
        Process DEM in ``tile_size × tile_size`` blocks to conserve RAM.
        ``None`` means whole DEM at once.
    memory_efficient : bool, default False
        Use chunk-based processing to reduce memory usage at the cost of
        some performance. Recommended for very large DEMs or limited RAM.
    progress : ProgressReporter or None, default None
        Progress reporter for long-running operations.

    Returns
    -------
    svf : ndarray (H, W)
        Sky‑view factor in [0, 1].  NaN pixels in input are preserved.
    """

    if dem.ndim != 2:
        raise ValueError("DEM must be 2‑D array")
    
    progress = progress or NullProgress()

    dem = dem.astype(np.float32, copy=False)
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize

    # Step count in pixels
    max_dist_px = int(max_radius / max(dx, dy))
    if max_dist_px < 1:
        raise ValueError("max_radius too small relative to cellsize")

    # 大きなスキャン距離に対する警告
    if max_dist_px > min(dem.shape) // 2:
        import warnings
        warnings.warn(
            f"max_radius ({max_radius}) may be too large for DEM size "
            f"{dem.shape}. Consider reducing max_radius or using tiled processing.",
            RuntimeWarning
        )

    az = np.linspace(0.0, 2 * math.pi, n_directions, endpoint=False, dtype=np.float32)
    sin_az = np.sin(az)
    cos_az = np.cos(az)

    nan_mask = np.isnan(dem)

    # スキャン関数の選択
    scan_func = _horizon_scan_chunked if memory_efficient else _horizon_scan_vectorized

    if tile_size is None:
        # 単一タイル（＝DEM 全体）として扱う
        h, w = dem.shape
        progress.set_range(1)  # 初期化
        progress.advance(text=f"SVF計算開始 - DEM: {h}×{w}, 方向数: {n_directions}, 最大距離: {max_radius}")
        
        svf = scan_func(dem, max_dist_px, dx, dy, sin_az, cos_az, progress=progress)
        
        progress.set_range(1)  # 完了用にリセット
        progress.advance(text="SVF計算完了")
    else:
        h, w = dem.shape
        svf = np.empty_like(dem)
        n_tiles_y = math.ceil(h / tile_size)
        n_tiles_x = math.ceil(w / tile_size)
        total_tiles = n_tiles_y * n_tiles_x
        
        progress.set_range(total_tiles)
        progress.advance(text=f"タイル処理開始 - {total_tiles}タイル ({n_tiles_y}×{n_tiles_x})")
        
        tile_count = 0
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                y0 = ty * tile_size
                x0 = tx * tile_size
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                
                tile = dem[y0:y1, x0:x1]
                tile_h, tile_w = tile.shape
                
                # 各タイル用の子プログレス（実際の詳細進捗は内部関数で管理）
                progress_prefix = f"タイル[{ty+1},{tx+1}] ({tile_h}×{tile_w}) - "
                
                svf_tile = scan_func(
                    tile, max_dist_px, dx, dy, sin_az, cos_az, 
                    progress=None,  # タイル処理時は内部進捗を無効化
                    progress_prefix=progress_prefix
                )
                
                svf[y0:y1, x0:x1] = svf_tile
                tile_count += 1
                
                progress.advance(
                    text=f"タイル {tile_count}/{total_tiles} 完了 - 位置[{ty+1},{tx+1}]"
                )
    
    progress.done()

    # 結果の後処理
    svf = np.clip(svf, 0.0, 1.0)
    if nan_mask.any():
        svf[nan_mask] = np.nan
    
    return svf