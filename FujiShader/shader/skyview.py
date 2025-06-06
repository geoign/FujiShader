"""
FujiShader.shader.skyview
=========================

Sky‑View Factor (SVF) 計算 - 純粋Python/NumPy実装でQGIS環境に最適化
SVF は各DEMセルから見える上半球の割合を表現します (0 = 完全に囲まれている, 1 = 完全な天空)
方位角に依存しない ambient occlusion の代替手法、冷気滞留解析、暖寒色マッピングなどに広く使用されます。

古典的な定義は以下の通りです：

    SVF = 1⁄n Σ cos² θᵢ            (Oke 1988)

ここで *θᵢ* は方向 *i* における地平線への仰角です。
均等に配置された方位角セット（デフォルト16方向）で積分を近似します。
地平線角度は、ユーザー定義の ``max_radius`` （ピクセルまたはメートル）まで
放射状にスキャンして求めます。

主な特徴
--------
* **純粋NumPy** – 外部JITコンパイラなしでQGIS Python環境に最適化
* **ベクトル化演算** – NumPyのブロードキャストと高度なインデクシングによる効率的バッチ処理
* **メートル単位対応** – 再投影後も物理的に意味のあるスキャン距離を維持
* **タイル処理** – どんなに大きなDEMでもシームレスにタイル結合して処理
* **NaN値完全対応** – メタデータとマニュアル指定のNaN値を適切に処理
"""
from __future__ import annotations

from ..core.progress import ProgressReporter, NullProgress
from typing import Optional, Tuple, Union

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["skyview_factor"]

# -----------------------------------------------------------------------------
# NaN値処理ユーティリティ
# -----------------------------------------------------------------------------

def _prepare_dem_with_nan_handling(
    dem: NDArray[np.float32],
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    DEMのNaN値処理を行い、処理用の配列とマスクを返す
    
    Parameters
    ----------
    dem : ndarray
        入力DEM
    set_nan : float, optional
        この値をNaNに設定
    replace_nan : float, optional
        NaN値をこの値で置換
        
    Returns
    -------
    processed_dem : ndarray
        処理済みDEM
    original_nan_mask : ndarray
        元のNaN位置のマスク（結果にNaNを復元するため）
    """
    processed_dem = dem.copy().astype(np.float32)
    
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(processed_dem)
    
    # 指定値をNaNに設定
    if set_nan is not None:
        set_mask = np.isclose(processed_dem, set_nan, equal_nan=False)
        processed_dem[set_mask] = np.nan
        original_nan_mask |= set_mask
    
    # NaN値を指定値で置換（計算用）
    if replace_nan is not None:
        nan_mask = np.isnan(processed_dem)
        processed_dem[nan_mask] = replace_nan
    
    return processed_dem, original_nan_mask


# -----------------------------------------------------------------------------
# _horizon_scan: NumPy ベクトル化コア
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
    """ベクトル化された地平線スキャンを使用してタイルのSVFを返す"""
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
            
            # NaN値のチェックを追加した双線形補間
            z_00 = dem[iy[mask], ix[mask]]
            z_01 = dem[iy[mask], ix[mask] + 1]
            z_10 = dem[iy[mask] + 1, ix[mask]]
            z_11 = dem[iy[mask] + 1, ix[mask] + 1]
            
            # いずれかがNaNの場合は補間結果もNaN
            interp_valid = ~(np.isnan(z_00) | np.isnan(z_01) | np.isnan(z_10) | np.isnan(z_11))
            
            z[mask] = np.where(
                interp_valid,
                (z_00 * (1 - fx[mask]) * (1 - fy[mask]) +
                 z_01 * fx[mask] * (1 - fy[mask]) +
                 z_10 * (1 - fx[mask]) * fy[mask] +
                 z_11 * fx[mask] * fy[mask]),
                np.nan
            )
            
            # 距離と角度計算
            dz = z - dem
            dist_x = dx * (xx - x_idx)
            dist_y = dy * (yy - y_idx)
            dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
            
            # ゼロ除算対策とNaN処理
            dist = np.where(dist > 1e-10, dist, 1e-10)
            angles = np.arctan2(dz, dist)
            
            # NaN値を含む計算では適切にNaNを伝播
            angles = np.where(np.isnan(dz), -1e9, angles)
            
            # 最大角度更新（有効な点のみ）
            update_mask = mask & (angles > max_angles) & ~np.isnan(angles)
            max_angles[update_mask] = angles[update_mask]
        
        # この方向のSVF寄与を加算
        # NaNが含まれる場合は適切に処理
        direction_contribution = np.cos(max_angles) ** 2
        direction_contribution = np.where(np.isfinite(direction_contribution), direction_contribution, 0.0)
        svf_sum += direction_contribution
        
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
    """メモリ効率版：ピクセルをチャンクで処理"""
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
                
                # 双線形補間（NaN対応）
                ix = xx[valid_mask].astype(np.int32)
                iy = yy[valid_mask].astype(np.int32)
                fx = xx[valid_mask] - ix
                fy = yy[valid_mask] - iy
                
                # 補間に使用する4点の値を取得
                z_00 = dem[iy, ix]
                z_01 = dem[iy, ix + 1]
                z_10 = dem[iy + 1, ix]
                z_11 = dem[iy + 1, ix + 1]
                
                # いずれかがNaNの場合は補間結果もNaN
                interp_valid = ~(np.isnan(z_00) | np.isnan(z_01) | np.isnan(z_10) | np.isnan(z_11))
                
                z_interp = np.where(
                    interp_valid,
                    (z_00 * (1 - fx) * (1 - fy) +
                     z_01 * fx * (1 - fy) +
                     z_10 * (1 - fx) * fy +
                     z_11 * fx * fy),
                    np.nan
                )
                
                # 角度計算
                z0_chunk = dem[y_chunk[valid_mask], x_chunk[valid_mask]]
                dz = z_interp - z0_chunk
                
                dist_x = dx * (xx[valid_mask] - x_chunk[valid_mask])
                dist_y = dy * (yy[valid_mask] - y_chunk[valid_mask])
                dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
                dist = np.where(dist > 1e-10, dist, 1e-10)
                
                angles = np.arctan2(dz, dist)
                
                # NaN値の処理
                angles = np.where(np.isnan(dz), -1e9, angles)
                
                # 最大角度更新
                update_indices = np.where(valid_mask)[0]
                angle_update_mask = (angles > max_angles[update_indices]) & np.isfinite(angles)
                max_angles[update_indices[angle_update_mask]] = angles[angle_update_mask]
            
            # この方向のSVF寄与を計算
            direction_contribution = np.cos(max_angles) ** 2
            direction_contribution = np.where(np.isfinite(direction_contribution), direction_contribution, 0.0)
            svf_chunk += direction_contribution
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
# 公開関数
# -----------------------------------------------------------------------------

def skyview_factor(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    tile_size: int | None = None,
    memory_efficient: bool = False,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    progress: ProgressReporter | None = None,
) -> NDArray[np.float32]:
    """DEM から Sky‑View Factor を計算します。

    Parameters
    ----------
    dem : ndarray (H, W)
        標高ラスタ（float32推奨）
    cellsize : float or (float, float), default 1.0
        ピクセルサイズ（マップ単位、例：メートル）。タプルの場合は ``(dy, dx)`` として解釈
    max_radius : float, default 100.0
        *cellsize* と同じ単位でのスキャン距離
    n_directions : int, default 16
        方位角セクター数（0–360°で均等配置）
    tile_size : int or None, default None
        RAM節約のために DEM を ``tile_size × tile_size`` ブロックで処理
        ``None`` の場合は DEM 全体を一度に処理
    memory_efficient : bool, default False
        非常に大きなDEMや限られたRAMに対して、パフォーマンスを犠牲にして
        メモリ使用量を削減するチャンクベース処理を使用
    set_nan : float, optional
        この値をNaNに設定（処理開始時）
    replace_nan : float, optional
        NaN値をこの値で置換（計算用、結果では元のNaN位置を復元）
    progress : ProgressReporter or None, default None
        長時間実行される操作の進捗レポーター

    Returns
    -------
    svf : ndarray (H, W)
        Sky‑View Factor [0, 1]範囲。入力のNaNピクセルは保持されます
    """

    if dem.ndim != 2:
        raise ValueError("DEM は 2次元配列である必要があります")
    
    progress = progress or NullProgress()

    # NaN値処理
    processed_dem, original_nan_mask = _prepare_dem_with_nan_handling(dem, set_nan, replace_nan)
    
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize

    # ピクセル単位でのステップ数
    max_dist_px = int(max_radius / max(dx, dy))
    if max_dist_px < 1:
        raise ValueError("max_radius が cellsize に対して小さすぎます")

    # 大きなスキャン距離に対する警告
    if max_dist_px > min(processed_dem.shape) // 2:
        import warnings
        warnings.warn(
            f"max_radius ({max_radius}) が DEM サイズ {processed_dem.shape} に対して大きすぎる可能性があります。"
            f"max_radius を減らすか、タイル処理の使用を検討してください。",
            RuntimeWarning
        )

    az = np.linspace(0.0, 2 * math.pi, n_directions, endpoint=False, dtype=np.float32)
    sin_az = np.sin(az)
    cos_az = np.cos(az)

    # スキャン関数の選択
    scan_func = _horizon_scan_chunked if memory_efficient else _horizon_scan_vectorized

    if tile_size is None:
        # 単一タイル（＝DEM 全体）として扱う
        h, w = processed_dem.shape
        progress.set_range(1)  # 初期化
        progress.advance(text=f"SVF計算開始 - DEM: {h}×{w}, 方向数: {n_directions}, 最大距離: {max_radius}")
        
        svf = scan_func(processed_dem, max_dist_px, dx, dy, sin_az, cos_az, progress=progress)
        
        progress.set_range(1)  # 完了用にリセット
        progress.advance(text="SVF計算完了")
    else:
        h, w = processed_dem.shape
        svf = np.empty_like(processed_dem)
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
                
                tile = processed_dem[y0:y1, x0:x1]
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
    
    # 元のNaN位置を復元
    if original_nan_mask.any():
        svf[original_nan_mask] = np.nan
    
    return svf