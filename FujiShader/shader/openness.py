"""
FujiShader.shader.openness
==========================

Positive & negative openness after Yokoyama *et al.* (2002)
-----------------------------------------------------------
*Positive openness* measures how wide the terrain opens upward; *negative
openness* measures how enclosed it is downward.  Both are **azimuth‑independent**
land‑surface parameters valuable for archaeological prospection and general
micro‑relief visualisation.

Definitions (simplified)
~~~~~~~~~~~~~~~~~~~~~~~~
For each azimuth *i* (0…n‑1) find the **maximum upward horizon angle**
``θᵢ = max atan2( z(p) − z₀, d(p, p₀) )`` within a search radius.
Then

    PositiveOpenness = mean( 90° − θᵢ )    [deg]

Similarly, using the **minimum downward angle** yields *negative openness*.
We normalise by 90° to return a **0–1 float** (1 = totally open, 0 = closed).

This module is optimized for QGIS environment without Numba dependency.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import math

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["positive_openness", "negative_openness"]

# -----------------------------------------------------------------------------
# Optimized horizon scan using vectorized NumPy operations
# -----------------------------------------------------------------------------

def _preprocess_dem(
    dem: NDArray[np.float32],
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    progress: Optional[ProgressReporter] = None,
) -> NDArray[np.float32]:
    """
    DEMの前処理：set_nanとreplace_nanオプションを適用
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        入力DEM
    set_nan : Optional[float]
        この値をNaNに設定
    replace_nan : Optional[float] 
        NaN値をこの値で置換
    progress : Optional[ProgressReporter]
        進捗レポーター
        
    Returns
    -------
    NDArray[np.float32]
        前処理されたDEM
    """
    if progress is None:
        progress = NullProgress()
    
    # コピーを作成（元のデータを変更しない）
    processed_dem = dem.copy()
    
    # set_nan: 特定の値をNaNに設定
    if set_nan is not None:
        mask = np.isclose(processed_dem, set_nan, equal_nan=False)
        if np.any(mask):
            nan_count = np.sum(mask)
            processed_dem[mask] = np.nan
            progress.advance(0, f"set_nan: {nan_count:,}個のピクセルを値{set_nan}からNaNに変換")
    
    # replace_nan: NaN値を特定の値で置換
    if replace_nan is not None:
        nan_mask = np.isnan(processed_dem)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            processed_dem[nan_mask] = replace_nan
            progress.advance(0, f"replace_nan: {nan_count:,}個のNaNピクセルを値{replace_nan}で置換")
    
    return processed_dem


def _scan_horizon_angles_vectorized(
    dem: NDArray[np.float32],
    max_steps: int,
    dx: float,
    dy: float,
    sin_az: NDArray[np.float32],
    cos_az: NDArray[np.float32],
    progress: ProgressReporter,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """ベクトル化された地平角計算（QGIS/NumPy最適化版）"""
    h, w = dem.shape
    n_dir = sin_az.size
    
    # プログレス範囲を設定（方向数 × ステップ数で総処理量を推定）
    total_operations = n_dir * max_steps
    progress.set_range(total_operations)
    current_op = 0
    
    # 出力配列を初期化
    ang_max = np.full((h, w), -np.pi/2, dtype=np.float32)
    ang_min = np.full((h, w), np.pi/2, dtype=np.float32)
    
    # 座標グリッドを事前計算
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    
    progress.advance(0, f"ベクトル化処理開始 (DEM: {h}×{w}, 方向数: {n_dir}, 最大ステップ: {max_steps})")
    
    # 各方向を処理
    for d_idx, (sin_d, cos_d) in enumerate(zip(sin_az, cos_az)):
        dir_max = np.full((h, w), -np.pi/2, dtype=np.float32)
        dir_min = np.full((h, w), np.pi/2, dtype=np.float32)
        
        # ベクトル化されたステップ計算
        for step in range(1, max_steps + 1):
            # サンプル座標を計算
            sample_x = x_grid + cos_d * step
            sample_y = y_grid + sin_d * step
            
            # 境界チェック
            valid_mask = (
                (sample_x >= 0) & (sample_x < w - 1) &
                (sample_y >= 0) & (sample_y < h - 1)
            )
            
            if not np.any(valid_mask):
                # 有効なサンプルがない場合は残りのステップをスキップ
                current_op += (max_steps - step + 1)
                progress.advance(max_steps - step + 1, 
                               f"方向 {d_idx + 1}/{n_dir} - 境界到達によりスキップ")
                break
                
            # バイリニア補間のインデックス
            ix = sample_x.astype(np.int32)
            iy = sample_y.astype(np.int32)
            fx = sample_x - ix
            fy = sample_y - iy
            
            # ベクトル化されたバイリニア補間
            z_interp = np.zeros_like(dem)
            mask_indices = np.where(valid_mask)
            
            if len(mask_indices[0]) > 0:
                iy_valid = iy[mask_indices]
                ix_valid = ix[mask_indices]
                fx_valid = fx[mask_indices]
                fy_valid = fy[mask_indices]
                
                z_interp[mask_indices] = (
                    dem[iy_valid, ix_valid] * (1 - fx_valid) * (1 - fy_valid) +
                    dem[iy_valid, ix_valid + 1] * fx_valid * (1 - fy_valid) +
                    dem[iy_valid + 1, ix_valid] * (1 - fx_valid) * fy_valid +
                    dem[iy_valid + 1, ix_valid + 1] * fx_valid * fy_valid
                )
            
            # 角度を計算
            dz = z_interp - dem
            dist = np.sqrt(
                (dx * (sample_x - x_grid))**2 + 
                (dy * (sample_y - y_grid))**2
            )
            
            # ゼロ除算を避ける
            dist = np.maximum(dist, 1e-10)
            angles = np.arctan2(dz, dist)
            
            # NaN値を含むピクセルを除外
            nan_mask = np.isnan(dem) | np.isnan(z_interp)
            angles[nan_mask] = np.nan
            
            # 有効なセルのみで方向別極値を更新
            valid_update_mask = valid_mask & ~nan_mask
            dir_max = np.where(valid_update_mask & (angles > dir_max), angles, dir_max)
            dir_min = np.where(valid_update_mask & (angles < dir_min), angles, dir_min)
            
            # プログレス更新
            current_op += 1
            if step % 10 == 0 or step == max_steps:  # 10ステップごと、または最終ステップで更新
                valid_count = np.sum(valid_mask)
                progress.advance(10 if step % 10 == 0 else step % 10, 
                               f"方向 {d_idx + 1}/{n_dir} - ステップ {step}/{max_steps} (有効セル: {valid_count:,})")
        
        # グローバル極値を更新
        ang_max = np.maximum(ang_max, dir_max)
        ang_min = np.minimum(ang_min, dir_min)
        
        # 方向完了の報告
        progress.advance(0, f"方向 {d_idx + 1}/{n_dir} 完了")
    
    return ang_max, ang_min


def _scan_horizon_angles_chunked(
    dem: NDArray[np.float32],
    max_steps: int,
    dx: float,
    dy: float,
    sin_az: NDArray[np.float32],
    cos_az: NDArray[np.float32],
    progress: ProgressReporter,
    chunk_size: int = 512,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """大きなDEM用のメモリ効率的なチャンク処理"""
    h, w = dem.shape
    
    # 出力配列を初期化
    ang_max = np.full((h, w), -np.pi/2, dtype=np.float32)
    ang_min = np.full((h, w), np.pi/2, dtype=np.float32)
    
    # チャンクを計算
    n_chunks_y = (h + chunk_size - 1) // chunk_size
    n_chunks_x = (w + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_y * n_chunks_x
    
    # プログレス範囲をチャンク数で設定
    progress.set_range(total_chunks)
    
    progress.advance(0, f"チャンク処理開始 (DEM: {h}×{w}, チャンク: {n_chunks_y}×{n_chunks_x}={total_chunks}, サイズ: {chunk_size})")
    
    for chunk_y in range(n_chunks_y):
        for chunk_x in range(n_chunks_x):
            # チャンクの境界を定義
            y_start = chunk_y * chunk_size
            y_end = min((chunk_y + 1) * chunk_size, h)
            x_start = chunk_x * chunk_size
            x_end = min((chunk_x + 1) * chunk_size, w)
            
            # 補間用のバッファを持つチャンクを抽出
            buffer = max_steps + 1
            y_buf_start = max(0, y_start - buffer)
            y_buf_end = min(h, y_end + buffer)
            x_buf_start = max(0, x_start - buffer)
            x_buf_end = min(w, x_end + buffer)
            
            dem_chunk = dem[y_buf_start:y_buf_end, x_buf_start:x_buf_end]
            
            chunk_num = chunk_y * n_chunks_x + chunk_x + 1
            chunk_h = y_end - y_start
            chunk_w = x_end - x_start
            
            progress.advance(0, f"チャンク {chunk_num}/{total_chunks} 処理中 ({chunk_h}×{chunk_w})")
            
            # チャンクを処理
            chunk_max, chunk_min = _process_chunk(
                dem_chunk, dem, max_steps, dx, dy, sin_az, cos_az,
                y_start, y_end, x_start, x_end,
                y_buf_start, x_buf_start
            )
            
            # 結果を保存
            ang_max[y_start:y_end, x_start:x_end] = chunk_max
            ang_min[y_start:y_end, x_start:x_end] = chunk_min
            
            # プログレス更新
            progress.advance(1, f"チャンク {chunk_num}/{total_chunks} 完了 ({100.0 * chunk_num / total_chunks:.1f}%)")
    
    return ang_max, ang_min


def _process_chunk(
    dem_chunk: NDArray[np.float32],
    dem_full: NDArray[np.float32],  
    max_steps: int,
    dx: float,
    dy: float,
    sin_az: NDArray[np.float32],
    cos_az: NDArray[np.float32],
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    y_buf_start: int,
    x_buf_start: int,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """DEMの単一チャンクを処理"""
    chunk_h = y_end - y_start
    chunk_w = x_end - x_start
    full_h, full_w = dem_full.shape
    
    ang_max = np.full((chunk_h, chunk_w), -np.pi/2, dtype=np.float32)
    ang_min = np.full((chunk_h, chunk_w), np.pi/2, dtype=np.float32)
    
    # チャンク用の座標グリッドを作成
    y_chunk_grid, x_chunk_grid = np.mgrid[0:chunk_h, 0:chunk_w].astype(np.float32)
    y_global_grid = y_chunk_grid + y_start
    x_global_grid = x_chunk_grid + x_start
    
    # チャンク中心の標高値を取得
    z0 = dem_full[y_start:y_end, x_start:x_end]
    
    n_dir = len(sin_az)
    
    for dir_idx, (sin_d, cos_d) in enumerate(zip(sin_az, cos_az)):
        dir_max = np.full((chunk_h, chunk_w), -np.pi/2, dtype=np.float32)
        dir_min = np.full((chunk_h, chunk_w), np.pi/2, dtype=np.float32)
        
        for step in range(1, max_steps + 1):
            # グローバル空間でのサンプル座標を計算
            sample_x = x_global_grid + cos_d * step
            sample_y = y_global_grid + sin_d * step
            
            # グローバル座標での境界チェック
            valid_mask = (
                (sample_x >= 0) & (sample_x < full_w - 1) &
                (sample_y >= 0) & (sample_y < full_h - 1)
            )
            
            if not np.any(valid_mask):
                continue
            
            # 補間された標高を取得
            z_interp = _bilinear_interpolate(
                dem_full, sample_x, sample_y, valid_mask
            )
            
            # 角度を計算
            dz = z_interp - z0
            dist = np.sqrt(
                (dx * cos_d * step)**2 + (dy * sin_d * step)**2
            )
            angles = np.arctan2(dz, dist)
            
            # NaN値を処理
            nan_mask = np.isnan(z0) | np.isnan(z_interp)
            angles[nan_mask] = np.nan
            
            # 極値を更新
            valid_update_mask = valid_mask & ~nan_mask
            dir_max = np.where(valid_update_mask & (angles > dir_max), angles, dir_max)
            dir_min = np.where(valid_update_mask & (angles < dir_min), angles, dir_min)
        
        ang_max = np.maximum(ang_max, dir_max)
        ang_min = np.minimum(ang_min, dir_min)
    
    return ang_max, ang_min


def _bilinear_interpolate(
    dem: NDArray[np.float32],
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """効率的なバイリニア補間"""
    result = np.zeros_like(x)
    
    if not np.any(valid_mask):
        return result
    
    # 有効な座標を取得
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # 整数部と小数部
    ix = x_valid.astype(np.int32)
    iy = y_valid.astype(np.int32)
    fx = x_valid - ix
    fy = y_valid - iy
    
    # バイリニア補間
    interpolated = (
        dem[iy, ix] * (1 - fx) * (1 - fy) +
        dem[iy, ix + 1] * fx * (1 - fy) +
        dem[iy + 1, ix] * (1 - fx) * fy +
        dem[iy + 1, ix + 1] * fx * fy
    )
    
    result[valid_mask] = interpolated
    return result


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def _openness_impl(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    positive: bool = True,
    progress: Optional[ProgressReporter] = None,
    use_chunked: bool = False,
    chunk_size: int = 512,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
) -> NDArray[np.float32]:
    """正/負のオープンネスの共通計算コア"""
    if dem.ndim != 2:
        raise ValueError("DEMは2次元である必要があります")

    progress = progress or NullProgress()
    
    # DEM前処理（set_nanとreplace_nanオプション）
    processed_dem = _preprocess_dem(dem, set_nan, replace_nan, progress)
    
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    max_dist_px = int(max_radius / max(dx, dy))
    if max_dist_px < 1:
        raise ValueError("max_radiusがcellsizeに対して小さすぎます")

    # 方位角方向を生成
    az = np.linspace(0.0, 2 * math.pi, n_directions, endpoint=False, dtype=np.float32)
    sin_az = np.sin(az)
    cos_az = np.cos(az)

    # DEMサイズに基づいて処理方法を選択
    dem_32 = processed_dem.astype(np.float32, copy=False)
    total_cells = dem_32.size
    h, w = dem_32.shape
    
    # 処理方法の決定とログ出力
    processing_method = "チャンク処理" if (use_chunked or total_cells > 1_000_000) else "ベクトル処理"
    openness_type = "正のオープンネス" if positive else "負のオープンネス"
    
    progress.advance(0, f"{openness_type}計算開始 ({processing_method})")
    progress.advance(0, f"パラメータ: DEM({h}×{w}), セルサイズ({dx:.2f}×{dy:.2f}), 最大半径({max_radius:.1f}), 方向数({n_directions})")
    
    if use_chunked or total_cells > 1_000_000:  # 大きなDEMの場合はチャンクを使用
        progress.advance(0, f"大規模DEM検出 (セル数: {total_cells:,}) - チャンク処理を実行")
        ang_max, ang_min = _scan_horizon_angles_chunked(
            dem_32, max_dist_px, dx, dy, sin_az, cos_az, progress, chunk_size
        )
    else:
        progress.advance(0, f"中規模DEM (セル数: {total_cells:,}) - ベクトル処理を実行")
        ang_max, ang_min = _scan_horizon_angles_vectorized(
            dem_32, max_dist_px, dx, dy, sin_az, cos_az, progress
        )

    progress.advance(0, "地平角スキャン完了 - オープンネス値を計算中...")

    # オープンネスを計算
    if positive:
        openness = (math.pi / 2 - ang_max)  # ラジアン
    else:
        openness = (ang_min + math.pi / 2)

    # 0-1範囲に正規化
    openness = np.clip(openness / (math.pi / 2), 0.0, 1.0)
    
    # 元のNaN値を保持
    nan_mask = np.isnan(dem)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        openness[nan_mask] = np.nan
        progress.advance(0, f"NaN値を保持: {nan_count:,}セル")
    
    # 統計情報の計算と表示
    valid_openness = openness[~nan_mask] if nan_count > 0 else openness
    if valid_openness.size > 0:
        min_val = np.min(valid_openness)
        max_val = np.max(valid_openness)
        mean_val = np.mean(valid_openness)
        progress.advance(0, f"{openness_type}完了 - 値範囲: [{min_val:.3f}, {max_val:.3f}], 平均: {mean_val:.3f}")
    
    progress.done()
    return openness.astype(np.float32, copy=False)


def positive_openness(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    progress: Optional[ProgressReporter] = None,
    use_chunked: bool = False,
    chunk_size: int = 512,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
) -> NDArray[np.float32]:
    """
    正のオープンネス (0-1)。1 = 完全に上方に開いている。
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        2次元配列としてのデジタル標高モデル
    cellsize : Union[Tuple[float, float], float], default 1.0
        (dy, dx)形式のセルサイズ、または正方形セル用の単一値
    max_radius : float, default 100.0
        地上単位での最大探索半径
    n_directions : int, default 16
        サンプリングする方位角方向の数
    progress : Optional[ProgressReporter], default None
        長時間実行操作用の進捗レポーター
    use_chunked : bool, default False
        メモリ効率のためのチャンク処理を強制
    chunk_size : int, default 512
        メモリ効率的処理用のチャンクサイズ
    set_nan : Optional[float], default None
        この値をNaNに設定
    replace_nan : Optional[float], default None
        NaN値をこの値で置換
        
    Returns
    -------
    NDArray[np.float32]
        0-1範囲に正規化された正のオープンネス値
    """
    return _openness_impl(
        dem,
        cellsize=cellsize,
        max_radius=max_radius,
        n_directions=n_directions,
        positive=True,
        progress=progress,
        use_chunked=use_chunked,
        chunk_size=chunk_size,
        set_nan=set_nan,
        replace_nan=replace_nan,
    )


def negative_openness(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    progress: Optional[ProgressReporter] = None,
    use_chunked: bool = False,
    chunk_size: int = 512,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
) -> NDArray[np.float32]:
    """
    負のオープンネス (0-1)。1 = 下方に高度に囲まれている。
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        2次元配列としてのデジタル標高モデル
    cellsize : Union[Tuple[float, float], float], default 1.0
        (dy, dx)形式のセルサイズ、または正方形セル用の単一値
    max_radius : float, default 100.0
        地上単位での最大探索半径
    n_directions : int, default 16
        サンプリングする方位角方向の数
    progress : Optional[ProgressReporter] = None
        長時間実行操作用の進捗レポーター
    use_chunked : bool, default False
        メモリ効率のためのチャンク処理を強制
    chunk_size : int, default 512
        メモリ効率的処理用のチャンクサイズ
    set_nan : Optional[float], default None
        この値をNaNに設定
    replace_nan : Optional[float], default None
        NaN値をこの値で置換
        
    Returns
    -------
    NDArray[np.float32]
        0-1範囲に正規化された負のオープンネス値
    """
    return _openness_impl(
        dem,
        cellsize=cellsize,
        max_radius=max_radius,
        n_directions=n_directions,
        positive=False,
        progress=progress,
        use_chunked=use_chunked,
        chunk_size=chunk_size,
        set_nan=set_nan,
        replace_nan=replace_nan,
    )