# fuji_shader/skyview_fast.py

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import math

import numpy as np

# NumPy型ヒントの互換性対応
try:
    from numpy.typing import NDArray
except ImportError:
    if TYPE_CHECKING:
        from numpy import ndarray as NDArray
    else:
        NDArray = "np.ndarray"


__all__ = ["skyview_factor_fast"]


def _bilinear_fast(dem: NDArray[np.float32],
                   yy: NDArray[np.float32],
                   xx: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    高速化されたバイリニア補間
    dem    : (H, W)
    yy/xx  : (..., n_samples) の形、浮動小数で画素座標を表す
    戻り値 : yy, xx と同じ shape の標高値（float32）
    """
    h, w = dem.shape
    
    # 境界処理を改善（浮動小数点精度を考慮）
    xx_clp = np.clip(xx, 0, w - 1.001)
    yy_clp = np.clip(yy, 0, h - 1.001)

    # int32変換を一度だけ実行
    ix = xx_clp.astype(np.int32)
    iy = yy_clp.astype(np.int32)
    
    # 境界チェック（安全性向上）
    ix = np.clip(ix, 0, w - 2)
    iy = np.clip(iy, 0, h - 2)
    
    fx = xx_clp - ix
    fy = yy_clp - iy

    # インデックスアクセスを最適化
    z00 = dem[iy, ix]
    z10 = dem[iy, ix + 1]
    z01 = dem[iy + 1, ix]
    z11 = dem[iy + 1, ix + 1]

    # バイリニア補間計算（型キャストを最後に一度だけ）
    return (z00 * (1 - fx) * (1 - fy) + 
            z10 * fx * (1 - fy) + 
            z01 * (1 - fx) * fy + 
            z11 * fx * fy).astype(np.float32)


def _precompute_direction_vectors(n_directions: int) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """方位角ベクトルの事前計算"""
    az = np.linspace(0, 2 * math.pi, n_directions, endpoint=False, dtype=np.float32)
    return np.sin(az), np.cos(az)


def _robust_nan_fill(dem: NDArray[np.float32], nan_mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """堅牢なNaN値の置き換え"""
    if not nan_mask.any():
        return dem
    
    dem_copy = dem.copy()
    finite_vals = dem[~nan_mask]
    
    if len(finite_vals) > 0:
        min_val = np.min(finite_vals)
        val_range = np.ptp(finite_vals)
        fill_val = min_val - max(100.0, val_range)
        dem_copy[nan_mask] = fill_val
    else:
        # 全てNaNの場合のフォールバック
        dem_copy[nan_mask] = -1000.0
    
    return dem_copy


def skyview_factor_fast(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    n_samples: int = 8,
    tile_size: int | None = None,
    progress=None,
) -> NDArray[np.float32]:
    """
    Sky-View Factor の高速近似 (疎サンプリング版)。

    Parameters
    ----------
    dem          : ndarray (H, W) の float32 標高ラスタ。NaN を含んでよい。
    cellsize     : float または (dy, dx)。地図単位のピクセルサイズ。デフォルト 1.0。
    max_radius   : float。スキャン半径（地図単位）。デフォルト 100.0。
    n_directions : int。方位分割数。必ず >=1。デフォルト 16。
    n_samples    : int。各方位につきサンプリングする半径ステップ数 (>=1)。デフォルト 8。
    tile_size    : int または None。タイル分割するときのブロックサイズ。None なら全画素一括計算。
    progress     : FujiShader 用プログレスレポーターオブジェクト（任意）。タイル処理時に使う。

    Returns
    -------
    svf_fast : ndarray (H, W) の float32。Sky-View Factor 近似値 [0,1]。入力 DEM の NaN は NaN のまま返す。
    """

    # ------------------------------
    # 1) 入力チェック / 最適化
    # ------------------------------
    if dem.ndim != 2:
        raise ValueError("DEM must be 2-D array")

    if n_directions < 1:
        raise ValueError("n_directions must be >= 1")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    # 型確保（コピー回避）
    if dem.dtype != np.float32:
        dem = dem.astype(np.float32)

    # NaN マスクを事前計算
    nan_mask = np.isnan(dem)
    if nan_mask.any():
        dem = _robust_nan_fill(dem, nan_mask)

    # ------------------------------
    # 2) パラメータの事前計算
    # ------------------------------
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    max_r_px = max_radius / max(dx, dy)

    # サンプリング半径（線形分布）
    r_px = np.linspace(1.0, max_r_px, n_samples, dtype=np.float32)
    r_m = r_px * max(dx, dy)

    # 方位角ベクトルの事前計算
    sin_az, cos_az = _precompute_direction_vectors(n_directions)

    h, w = dem.shape

    # ------------------------------
    # 3) タイル分割処理
    # ------------------------------
    if tile_size is not None:
        out = np.empty((h, w), dtype=np.float32)

        # タイル数の計算
        n_tiles_y = math.ceil(h / tile_size)
        n_tiles_x = math.ceil(w / tile_size)
        total_tiles = n_tiles_y * n_tiles_x

        # プログレス初期化
        if progress is not None:
            progress.set_range(total_tiles)

        tile_count = 0
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1 = min(y + tile_size, h)
                x1 = min(x + tile_size, w)
                tile = dem[y:y1, x:x1]
                
                # 再帰呼び出し（tile_size=None）
                svf_tile = skyview_factor_fast(
                    tile,
                    cellsize=cellsize,
                    max_radius=max_radius,
                    n_directions=n_directions,
                    n_samples=n_samples,
                    tile_size=None,
                    progress=None,  # タイル内ではプログレス無効
                )
                out[y:y1, x:x1] = svf_tile

                # プログレス更新
                tile_count += 1
                if progress is not None:
                    progress.advance(1, f"Processing tile {tile_count}/{total_tiles}")

        # 処理完了をマーク
        if progress is not None:
            progress.done()

        # NaN復元
        if nan_mask.any():
            out[nan_mask] = np.nan
        return out

    # ------------------------------
    # 4) メイン計算：全画素一括処理
    # ------------------------------
    # プログレス初期化（方位角の数で進捗管理）
    if progress is not None:
        # 初期化フェーズ + 方位角処理 + 最終化フェーズ
        total_steps = 1 + n_directions + 1
        progress.set_range(total_steps)
        progress.advance(1, "Initializing Sky-View Factor calculation...")

    # 座標グリッドの事前計算
    y0, x0 = np.mgrid[0:h, 0:w]
    
    # SVF累積配列
    sv_sum = np.zeros((h, w), dtype=np.float32)
    
    # 定数配列の事前計算（ブロードキャスト用）
    r_m_bc = r_m[None, None, :]  # (1, 1, n_samples)

    # 各方位角の処理
    for d in range(n_directions):
        # サンプリング座標の計算
        cos_d, sin_d = cos_az[d], sin_az[d]
        xx = x0[..., None] + cos_d * r_px  # (H, W, n_samples)
        yy = y0[..., None] + sin_d * r_px
        
        # 有効範囲マスク
        mask_valid = ((xx >= 0) & (xx < w) & (yy >= 0) & (yy < h))
        
        # バイリニア補間
        z_samp = _bilinear_fast(dem, yy, xx)
        
        # 高さ差と角度計算
        dz = z_samp - dem[..., None]
        
        # 数値安定性のためのクリッピング
        dz = np.clip(dz, -1e6, 1e6)
        
        # 角度計算
        ang = np.arctan2(dz, r_m_bc)
        
        # 無効サンプルを除外
        ang = np.where(mask_valid, ang, -np.inf)
        
        # 方向ごとの最大角度
        max_ang = np.max(ang, axis=-1)
        
        # 有効サンプルがない場合は角度0（完全開放）
        has_valid = np.any(mask_valid, axis=-1)
        max_ang = np.where(has_valid, max_ang, 0.0)
        
        # cos^2 で累積（ベクトル化）
        sv_sum += np.square(np.cos(max_ang))

        # プログレス更新
        if progress is not None:
            progress.advance(1, f"Processing direction {d+1}/{n_directions} (azimuth: {(d * 360 / n_directions):.1f}°)")

    # ------------------------------
    # 5) 結果の正規化とクリッピング
    # ------------------------------
    if progress is not None:
        progress.advance(1, "Finalizing Sky-View Factor calculation...")

    svf = sv_sum * (1.0 / n_directions)  # 乗算の方が除算より高速
    svf = np.clip(svf, 0.0, 1.0)
    
    # NaN復元
    if nan_mask.any():
        svf[nan_mask] = np.nan

    # 処理完了
    if progress is not None:
        progress.done()

    return svf.astype(np.float32)