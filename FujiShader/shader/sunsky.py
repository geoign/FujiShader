"""
FujiShader.shader.sunsky
========================

物理ベースの*方向性*照明モデル（QGIS最適化版）
---------------------------------------------
このモジュールは2つの補完的なシェーディング関数を提供します：

* **`direct_light`** – 太陽位置（方位角、高度角）に基づく厳密なランベルト山岳陰影に
  **オプションの落影**を含む
* **`sky_light`** – 離散的な方位角セットを使用した半球天空光積分；
  本質的には*方向重み付きSVF*

両方とも**[0–1]**範囲の*放射照度重み*を返すため、カラーパイプライン
（例：暖色–寒色合成）に供給したり、反射率マップと乗算したりできます。

設計方針
~~~~~~~~
* 他のFujiShaderモジュールと同じ*cellsize / max_radius / n_directions*パラメータ
  → 一貫したUX
* QGIS環境に最適化されたピュアNumPy実装
* 落影ルーチンはより良いパフォーマンスのためにベクトル化演算を使用
* 大容量DEMに適したメモリ効率的な実装
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
# ヘルパー: 勾配 → 単位法線ベクトル（最適化版）
# -----------------------------------------------------------------------------

def _unit_normals(dem: NDArray[np.float32], dy: float, dx: float) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """中央差分による(nx, ny, nz)単位面法線ベクトルを返す。
    
    QGIS環境でのメモリ効率とスピードに最適化済み。
    """
    # エッジ処理改善のためnumpy gradientを使用（edge_order=1）
    dz_dy, dz_dx = np.gradient(dem, dy, dx, edge_order=1)
    
    # 大きさを一度計算して再利用
    mag_sq = dz_dx * dz_dx + dz_dy * dz_dy + 1.0
    nz = np.reciprocal(np.sqrt(mag_sq))  # 1.0 / sqrtより若干高速
    
    # 法線ベクトルを計算
    nx = -dz_dx * nz
    ny = -dz_dy * nz
    
    return nx.astype(np.float32, copy=False), ny.astype(np.float32, copy=False), nz.astype(np.float32, copy=False)

# -----------------------------------------------------------------------------
# NaN値処理ヘルパー
# -----------------------------------------------------------------------------

def _process_nan_values(
    dem: NDArray[np.float32], 
    set_nan: Optional[float] = None, 
    replace_nan: Optional[float] = None
) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    DEM配列のNaN値処理を行う
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        入力DEM配列
    set_nan : float, optional
        この値をNaNに設定する
    replace_nan : float, optional
        NaN値をこの値で一時的に置換する（計算用）
        
    Returns
    -------
    processed_dem : NDArray[np.float32]
        処理済みDEM配列
    original_nan_mask : NDArray[np.bool_]
        元のNaN位置のマスク（結果復元用）
    """
    # float32にコピー
    processed_dem = dem.astype(np.float32, copy=True)
    
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(processed_dem)
    
    # 指定値をNaNに設定
    if set_nan is not None:
        set_nan_mask = np.isclose(processed_dem, set_nan, equal_nan=False)
        if np.any(set_nan_mask):
            processed_dem[set_nan_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | set_nan_mask
    
    # replace_nan処理：NaN値を一時的に置換
    if replace_nan is not None:
        current_nan_mask = np.isnan(processed_dem)
        if np.any(current_nan_mask):
            processed_dem[current_nan_mask] = replace_nan
    
    return processed_dem, original_nan_mask

# -----------------------------------------------------------------------------
# ベクトル化落影計算
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
    メモリ効率のためのチャンク処理を使用したベクトル化影計算
    
    このアプローチはメモリ使用量とパフォーマンスのバランスを取るため
    ピクセルをチャンクで処理し、QGISでの大容量DEMに適しています。
    """
    h, w = dem.shape
    shadow_mask = np.zeros((h, w), dtype=np.bool_)
    
    if progress is None:
        progress = NullProgress()
    
    # 効率化のためステップベクトルを事前計算
    step_dx = -cos_az * np.arange(1, max_steps + 1)
    step_dy = -sin_az * np.arange(1, max_steps + 1)
    
    # メモリ管理のためチャンクで処理
    total_pixels = h * w
    total_chunks = (total_pixels + chunk_size - 1) // chunk_size
    
    # 進捗レポート設定
    progress.set_range(total_chunks * max_steps)
    processed_operations = 0
    
    for chunk_idx, chunk_start in enumerate(range(0, total_pixels, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_pixels)
        chunk_indices = np.arange(chunk_start, chunk_end)
        
        # 平坦インデックスを2D座標に変換
        y_coords = chunk_indices // w
        x_coords = chunk_indices % w
        
        # 現在位置での標高を取得
        z0 = dem[y_coords, x_coords]
        
        # このチャンクの各ステップを確認
        for step_idx in range(max_steps):
            dx_step = step_dx[step_idx]
            dy_step = step_dy[step_idx]
            
            # ターゲット位置を計算
            xx = x_coords + dx_step
            yy = y_coords + dy_step
            
            # 境界チェック
            valid_mask = (xx >= 0) & (yy >= 0) & (xx < w - 1) & (yy < h - 1)
            
            if np.any(valid_mask):
                # 補間用の整数部と小数部を取得
                xx_valid = xx[valid_mask]
                yy_valid = yy[valid_mask]
                ix = np.floor(xx_valid).astype(np.int32)
                iy = np.floor(yy_valid).astype(np.int32)
                fx = xx_valid - ix
                fy = yy_valid - iy
                
                # インデックスが境界内であることを確認
                ix = np.clip(ix, 0, w - 2)
                iy = np.clip(iy, 0, h - 2)
                
                # バイリニア補間
                z_interp = (
                    dem[iy, ix] * (1 - fx) * (1 - fy) +
                    dem[iy, ix + 1] * fx * (1 - fy) +
                    dem[iy + 1, ix] * (1 - fx) * fy +
                    dem[iy + 1, ix + 1] * fx * fy
                )
                
                # 水平距離と仰角を計算
                horiz_dist = np.sqrt(
                    (dx * (x_coords[valid_mask] - xx_valid)) ** 2 +
                    (dy * (y_coords[valid_mask] - yy_valid)) ** 2
                )
                
                # ゼロ除算を回避
                horiz_dist = np.maximum(horiz_dist, 1e-8)
                
                elevation_angle = np.arctan2(z_interp - z0[valid_mask], horiz_dist)
                sun_angle = np.arcsin(sin_alt)
                
                # 現在影に入っているピクセルの影マスクを更新
                shadow_pixels = chunk_indices[valid_mask][elevation_angle > sun_angle]
                if len(shadow_pixels) > 0:
                    shadow_y = shadow_pixels // w
                    shadow_x = shadow_pixels % w
                    shadow_mask[shadow_y, shadow_x] = True
            
            # 進捗更新
            processed_operations += 1
            if processed_operations % 100 == 0:  # オーバーヘッド回避のため100操作ごとに更新
                progress.advance(100, f"影計算: チャンク {chunk_idx + 1}/{total_chunks}, ステップ {step_idx + 1}/{max_steps}")
    
    # 残りの進捗を進める
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
    小さなDEMまたはメモリが問題でない場合の簡単な影計算
    
    このバージョンは最大ベクトル化のため全ピクセルを同時に処理します。
    """
    h, w = dem.shape
    shadow_mask = np.zeros((h, w), dtype=np.bool_)
    
    if progress is None:
        progress = NullProgress()
    
    # 進捗レポート設定
    progress.set_range(max_steps)
    
    # 座標グリッドを作成
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    z0 = dem.copy()
    
    sun_angle = np.arcsin(sin_alt)
    
    # 各ステップを確認
    for step in range(1, max_steps + 1):
        # サンプル位置を計算
        xx = x_grid - cos_az * step
        yy = y_grid - sin_az * step
        
        # 境界チェック
        valid = (xx >= 0) & (yy >= 0) & (xx <= w - 1) & (yy <= h - 1)
        
        if not np.any(valid):
            # 残りステップの進捗を更新
            progress.advance(max_steps - step + 1, "影計算が早期完了（有効ピクセルなし）")
            break
            
        # バイリニア補間
        ix = np.clip(xx.astype(np.int32), 0, w - 2)
        iy = np.clip(yy.astype(np.int32), 0, h - 2)
        fx = np.clip(xx - ix, 0, 1)
        fy = np.clip(yy - iy, 0, 1)
        
        # インデックスが有効であることを確認
        ix = np.where(valid, ix, 0)
        iy = np.where(valid, iy, 0)
        
        z_interp = (
            dem[iy, ix] * (1 - fx) * (1 - fy) +
            dem[iy, np.minimum(ix + 1, w - 1)] * fx * (1 - fy) +
            dem[np.minimum(iy + 1, h - 1), ix] * (1 - fx) * fy +
            dem[np.minimum(iy + 1, h - 1), np.minimum(ix + 1, w - 1)] * fx * fy
        )
        
        # 仰角を計算
        horiz_dist = np.sqrt((dx * (x_grid - xx)) ** 2 + (dy * (y_grid - yy)) ** 2)
        horiz_dist = np.maximum(horiz_dist, 1e-8)  # ゼロ除算回避
        
        elevation_angle = np.arctan2(z_interp - z0, horiz_dist)
        
        # 影マスクを更新
        shadow_mask |= valid & (elevation_angle > sun_angle)
        
        # 進捗更新
        progress.advance(1, f"影計算: ステップ {step}/{max_steps}")
    
    return shadow_mask

# -----------------------------------------------------------------------------
# 公開関数
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
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """オプションの落影付きランベルト山岳陰影（QGIS最適化版）

    Parameters
    ----------
    dem : ndarray (H, W)
        標高ラスタ
    azimuth_deg, altitude_deg : float
        太陽位置（0° = 北、時計回り正）
    cellsize : float or (dy, dx)
        ピクセルサイズ。単位は*max_shadow_radius*と一致する必要がある
    cast_shadows : bool
        落影マスクを計算するかどうか
    max_shadow_radius : float
        影投射の探索距離
    memory_efficient : bool, default True
        大容量DEMのチャンク処理を使用。小容量DEMで完全ベクトル化を
        優先する場合はFalseに設定
    set_nan : float, optional
        この値をNaNに設定する
    replace_nan : float, optional
        NaN値をこの値で一時的に置換する（計算用）
    progress : ProgressReporter, optional
        進捗レポートコールバック

    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        山岳陰影値。0=完全影、1=完全照明
    """
    # 入力検証
    if dem.ndim != 2:
        raise ValueError("DEMは2次元配列である必要があります")
    if not (0 <= altitude_deg <= 90):
        raise ValueError("高度は[0, 90]度の範囲である必要があります")
    
    if progress is None:
        progress = NullProgress()
    
    # メイン進捗ステップを設定
    total_steps = 4 if cast_shadows and max_shadow_radius > 0 else 3
    progress.set_range(total_steps)
    
    progress.advance(0, "NaN値処理中...")
    
    # NaN値処理
    dem_processed, original_nan_mask = _process_nan_values(dem, set_nan, replace_nan)
    
    progress.advance(1, "表面法線ベクトル計算中...")
    
    # cellsizeパラメータを処理
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    
    # 角度をラジアンに変換
    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    sin_alt = math.sin(alt)
    cos_alt = math.cos(alt)
    sin_az = math.sin(az)
    cos_az = math.cos(az)

    # demがfloat32であることを確認（一貫性のため）
    dem_f32 = dem_processed.astype(np.float32, copy=False)
    
    # 表面法線ベクトルを計算
    nx, ny, nz = _unit_normals(dem_f32, dy, dx)
    progress.advance(1, "照明計算中...")
    
    # 入射角の余弦（ランベルトの法則）
    cos_i = (nx * cos_az + ny * sin_az) * cos_alt + nz * sin_alt
    cos_i = np.clip(cos_i, 0.0, 1.0)

    # 要求された場合は落影を計算
    if cast_shadows and max_shadow_radius > 0:
        progress.advance(0, "落影計算中...")
        
        max_steps = max(1, int(max_shadow_radius / max(dx, dy)))
        
        # DEMサイズとユーザー設定に基づいて影計算方法を選択
        total_pixels = dem.shape[0] * dem.shape[1]
        use_chunked = memory_efficient or total_pixels > 1_000_000
        
        # 影計算用のサブ進捗レポーターを作成
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
        
        # 影を適用
        cos_i[shadow_mask] = 0.0
        progress.advance(1, "影適用完了")
    
    # 元のNaN位置を復元
    cos_i[original_nan_mask] = np.nan
    
    progress.advance(1, "直射光計算完了...")
    
    return cos_i.astype(np.float32, copy=False)


def sky_light(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    max_radius: float = 100.0,
    n_directions: int = 16,
    weight_cos2: bool = True,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """等方性天空光積分（0–1）- QGIS最適化版

    Parameters
    ----------
    dem : ndarray (H,W)
        標高ラスタ
    cellsize : float or (dy,dx)
        ピクセルサイズ
    max_radius : float
        水平線スキャン距離
    n_directions : int
        方位角セクター数
    weight_cos2 : bool, default True
        各方向を``cos^2(theta)``で重み付け（Okeスタイル）。Falseの場合は
        単純なSVF平均を使用
    set_nan : float, optional
        この値をNaNに設定する
    replace_nan : float, optional
        NaN値をこの値で一時的に置換する（計算用）
    progress : ProgressReporter, optional
        進捗レポートコールバック

    Returns
    -------
    sky : ndarray (H, W) in [0,1]
        天空率。0=完全遮蔽、1=完全天空可視
    """
    if progress is None:
        progress = NullProgress()
    
    # 進捗追跡を設定
    progress.set_range(4)
    progress.advance(0, "NaN値処理中...")
    
    # NaN値処理
    dem_processed, original_nan_mask = _process_nan_values(dem, set_nan, replace_nan)
    
    progress.advance(1, "天空率計算中...")

    # 循環依存回避のためローカルインポート
    from FujiShader.shader.skyview import skyview_factor as _svf

    # SVF計算用のサブ進捗レポーターを作成
    class SubProgress:
        def __init__(self, parent_progress: ProgressReporter):
            self.parent = parent_progress
        
        def set_range(self, maximum: int) -> None:
            pass  # 親が主進捗を処理
        
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text:
                self.parent.advance(0, f"SVF: {text}")
        
        def done(self) -> None:
            pass
    
    sub_progress = SubProgress(progress)
    svf = _svf(dem_processed, cellsize=cellsize, max_radius=max_radius, 
               n_directions=n_directions, progress=sub_progress)
    
    progress.advance(1, "重み付け適用中...")
    
    if weight_cos2:
        # SVFをコサイン二乗重み付け天空照明に変換
        # SVFは1 - mean(cos^2(θ))を表すので、1 - SVFが必要
        cos2_mean = 1.0 - svf
        sky = cos2_mean
    else:
        # 生のSVFを使用
        sky = svf
    
    # 元のNaN位置を復元
    sky[original_nan_mask] = np.nan
    
    progress.advance(1, "天空光計算完了...")
    
    result = np.clip(sky, 0.0, 1.0).astype(np.float32, copy=False)
    progress.advance(1, "天空光計算完了")
    
    return result


# -----------------------------------------------------------------------------
# QGIS統合用ユーティリティ関数
# -----------------------------------------------------------------------------

def estimate_memory_usage(dem_shape: Tuple[int, int], max_shadow_radius: float, 
                         cellsize: float) -> dict:
    """
    影計算のメモリ使用量を推定してユーザーが適切な設定を選択できるよう支援
    
    Parameters
    ----------
    dem_shape : tuple of int
        DEMの形状（高さ、幅）
    max_shadow_radius : float
        最大影探索半径
    cellsize : float
        DEMのセルサイズ
    
    Returns
    -------
    dict
        メモリ推定値（MB単位）を含む辞書
    """
    h, w = dem_shape
    total_pixels = h * w
    max_steps = max(1, int(max_shadow_radius / cellsize))
    
    # 異なるアプローチのメモリを推定
    base_memory = total_pixels * 4 / 1024 / 1024  # float32のDEM
    shadow_mask_memory = total_pixels / 1024 / 1024 / 8  # booleanマスク
    
    # 単純ベクトル化アプローチ
    simple_temp_memory = total_pixels * 8 * 4 / 1024 / 1024  # float64の座標グリッド
    
    # チャンクアプローチ（1000ピクセルチャンクを想定）
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
    利用可能メモリに基づいて最適なチャンクサイズを計算
    
    Parameters
    ----------
    available_memory_mb : float
        利用可能メモリ（メガバイト）
    dem_shape : tuple of int
        DEMの形状
    
    Returns
    -------
    int
        最適なチャンクサイズ（ピクセル単位）
    """
    # 処理あたりのピクセルメモリを推定（ピクセルあたり約32バイト）
    memory_per_pixel = 32
    max_pixels = int(available_memory_mb * 1024 * 1024 / memory_per_pixel)
    
    # 総ピクセル数を超えないよう確認
    total_pixels = dem_shape[0] * dem_shape[1]
    chunk_size = min(max_pixels, total_pixels)
    
    # 効率のため最小チャンクサイズを確保
    return max(100, chunk_size)


# -----------------------------------------------------------------------------
# バッチ処理用便利関数
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
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    progress: Optional[ProgressReporter] = None
) -> NDArray[np.float32]:
    """
    直射日光と天空照明の重み付け合成を計算
    
    これは操作全体の適切な進捗レポートと共にdirect_lightとsky_lightを
    組み合わせる便利関数です。
    
    Parameters
    ----------
    dem : ndarray (H, W)
        標高ラスタ
    azimuth_deg, altitude_deg : float
        太陽位置パラメータ
    cellsize : float or (dy, dx)
        ピクセルサイズ
    cast_shadows : bool
        直射光の落影を計算するかどうか
    max_shadow_radius : float
        直射光の影探索半径
    max_sky_radius : float
        天空率半径
    n_directions : int
        天空計算の方位角方向数
    sun_weight, sky_weight : float
        直射光と天空照明の組み合わせ重み
    memory_efficient : bool
        メモリ効率処理を使用
    set_nan : float, optional
        この値をNaNに設定する
    replace_nan : float, optional
        NaN値をこの値で一時的に置換する（計算用）
    progress : ProgressReporter, optional
        進捗レポートコールバック
    
    Returns
    -------
    composite : ndarray (H, W) in [0,1]
        重み付け合成照明
    """
    if progress is None:
        progress = NullProgress()
    
    progress.set_range(3)
    
    # 直射光を計算
    progress.advance(0, "直射日光計算中...")
    
    class DirectLightProgress:
        def __init__(self, parent: ProgressReporter):
            self.parent = parent
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text: self.parent.advance(0, f"直射光: {text}")
        def done(self) -> None: pass
    
    direct = direct_light(
        dem, azimuth_deg=azimuth_deg, altitude_deg=altitude_deg,
        cellsize=cellsize, cast_shadows=cast_shadows,
        max_shadow_radius=max_shadow_radius, memory_efficient=memory_efficient,
        set_nan=set_nan, replace_nan=replace_nan,
        progress=DirectLightProgress(progress)
    )
    
    progress.advance(1, "天空照明計算中...")
    
    # 天空光を計算
    class SkyLightProgress:
        def __init__(self, parent: ProgressReporter):
            self.parent = parent
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None:
            if text: self.parent.advance(0, f"天空光: {text}")
        def done(self) -> None: pass
    
    sky = sky_light(
        dem, cellsize=cellsize, max_radius=max_sky_radius,
        n_directions=n_directions, set_nan=set_nan, replace_nan=replace_nan,
        progress=SkyLightProgress(progress)
    )
    
    progress.advance(1, "照明合成中...")
    
    # 重みを正規化
    total_weight = sun_weight + sky_weight
    if total_weight <= 0:
        raise ValueError("総重みは正の値である必要があります")
    
    sun_weight_norm = sun_weight / total_weight
    sky_weight_norm = sky_weight / total_weight
    
    # 合成
    composite = sun_weight_norm * direct + sky_weight_norm * sky
    
    progress.advance(1, "合成照明完了")
    
    return np.clip(composite, 0.0, 1.0).astype(np.float32, copy=False)