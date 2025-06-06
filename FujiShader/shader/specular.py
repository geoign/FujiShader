"""
FujiShader.shader.specular
==========================

地形用の金属的・スペキュラハイライト
-----------------------------------------
DEMを*スペキュラレスポンス*マップに変換する軽量Blinn–Phong実装。
拡散シェーディングと組み合わせることで、尾根が磨かれた金属のように輝きます。

なぜBlinn–Phong？
~~~~~~~~~~~~~~~~
* 高速: ピクセルあたり単一の `(N·H)^n` 冪乗項 – ソフトウェアやQGIS CPUでも十分高速。
* カートグラフィに十分な物理的妥当性（低い*n*値（広いハイライト）を選択し、
  拡散光と組み合わせた場合）。

設計ノート
~~~~~~~~~~~~
* **正射投影カメラ**（一定の視線ベクトル）を仮定 – 地図ビューレンダリングに適している。
* `direct_light()`と互換性あり。以下のように組み合わせ可能:

    ```python
    diff = fs.direct_light(dem, azimuth_deg=315, altitude_deg=45)
    spec = fs.metallic_shade(dem, azimuth_deg=315, altitude_deg=45,
                             shininess=32, view_alt_deg=60)
    rgb  = np.clip(diff[...,None] * base_rgb + spec[...,None]*[1,1,1], 0, 1)
    ```
* NumPyによる高速化実装。
"""
from __future__ import annotations

import math
from typing import Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["metallic_shade"]

# -----------------------------------------------------------------------------
# 共有ヘルパー – 表面法線（sunskyから再利用するがローカルで再実装）
# -----------------------------------------------------------------------------

def _unit_normals(arr: NDArray[np.float32], dy: float, dx: float) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """DEMから勾配を使って単位表面法線を計算する。"""
    # NaN値の存在を考慮した勾配計算
    dz_dy, dz_dx = np.gradient(arr, dy, dx, edge_order=1)
    
    # NaN値が混入している場合、その位置での法線もNaNにする
    nan_mask = np.isnan(arr) | np.isnan(dz_dx) | np.isnan(dz_dy)
    
    # 法線ベクトルの正規化（0除算を避けるために小さい値を追加）
    norm_factor = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
    norm_factor = np.where(norm_factor == 0, 1e-12, norm_factor)
    
    nz = 1.0 / norm_factor
    nx = -dz_dx * nz
    ny = -dz_dy * nz
    
    # NaN位置を復元
    nx = np.where(nan_mask, np.nan, nx)
    ny = np.where(nan_mask, np.nan, ny)
    nz = np.where(nan_mask, np.nan, nz)
    
    return nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)

# -----------------------------------------------------------------------------
# コア実装 – ベクトル化NumPy（十分高速、Numbaは不要）
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
    set_nan: Optional[float] = None,  # この値をNaNに設定
    replace_nan: Optional[float] = None,  # NaN値をこの値で置換（計算用）
    progress: Optional[ProgressReporter] = None,
) -> NDArray[np.float32]:
    """Blinn–Phongスペキュラ項 *S* を [0,1] の範囲で返す。

    Parameters
    ----------
    dem : ndarray (H, W)
        標高ラスタ。
    azimuth_deg, altitude_deg : float
        太陽ベクトル（0° = 北、時計回りが正）。
    view_az_deg : float または None, デフォルト None
        カメラ方位角。``None`` = 太陽方位角（トップライトビュー）。
    view_alt_deg : float, デフォルト 60
        カメラ仰角（90° = 天底、0° = 地平線）。
    cellsize : float または (dy,dx)
        ピクセルサイズ。
    shininess : float, デフォルト 32
        Phong指数 *n*。低い値 = 広いハイライト。
    specular_strength : float, デフォルト 0.6
        線形乗数（通常0–1）。拡散光との混合用。
    gamma : float, デフォルト 2.2
        知覚的出力用ガンマ補正係数（1.0 = 線形）。
    set_nan : float または None, デフォルト None
        この値をNaNに設定（処理開始時）。
    replace_nan : float または None, デフォルト None
        NaN値をこの値で置換（計算用、結果では元のNaN位置を復元）。
    progress : ProgressReporter または None
        進捗レポーター（オプション）。
        
    Returns
    -------
    ndarray (H, W)
        [0,1] 範囲のスペキュラレスポンス値。
    """
    progress = progress or NullProgress()
    
    # プログレス範囲を設定（処理ステップ数に基づく）
    total_steps = 8  # ステップ数を増加
    progress.set_range(total_steps)
    
    # ステップ1: 準備とNaN処理
    progress.advance(text="DEM データの準備とNaN処理...")
    
    # 入力データをfloat32にコピー
    dem_work = dem.astype(np.float32, copy=True)
    
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(dem_work)
    
    # set_nan処理：指定値をNaNに設定
    if set_nan is not None:
        set_nan_mask = np.isclose(dem_work, set_nan, equal_nan=False)
        if np.any(set_nan_mask):
            dem_work[set_nan_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | set_nan_mask
    
    # replace_nan処理：NaN値を一時的に置換（計算のため）
    if replace_nan is not None:
        current_nan_mask = np.isnan(dem_work)
        if np.any(current_nan_mask):
            dem_work[current_nan_mask] = replace_nan
    
    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    
    # ステップ2: 表面法線の計算
    progress.advance(text="表面法線の計算...")
    nx, ny, nz = _unit_normals(dem_work, dy, dx)

    # ステップ3: 光ベクトルの計算
    progress.advance(text="光ベクトルの設定...")
    azL = math.radians(azimuth_deg)
    altL = math.radians(altitude_deg)
    Lx = math.sin(azL) * math.cos(altL)
    Ly = math.cos(azL) * math.cos(altL)
    Lz = math.sin(altL)

    # ステップ4: 視点ベクトルの計算
    progress.advance(text="視点ベクトルの計算...")
    if view_az_deg is None:
        view_az_deg = azimuth_deg  # デフォルトで太陽方向から見る
    azV = math.radians(view_az_deg)
    altV = math.radians(view_alt_deg)
    Vx = math.sin(azV) * math.cos(altV)
    Vy = math.cos(azV) * math.cos(altV)
    Vz = math.sin(altV)

    # ステップ5: ハーフベクトルの計算
    progress.advance(text="ハーフベクトルの計算...")
    # ハーフベクトル H = normalize(L + V)
    Hx = Lx + Vx
    Hy = Ly + Vy
    Hz = Lz + Vz
    norm = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz) + 1e-12
    Hx /= norm; Hy /= norm; Hz /= norm

    # ステップ6: スペキュラ項の計算
    progress.advance(text="スペキュラハイライトの計算...")
    # 内積 N·H （0以上にクランプ）
    ndh = nx * Hx + ny * Hy + nz * Hz
    
    # NaN値の処理：法線がNaNの場合、内積もNaNになる
    ndh = np.clip(ndh, 0.0, 1.0)
    
    # スペキュラ項（線形）
    # NaN値がある場合、冪乗でもNaNが保持される
    S_lin = specular_strength * (ndh ** shininess)

    # ステップ7: ガンマ補正
    progress.advance(text="ガンマ補正の適用...")
    if gamma != 1.0:
        # ガンマ補正前にクランプ（NaN値は保持される）
        S_clamped = np.clip(S_lin, 0.0, 1.0)
        S_out = S_clamped ** (1.0 / gamma)
    else:
        S_out = np.clip(S_lin, 0.0, 1.0)

    # ステップ8: 最終処理とNaN復元
    progress.advance(text="最終処理とNaN値の復元...")
    
    # 元のNaN位置を復元
    S_out = np.where(original_nan_mask, np.nan, S_out)
    
    result = S_out.astype(np.float32, copy=False)

    # 処理完了
    progress.done()
    return result