"""
FujiShader.shader.ao_ray
========================

マルチレイ・アンビエントオクルージョン（地形AO）- ベクトル化実装
--------------------------------------------------------

* 各ピクセルから *n_rays* 本のレイを放射して最大水平角を計算
* 内部ループはNumPy配列演算でベクトル化 → 純粋Pythonより数倍高速
* `stride` オプションにより粗いサンプリング間隔を設定可能で、さらなる高速化が可能
  - stride=1 で正確なピクセル単位サンプリング（従来手法と同等品質）
  - stride>1 で速度を優先し、多少の精度を犠牲にする
* Cloud Optimized GeoTIFF (COG) での大容量データ処理に対応
* NaN値の適切な処理とオプションによる値の置換機能
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from FujiShader.core.progress import ProgressReporter, NullProgress

__all__ = ["ambient_occlusion"]


def _bilinear(dem: NDArray[np.float32], y: NDArray, x: NDArray) -> NDArray[np.float32]:
    """
    浮動小数点座標(y,x)での DEM の双線形補間。
    範囲外座標や任意の角でNaNを含む場合はNaNを返す。
    """
    h, w = dem.shape
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)

    # 範囲外ピクセルをマーク
    invalid = (x0 < 0) | (x0 >= w - 1) | (y0 < 0) | (y0 >= h - 1)

    # 配列アクセスエラーを防ぐため、座標を安全な範囲にクリップ
    y0_safe = np.clip(y0, 0, h - 2)
    x0_safe = np.clip(x0, 0, w - 2)

    # 補間重み
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)

    # 四隅をサンプリング
    z00 = dem[y0_safe,     x0_safe]
    z10 = dem[y0_safe,     x0_safe + 1]
    z01 = dem[y0_safe + 1, x0_safe]
    z11 = dem[y0_safe + 1, x0_safe + 1]

    # 任意の角でNaNを含むピクセルを無効としてマーク
    invalid |= np.isnan(z00) | np.isnan(z10) | np.isnan(z01) | np.isnan(z11)

    # 双線形補間
    z = (
        z00 * (1 - tx) * (1 - ty)
        + z10 * tx * (1 - ty)
        + z01 * (1 - tx) * ty
        + z11 * tx * ty
    ).astype(np.float32, copy=False)

    # 無効ピクセルをNaNに設定
    z[invalid] = np.nan
    return z


def _preprocess_dem(
    dem: NDArray[np.float32],
    replace_nan: Union[float, None] = None,
    set_nan: Union[float, None] = None,
) -> NDArray[np.float32]:
    """
    DEM の前処理を行う。
    
    Parameters
    ----------
    dem : 2-D float32 配列
        入力DEM
    replace_nan : float or None
        NaN値をこの値で置換する。Noneの場合は置換しない
    set_nan : float or None
        この値をNaNに設定する。Noneの場合は設定しない
    
    Returns
    -------
    processed_dem : 2-D float32 配列
        前処理済みDEM
    """
    dem_processed = dem.copy()
    
    # 特定の値をNaNに設定
    if set_nan is not None:
        dem_processed[dem_processed == set_nan] = np.nan
    
    # NaN値を特定の値で置換
    if replace_nan is not None:
        dem_processed[np.isnan(dem_processed)] = replace_nan
    
    return dem_processed


def ambient_occlusion(
    dem: NDArray[np.float32],
    *,
    cellsize: Tuple[float, float] | float = 1.0,
    max_radius: float = 100.0,
    n_rays: int = 64,
    stride: int = 1,
    replace_nan: Union[float, None] = None,
    set_nan: Union[float, None] = None,
    progress: ProgressReporter | None = None,
    _stream_state=None,           # ストリームタイル処理との互換性のため（未使用）
) -> NDArray[np.float32]:
    """
    コサイン重み付きアンビエントオクルージョン（0-1）を計算する。

    Parameters
    ----------
    dem : 2-D float32 配列
        入力DEM (NaN = NoData)
    cellsize : float または (dx,dy) のタプル
        ピクセル解像度（単一値または (dx,dy) タプル）
    max_radius : float
        レイの最大距離（地上単位）
    n_rays : int
        レイ数（方向数）
    stride : int
        ピクセル単位のサンプリング間隔 - 大きい値（2,4）で高速化
    replace_nan : float or None
        NaN値をこの値で置換する。Noneの場合は置換しない
    set_nan : float or None
        この値をNaNに設定する。Noneの場合は設定しない
    progress : ProgressReporter 実装（オプション）
        進捗レポーター

    Returns
    -------
    ao : 2-D float32 配列
        *dem* と同じ形状。0 = 完全に遮蔽、1 = 遮蔽なし
    """
    # 入力検証
    if dem.ndim != 2:
        raise ValueError("入力DEMは2次元配列である必要があります")
    if not (1 <= n_rays <= 360):
        raise ValueError("n_raysは1から360の間である必要があります")
    if stride < 1:
        raise ValueError("strideは1ピクセル以上である必要があります")
    if max_radius <= 0:
        raise ValueError("max_radiusは正の値である必要があります")

    # float32型を確保
    if dem.dtype != np.float32:
        dem = dem.astype(np.float32)

    # DEM前処理
    dem = _preprocess_dem(dem, replace_nan=replace_nan, set_nan=set_nan)

    progress = progress or NullProgress()

    h, w = dem.shape
    nan_mask = np.isnan(dem)

    # ピクセル解像度
    if isinstance(cellsize, (int, float)):
        dx = dy = float(cellsize)
    else:
        dx, dy = map(float, cellsize)

    # レイ方向の単位ベクトル
    thetas = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False, dtype=np.float32)
    ux = np.cos(thetas)
    uy = np.sin(thetas)

    # サンプリングステップ配列 (1, 1+stride, 1+2*stride, ...)
    max_steps = int(np.ceil(max_radius / min(dx, dy)))
    step_indices = np.arange(1, max_steps + 1, stride, dtype=np.int32)

    # ブロードキャスト用のピクセルグリッド
    I = np.arange(h, dtype=np.float32)[:, None]  # 形状 (h,1)
    J = np.arange(w, dtype=np.float32)[None, :]  # 形状 (1,w)

    # AO出力配列（遮蔽値を累積）
    ao_map = np.zeros((h, w), dtype=np.float32)

    # 進捗設定：総ステップ数はn_rays
    progress.set_range(n_rays)

    for k in range(n_rays):
        vx, vy = ux[k], uy[k]

        # 進捗表示用テキスト
        progress_text = f"計算中: レイ {k + 1}/{n_rays} (方向角: {np.degrees(thetas[k]):.1f}°)"

        # このレイ方向に沿った各ピクセルの最大水平角を追跡
        max_angle = np.full((h, w), -np.inf, dtype=np.float32)

        for step in step_indices:
            # レイに沿ったサンプル座標
            x_f = J + vx * step
            y_f = I + vy * step

            # これらの座標での標高をサンプリング
            z_sample = _bilinear(dem, y_f, x_f)
            valid = ~np.isnan(z_sample)

            if not np.any(valid):
                # このステップですべてのサンプルが無効、以降のステップも無効
                break

            # このステップでの地上距離を計算
            dist = np.sqrt((vx * step * dx)**2 + (vy * step * dy)**2)

            # 現在のピクセルからサンプル点への仰角を計算
            angle = np.full_like(z_sample, -np.inf, dtype=np.float32)
            angle[valid] = np.arctan((z_sample[valid] - dem[valid]) / dist)

            # これまでに遭遇した最大角度を更新
            np.maximum(max_angle, angle, out=max_angle, where=valid)

            # 早期終了: すべてのピクセルが垂直に近い角度に達した場合
            if np.all((max_angle >= (np.pi/2 - 1e-6)) | nan_mask):
                break

        # 有効なサンプルが見つからなかったピクセルを処理
        max_angle[max_angle == -np.inf] = 0.0

        # 遮蔽を計算: (π/2 - max_horizon_angle) / (π/2)
        # 水平角が高い = より多くの遮蔽 = 低いAO値
        occl = (np.pi/2 - max_angle) / (np.pi/2)
        occl[nan_mask] = np.nan

        ao_map += occl
        
        # 進捗更新：1ステップ進捗、説明テキスト付き
        progress.advance(step=1, text=progress_text)

    progress.done()

    # すべてのレイで平均化
    ao_map /= float(n_rays)

    # 有効範囲[0,1]を保証し、NaN値を保持
    ao_map = np.clip(ao_map, 0.0, 1.0, out=ao_map)
    ao_map[nan_mask] = np.nan

    return ao_map.astype(np.float32, copy=False)