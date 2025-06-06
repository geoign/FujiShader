"""
FujiShader.shader.hypsometric
=============================

Hypsometric tinting with slope‐aware shading
-------------------------------------------
標高バンドでの**色相**エンコーディングと**明度**の傾斜・外部シェード層による
調整を行い、他のFujiShader出力との合成に対応したRGB画像を生成します。
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib import cm
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["hypsometric_tint"]


def hypsometric_tint(
    dem: NDArray[np.float32],
    *,
    breaks: Sequence[float] = (0, 500, 1500, 3000, 6000),
    cmap_name: str = "terrain",
    shade: NDArray[np.float32] | None = None,
    shade_weight: float = 0.5,
    replace_nan: float | None = None,
    set_nan: float | None = None,
    progress: ProgressReporter | None = None,
) -> NDArray[np.float32]:
    """標高段彩図のRGB画像を返す [0,1]。
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        デジタル標高モデル
    breaks : Sequence[float]
        標高段彩のブレークポイント
    cmap_name : str
        Matplotlibカラーマップ名
    shade : NDArray[np.float32] | None
        オプション：シェードレイヤー [0,1]
    shade_weight : float
        シェード合成の重み [0,1]
    replace_nan : float | None
        NaN値をこの値で置換（処理開始時）
    set_nan : float | None
        この値をNaNに設定（処理開始時）
    progress : ProgressReporter | None
        進捗レポーターインスタンス
        
    Returns
    -------
    NDArray[np.float32]
        RGB画像 形状(*dem.shape, 3)、値域[0,1]、NaN値は適切に保持
    """
    # 進捗レポーター初期化
    progress = progress or NullProgress()
    
    # 処理ステップ数を計算（前処理 + バンド処理 + シェーディング処理 + 後処理）
    n_bands = len(breaks) - 1
    total_steps = 2 + n_bands + (1 if shade is not None else 0) + 1
    progress.set_range(total_steps)
    
    # 前処理
    progress.advance(text="前処理: DEM データの準備中...")
    dem_f = dem.astype(np.float32, copy=False)
    
    # set_nanオプション処理: 指定された値をNaNに設定
    if set_nan is not None:
        nan_mask = (dem_f == set_nan)
        dem_f = np.where(nan_mask, np.nan, dem_f)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            progress.advance(text=f"set_nan: {set_nan}の値 {nan_count:,} ピクセルをNaNに設定")
    
    # replace_nanオプション処理: NaN値を指定された値で置換
    if replace_nan is not None:
        nan_mask = np.isnan(dem_f)
        dem_f = np.where(nan_mask, replace_nan, dem_f)
        nan_count = np.sum(nan_mask)
        if nan_count > 0:
            progress.advance(text=f"replace_nan: {nan_count:,} 個のNaN値を {replace_nan} で置換")
    
    # 有効マスクの計算（NaN値以外）
    valid_mask = ~np.isnan(dem_f)
    valid_pixels = np.sum(valid_mask)
    total_pixels = dem.size
    
    progress.advance(text=f"有効ピクセル数: {valid_pixels:,} / {total_pixels:,}")
    
    # カラーマップとRGB配列の初期化
    try:
        cmap = cm.get_cmap(cmap_name, n_bands)
    except (ValueError, TypeError):
        # 無効なカラーマップ名の場合はデフォルトを使用
        progress.advance(text=f"警告: カラーマップ '{cmap_name}' が見つかりません。'terrain' を使用します")
        cmap = cm.get_cmap("terrain", n_bands)
    
    # RGB配列を初期化（NaN値はNaNのまま保持）
    rgb = np.full((*dem.shape, 3), np.nan, dtype=np.float32)
    
    # 各標高バンドの処理（ベクトル化による高速化）
    for i in range(n_bands):
        # 最後のバンドは上限なし
        if i == n_bands - 1:
            mask = (dem_f >= breaks[i]) & valid_mask
            elevation_range = f"{breaks[i]:.0f}m以上"
        else:
            mask = (dem_f >= breaks[i]) & (dem_f < breaks[i + 1]) & valid_mask
            elevation_range = f"{breaks[i]:.0f}-{breaks[i + 1]:.0f}m"
        
        # マスクされたピクセル数を計算
        band_pixels = np.sum(mask)
        
        if band_pixels > 0:
            # カラーマップから色を取得
            band_color = cmap(i)[:3]
            
            # ベクトル化された代入（高速化）
            rgb[mask] = band_color
        
        progress.advance(
            text=f"標高バンド {i+1}/{n_bands}: {elevation_range} "
                 f"({band_pixels:,} ピクセル)"
        )

    # シェーディング適用
    if shade is not None:
        progress.advance(text="シェーディング処理を適用中...")
        
        # shadeの有効性をチェック
        if shade.shape != dem.shape:
            progress.advance(text=f"警告: シェード配列の形状が一致しません ({shade.shape} vs {dem.shape})")
            # 形状が一致しない場合は処理をスキップ
        else:
            # shadeのNaN値処理
            shade_valid_mask = ~np.isnan(shade) & valid_mask
            
            if np.any(shade_valid_mask):
                # シェーディング統計情報（有効値のみ）
                shade_valid_values = shade[shade_valid_mask]
                shade_min = np.min(shade_valid_values)
                shade_max = np.max(shade_valid_values)
                shade_mean = np.mean(shade_valid_values)
                
                progress.advance(
                    text=f"シェード統計: min={shade_min:.3f}, max={shade_max:.3f}, "
                         f"mean={shade_mean:.3f}, weight={shade_weight:.2f}"
                )
                
                # シェーディング値の計算（NaN安全）
                shade_clipped = np.clip(shade, 0, 1)
                v = (1.0 - shade_weight) + shade_weight * shade_clipped
                
                # NaN値を考慮したシェーディング適用
                for channel in range(3):
                    # 有効なピクセルのみにシェーディングを適用
                    valid_shade_mask = shade_valid_mask & ~np.isnan(rgb[:, :, channel])
                    rgb[valid_shade_mask, channel] *= v[valid_shade_mask]
            else:
                progress.advance(text="警告: シェード配列に有効な値がありません")
    
    # 最終処理
    progress.advance(text="最終処理: RGB値のクリッピングと統計計算中...")
    
    # 有効値のみクリッピング（NaN値は保持）
    valid_rgb_mask = ~np.isnan(rgb)
    rgb = np.where(valid_rgb_mask, np.clip(rgb, 0, 1), rgb)
    
    # 結果の統計情報（有効値のみ）
    if np.any(valid_rgb_mask):
        valid_rgb_values = rgb[valid_rgb_mask]
        result_min = np.min(valid_rgb_values)
        result_max = np.max(valid_rgb_values)
        
        progress.advance(
            text=f"完了: RGB範囲=[{result_min:.3f}, {result_max:.3f}], "
                 f"出力形状={rgb.shape}, 有効RGB値={np.sum(valid_rgb_mask):,}"
        )
    else:
        progress.advance(text=f"完了: 出力形状={rgb.shape}, 有効RGB値なし")
    
    progress.done()
    return rgb