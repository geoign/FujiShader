"""
FujiShader.shader.hypsometric
=============================

Hypsometric tinting with slope‐aware shading
-------------------------------------------
Generates an RGB image where **hue** encodes elevation bands and **value** is
modulated by slope or an external shade layer—ready to blend with other
FujiShader outputs.
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
    progress: ProgressReporter | None = None,
) -> NDArray[np.float32]:
    """Return RGB hypsometric map [0,1].
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        Digital elevation model
    breaks : Sequence[float]
        Elevation breakpoints for hypsometric bands
    cmap_name : str
        Matplotlib colormap name
    shade : NDArray[np.float32] | None
        Optional shade layer [0,1]
    shade_weight : float
        Weight for shade blending [0,1]
    progress : ProgressReporter | None
        Progress reporter instance
        
    Returns
    -------
    NDArray[np.float32]
        RGB image with shape (*dem.shape, 3), values in [0,1]
    """
    # ProgressReporter 初期化
    progress = progress or NullProgress()
    
    # 処理ステップ数を計算（前処理 + バンド処理 + シェーディング処理）
    n_bands = len(breaks) - 1
    total_steps = 1 + n_bands + (1 if shade is not None else 0)
    progress.set_range(total_steps)
    
    # 前処理
    progress.advance(text="前処理: DEM データの準備中...")
    dem_f = dem.astype(np.float32, copy=False)
    
    # NaN値を処理
    valid_mask = ~np.isnan(dem_f)
    valid_pixels = np.sum(valid_mask)
    total_pixels = dem.size
    
    progress.advance(text=f"有効ピクセル数: {valid_pixels:,} / {total_pixels:,}")
    
    # カラーマップとRGB配列の初期化
    try:
        cmap = cm.get_cmap(cmap_name, n_bands)
    except ValueError:
        # 無効なカラーマップ名の場合はデフォルトを使用
        progress.advance(text=f"警告: カラーマップ '{cmap_name}' が見つかりません。'terrain' を使用します")
        cmap = cm.get_cmap("terrain", n_bands)
    
    rgb = np.zeros((*dem.shape, 3), dtype=np.float32)
    
    # 各標高バンドの処理
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
        
        # カラーマップから色を取得
        band_color = cmap(i)[:3]
        rgb[mask] = band_color
        
        progress.advance(
            text=f"標高バンド {i+1}/{n_bands}: {elevation_range} "
                 f"({band_pixels:,} ピクセル, RGB={band_color[0]:.2f},{band_color[1]:.2f},{band_color[2]:.2f})"
        )

    # シェーディング適用
    if shade is not None:
        progress.advance(text="シェーディング処理を適用中...")
        
        # shadeの有効性をチェック
        if shade.shape != dem.shape:
            progress.advance(text=f"警告: シェード配列の形状が一致しません ({shade.shape} vs {dem.shape})")
            # 形状が一致しない場合は処理をスキップ
        else:
            # shadeも有効値のみ処理
            shade_valid = np.where(valid_mask, shade, 1.0)
            
            # シェーディング統計情報
            shade_min = np.nanmin(shade_valid[valid_mask]) if np.any(valid_mask) else 0.0
            shade_max = np.nanmax(shade_valid[valid_mask]) if np.any(valid_mask) else 1.0
            shade_mean = np.nanmean(shade_valid[valid_mask]) if np.any(valid_mask) else 0.5
            
            progress.advance(
                text=f"シェード統計: min={shade_min:.3f}, max={shade_max:.3f}, "
                     f"mean={shade_mean:.3f}, weight={shade_weight:.2f}"
            )
            
            # シェーディングを適用
            v = (1.0 - shade_weight) + shade_weight * np.clip(shade_valid, 0, 1)
            rgb *= v[..., None]
    
    # 最終処理
    progress.advance(text="最終処理: RGB値のクリッピング中...")
    result = np.clip(rgb, 0, 1)
    
    # 結果の統計情報
    result_min = np.nanmin(result[valid_mask]) if np.any(valid_mask) else 0.0
    result_max = np.nanmax(result[valid_mask]) if np.any(valid_mask) else 1.0
    
    progress.advance(
        text=f"完了: RGB範囲=[{result_min:.3f}, {result_max:.3f}], "
             f"出力形状={result.shape}"
    )
    
    progress.done()
    return result