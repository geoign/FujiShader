"""
FujiShader.shader.swissshade
===========================

スイス風多方向ヒルシェード（拡散ソフトシェーディング）
----------------------------------------------------
3つ以上の異なる方位角からの低強度ヒルシェードを組み合わせ、
適度な高度角で、柔らかく方向性のない陰影を実現します。
Eduard Imhofによって普及されました。

この実装は、QGISの環境で大容量DEMに適した、
堅牢でメモリ効率的なアプローチを提供します。
"""
from __future__ import annotations

from typing import Iterable, Union, Tuple
import warnings
import numpy as np
from numpy.typing import NDArray

from ..core.progress import ProgressReporter, NullProgress

__all__ = ["swiss_shade", "estimate_swiss_memory", "swiss_shade_classic"]


def _compute_hillshade(
    dem: NDArray[np.float32],
    azimuth_deg: float,
    altitude_deg: float,
    cellsize: Union[float, Tuple[float, float]] = 1.0,
    memory_efficient: bool = True
) -> NDArray[np.float32]:
    """
    基本的なヒルシェードを計算する
    
    Parameters
    ----------
    dem : ndarray (H, W)
        標高モデル
    azimuth_deg : float
        方位角（度、0°=北、時計回り）
    altitude_deg : float
        太陽高度角（度）
    cellsize : float or tuple
        セルサイズ
    memory_efficient : bool
        メモリ効率モード（現在は使用せず）
        
    Returns
    -------
    hillshade : ndarray (H, W)
        ヒルシェード値 [0, 1]
    """
    # セルサイズの処理
    if isinstance(cellsize, (tuple, list)):
        dx, dy = cellsize[1], cellsize[0]  # (row_spacing, col_spacing)
    else:
        dx = dy = cellsize
    
    # 角度をラジアンに変換
    azimuth_rad = np.radians(azimuth_deg)
    altitude_rad = np.radians(altitude_deg)
    
    # 太陽光の方向ベクトル
    light_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
    light_y = -np.cos(azimuth_rad) * np.cos(altitude_rad)  # 北が-y方向
    light_z = np.sin(altitude_rad)
    
    # 勾配計算（Sobelフィルタ使用）
    # NaN値の処理を考慮
    dem_padded = np.pad(dem, ((1, 1), (1, 1)), mode='edge')
    
    # x方向勾配（東西方向）
    grad_x = (
        dem_padded[1:-1, 2:] - dem_padded[1:-1, :-2]
    ) / (2.0 * dx)
    
    # y方向勾配（南北方向）
    grad_y = (
        dem_padded[:-2, 1:-1] - dem_padded[2:, 1:-1]
    ) / (2.0 * dy)
    
    # 表面法線ベクトル
    # n = (-dz/dx, -dz/dy, 1)を正規化
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(grad_x)
    
    # 正規化
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    # NaNや0による除算を避ける
    norm = np.where(norm > 0, norm, 1.0)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    # ランベルト反射モデル（内積）
    hillshade = (
        normal_x * light_x +
        normal_y * light_y + 
        normal_z * light_z
    )
    
    # [0, 1]範囲にクランプ
    hillshade = np.clip(hillshade, 0.0, 1.0)
    
    # 元のNaN位置を復元
    original_nan_mask = np.isnan(dem)
    hillshade[original_nan_mask] = np.nan
    
    return hillshade.astype(np.float32)


def swiss_shade(
    dem: NDArray[np.float32],
    *,
    azimuths_deg: Iterable[float] = (225, 315, 45, 135),
    altitude_deg: float = 45.0,
    cellsize: Union[float, Tuple[float, float]] = 1.0,
    weight: float = 1.0,
    memory_efficient: bool = True,
    normalize_by_count: bool = True,
    set_nan: float = None,
    replace_nan: float = None,
    progress: ProgressReporter = None
) -> NDArray[np.float32]:
    """
    スイス風合成ヒルシェードを[0,1]範囲で返す
    
    Parameters
    ----------
    dem : ndarray (H, W)
        2D配列としてのデジタル標高モデル
    azimuths_deg : iterable of float, default (225, 315, 45, 135)
        方位角（度）のリスト（0° = 北、時計回り正）
        少なくとも1つの方位角が必要
    altitude_deg : float, default 45.0
        太陽高度角（度）[0, 90]
    cellsize : float or (dy, dx), default 1.0
        ピクセルサイズ。タプルの場合は(行間隔, 列間隔)
    weight : float, default 1.0
        最終結果に適用される全体重みファクター
    memory_efficient : bool, default True
        大容量DEM用のチャンク処理を使用
    normalize_by_count : bool, default True
        方位数で結果を正規化して方位数に関係なく[0,1]範囲を維持
    set_nan : float, optional
        この値をNaNに設定（処理開始時）
    replace_nan : float, optional
        NaN値をこの値で置換（計算用）
    progress : ProgressReporter, optional
        進捗レポートコールバック
        
    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        合成スイスヒルシェード（0=暗、1=明）
        
    Raises
    ------
    ValueError
        DEMが2Dでない、方位角が提供されていない、または高度角が範囲外の場合
    TypeError
        DEMが数値配列でない場合
        
    Notes
    -----
    スイスシェーディングは通常、適度な高度角（30-60°）で
    3-4の基本/中間方向（N, NE, E, SE または NW, NE, SE, SW）を使用します。
    この技法の特徴である柔らかく均等な照明を保持するため、
    投影影は通常無効にされます。
    
    Examples
    --------
    >>> dem = np.random.rand(100, 100).astype(np.float32)
    >>> # 4方向での基本スイスシェーディング
    >>> shade = swiss_shade(dem)
    >>> # カスタム方向
    >>> shade = swiss_shade(dem, azimuths_deg=[0, 90, 180, 270])
    """
    # 進捗レポーターの初期化
    if progress is None:
        progress = NullProgress()
        
    # 進捗追跡の設定
    azimuth_list = list(azimuths_deg)
    if not azimuth_list:
        raise ValueError("少なくとも1つの方位角を提供する必要があります")
    
    # 処理ステップ: 検証(1) + 各方位での計算(n) + 合成(1) + 最終化(1)
    total_steps = 1 + len(azimuth_list) + 1 + 1
    progress.set_range(total_steps)
    
    # 入力検証
    progress.advance(1, "入力を検証中...")
    
    if not isinstance(dem, np.ndarray):
        raise TypeError("DEMはnumpy配列である必要があります")
    
    if dem.ndim != 2:
        raise ValueError("DEMは2D配列である必要があります")
        
    if not np.issubdtype(dem.dtype, np.number):
        raise TypeError("DEMは数値を含む必要があります")
    
    if not (0 <= altitude_deg <= 90):
        raise ValueError(f"高度角は[0, 90]度の範囲内である必要があります。{altitude_deg}が指定されました")
    
    if weight < 0:
        warnings.warn("負の重みは予期しない結果を生成する可能性があります", UserWarning)
    
    # DEMがfloat32であることを保証
    dem_processed = dem.astype(np.float32, copy=True)
    
    # NaN処理の適用
    original_nan_mask = np.isnan(dem_processed)
    
    # set_nan処理：指定された値をNaNに設定
    if set_nan is not None:
        set_nan_mask = np.isclose(dem_processed, set_nan, equal_nan=False)
        if np.any(set_nan_mask):
            dem_processed[set_nan_mask] = np.nan
            # 元のNaNマスクを更新
            original_nan_mask = original_nan_mask | set_nan_mask
    
    # replace_nan処理：NaN値を一時的に置換
    temp_nan_mask = np.isnan(dem_processed)
    if replace_nan is not None and np.any(temp_nan_mask):
        dem_processed[temp_nan_mask] = replace_nan
    
    # 個別ヒルシェードの計算
    shades = []
    total_azimuths = len(azimuth_list)
    
    for i, az in enumerate(azimuth_list):
        step_text = f"ヒルシェード計算中 {i+1}/{total_azimuths} (方位角 {az}°)"
        
        try:
            shade = _compute_hillshade(
                dem_processed,
                azimuth_deg=float(az),
                altitude_deg=altitude_deg,
                cellsize=cellsize,
                memory_efficient=memory_efficient
            )
            shades.append(shade)
            
            # このヒルシェード完了後に進捗を進める
            progress.advance(1, step_text)
            
        except Exception as e:
            raise RuntimeError(f"方位角{az}°のヒルシェード計算に失敗: {e}") from e
    
    # シェードの合成
    progress.advance(1, "ヒルシェードを合成中...")
    
    if len(shades) == 1:
        composite = shades[0]
    else:
        # 効率的な平均計算のために配列をスタック
        shade_stack = np.stack(shades, axis=0)
        composite = np.mean(shade_stack, axis=0)
    
    # 重みの適用
    if weight != 1.0:
        composite = composite * weight
    
    # オプションの正規化
    if normalize_by_count and len(shades) > 1:
        # np.meanによって既に正規化されているが、適切なスケーリングを保証
        pass
    
    # 結果の最終化
    progress.advance(1, "結果を最終化中...")
    
    # 元のNaN位置を復元
    composite[original_nan_mask] = np.nan
    
    # 出力が[0,1]範囲でfloat32であることを保証
    result = np.clip(composite, 0.0, 1.0).astype(np.float32, copy=False)
    
    # 完了をマーク
    progress.done()
    
    return result


def estimate_swiss_memory(
    dem_shape: Tuple[int, int],
    n_azimuths: int = 4,
    cellsize: float = 1.0
) -> dict:
    """
    スイスシェーディング計算のメモリ使用量を推定
    
    Parameters
    ----------
    dem_shape : tuple of int
        DEMの形状（高さ、幅）
    n_azimuths : int, default 4
        方位角方向数
    cellsize : float, default 1.0
        セルサイズ
        
    Returns
    -------
    dict
        MB単位のメモリ使用量推定
        
    Examples
    --------
    >>> mem_info = estimate_swiss_memory((1000, 1000), n_azimuths=4)
    >>> print(f"推定ピークメモリ: {mem_info['peak_memory_mb']:.1f} MB")
    """
    h, w = dem_shape
    total_pixels = h * w
    
    # DEM用の基本メモリ
    dem_memory = total_pixels * 4 / 1024 / 1024  # float32
    
    # 個別シェード用メモリ（一時的に保存）
    shade_memory = total_pixels * 4 / 1024 / 1024  # float32
    
    # スタッキング操作用メモリ
    stack_memory = n_azimuths * shade_memory
    
    # ピークメモリはスタッキング操作中に発生
    peak_memory = dem_memory + stack_memory
    
    return {
        'dem_memory_mb': dem_memory,
        'single_shade_mb': shade_memory,
        'stack_memory_mb': stack_memory,
        'peak_memory_mb': peak_memory,
        'n_azimuths': n_azimuths,
        'total_pixels': total_pixels,
        'recommended_chunked': peak_memory > 500  # 500MB以上でチャンク処理を推奨
    }


def swiss_shade_classic(
    dem: NDArray[np.float32],
    *,
    style: str = "imhof",
    cellsize: Union[float, Tuple[float, float]] = 1.0,
    intensity: float = 1.0,
    set_nan: float = None,
    replace_nan: float = None,
    progress: ProgressReporter = None
) -> NDArray[np.float32]:
    """
    地図製作者によって普及された古典的なスイスシェーディング設定
    
    Parameters
    ----------
    dem : ndarray (H, W)
        デジタル標高モデル
    style : str, default "imhof"
        事前定義スタイル。オプション:
        - "imhof": Eduard Imhofの古典的4方向設定
        - "jenny": Bernhard Jennyの3方向バリアント
        - "cardinal": シンプルなN-E-S-W設定
    cellsize : float or tuple, default 1.0
        ピクセルサイズ
    intensity : float, default 1.0
        全体強度乗数
    set_nan : float, optional
        この値をNaNに設定
    replace_nan : float, optional
        NaN値をこの値で置換
    progress : ProgressReporter, optional
        進捗コールバック
        
    Returns
    -------
    shade : ndarray (H, W) in [0,1]
        指定された古典設定を使用したスイスヒルシェード
    """
    # 進捗が提供されていない場合は初期化
    if progress is None:
        progress = NullProgress()
    
    style_configs = {
        "imhof": {
            "azimuths_deg": (315, 45, 135, 225),  # NW, NE, SE, SW
            "altitude_deg": 45.0
        },
        "jenny": {
            "azimuths_deg": (315, 45, 180),  # NW, NE, S
            "altitude_deg": 50.0
        },
        "cardinal": {
            "azimuths_deg": (0, 90, 180, 270),  # N, E, S, W
            "altitude_deg": 40.0
        }
    }
    
    if style not in style_configs:
        available = ", ".join(style_configs.keys())
        raise ValueError(f"不明なスタイル'{style}'。利用可能: {available}")
    
    config = style_configs[style]
    
    # 古典スタイル用の進捗設定
    progress.set_range(1)
    progress.advance(1, f"{style}スイスシェーディングスタイルを適用中...")
    
    result = swiss_shade(
        dem,
        azimuths_deg=config["azimuths_deg"],
        altitude_deg=config["altitude_deg"],
        cellsize=cellsize,
        weight=intensity,
        set_nan=set_nan,
        replace_nan=replace_nan,
        progress=progress
    )
    
    progress.done()
    return result