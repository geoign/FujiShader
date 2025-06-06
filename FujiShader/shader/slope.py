"""
FujiShader.shader.slope
======================

DEMからの高速ベクトル化傾斜計算
---------------------------------------------
**`slope`** 関数は地表傾斜を度（デフォルト）またはパーセントで返します。
*gdaldem slope* の動作を模倣しながら、*FujiShader* の他の部分との密接な統合のために
純粋なNumPyを使用して設計されています。

重要なポイント
~~~~~~~~~~
* **中央差分** via :func:`numpy.gradient` – O(N) 実行、最適に近いキャッシュ使用。
* **任意のセルサイズ** – スカラーまたは (dy, dx) タプルを受け取るため、
  非正方形ピクセルが正しく処理されます。
* **NaN対応** – オプションの *set_nan* および *replace_nan* がエッジアーティファクトを
  最小化するためにボイドを一時的に埋め、出力でNaNを復元します。
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["slope"]

_EPS: float = 1e-12


def slope(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    unit: str = "degree",
    set_nan: Union[float, None] = None,
    replace_nan: Union[float, None] = None,
    treat_nan: Union[float, None] = None,  # 後方互換性のため保持
    dtype=np.float32,
    progress: Union[ProgressReporter, None] = None,
) -> NDArray[np.float32]:
    """DEMから地形傾斜を計算します。

    Parameters
    ----------
    dem : ndarray (H, W)
        標高ラスタ（float32が推奨）。
    cellsize : float または (float, float)、デフォルト 1.0
        ピクセルサイズ（地図単位）。タプルの場合、(dy, dx) として解釈されます。
    unit : {"degree", "percent"}、デフォルト "degree"
        出力単位。"percent" は *rise / run* ×100 を返します。
    set_nan : float または None
        この値をNaNに設定します（処理開始時）。
    replace_nan : float または None
        NaN値をこの値で一時的に置換します（勾配計算用）。
        計算後、元のNaN位置は復元されます。
    treat_nan : float または None
        後方互換性のため保持。replace_nanと同じ機能です。
    dtype : NumPy dtype、デフォルト float32
        出力配列のデータ型。
    progress : ProgressReporter または None
        進捗レポーター（オプション）。

    Returns
    -------
    slope : ndarray (H, W)
        要求された単位での傾斜。入力からのNaNは保持されます。
    """
    # 早期検証
    if dem.ndim != 2:
        raise ValueError("DEMは2次元配列である必要があります")
    
    if dem.size == 0:
        raise ValueError("DEMは空にできません")
    
    # 単位パラメータを早期に検証して不要な計算を避ける
    if unit not in ("degree", "percent", "percentage") and not unit.startswith("perc"):
        raise ValueError("unit は 'degree' または 'percent' である必要があります")
    
    # 進捗レポーターの初期化
    progress = progress or NullProgress()
    # 処理ステップの定義:
    # 1. データ準備・検証
    # 2. NaN処理
    # 3. 勾配計算
    # 4. 単位変換
    # 5. 最終処理
    progress.set_range(5)
    progress.advance(text="傾斜計算を初期化中...")

    dy, dx = (cellsize, cellsize) if np.isscalar(cellsize) else cellsize
    if dx <= 0 or dy <= 0:
        raise ValueError("cellsize は正の値である必要があります")

    arr = dem.astype(dtype, copy=False)
    progress.advance(text="データ準備が完了しました")

    # 後方互換性: treat_nanがあればreplace_nanとして使用
    if treat_nan is not None and replace_nan is None:
        replace_nan = treat_nan

    # NaN処理の実行
    original_nan_mask = np.isnan(arr)
    nan_processing_applied = False
    
    # set_nan処理: 指定された値をNaNに設定
    if set_nan is not None:
        set_mask = np.isclose(arr, set_nan, equal_nan=False)
        if np.any(set_mask):
            if not nan_processing_applied:
                arr = arr.copy()
                nan_processing_applied = True
            arr[set_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | set_mask
    
    # replace_nan処理: NaN値を一時的に置換
    if replace_nan is not None and original_nan_mask.any():
        if not nan_processing_applied:
            arr = arr.copy()
            nan_processing_applied = True
        arr[original_nan_mask] = replace_nan
        progress.advance(text="NaN値を一時的に置換中...")
    else:
        progress.advance(text="NaN処理をスキップしました")

    # エッジケースの処理: 単一要素配列
    if arr.size == 1:
        # 単一ピクセルの傾斜は0
        result = np.zeros_like(arr, dtype=dtype)
        if original_nan_mask.any():
            result[original_nan_mask] = np.nan
        # 残りのステップを完了扱いにする
        progress.advance(text="単一ピクセル処理が完了しました")
        progress.advance(text="処理が完了しました")
        progress.done()
        return result

    # 中央差分を使用した勾配計算
    progress.advance(text="勾配を計算中...")
    dz_dy, dz_dx = np.gradient(arr, dy, dx, edge_order=1)

    # 傾斜の大きさを計算
    grad_mag = np.hypot(dz_dx, dz_dy)

    # 要求された単位に変換
    progress.advance(text=f"単位を{unit}に変換中...")
    if unit == "degree":
        out = np.degrees(np.arctan(grad_mag))
    elif unit in ("percent", "percentage") or unit.startswith("perc"):
        out = grad_mag * 100.0  # パーセントに変換
    else:
        # 早期検証により到達しないはず
        raise ValueError("unit は 'degree' または 'percent' である必要があります")

    # 元の入力からNaN値を復元
    if original_nan_mask.any():
        out[original_nan_mask] = np.nan  # type: ignore[index]
    
    result = out.astype(dtype, copy=False)

    # 計算完了をマーク
    progress.done()
    return result