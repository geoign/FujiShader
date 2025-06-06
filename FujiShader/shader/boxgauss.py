"""
FujiShader.shader.topo_boxgauss
==================================

ボックスフィルタを用いたガウシアン近似によるアンシャープマスク実装
--------------------------------------------------------------------
* σ≈r/2のガウシアンを3パスの1次元ボックスフィルタで近似し、高速なO(N)計算を実現
  メモリ使用量を抑制。アンシャープマスキング効果のため`DEM - Blur`を返す。
* SciPy依存のない純粋なNumPy実装。
* マスク統合による除算でNaN処理を行い、元実装の品質を維持。
* COGストリーミング処理とQGIS進捗レポーター対応。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core.progress import ProgressReporter

__all__ = [
    "topo_boxgauss",
    "multi_scale_boxgauss",
]

# ----------------------------------------------------------------------------
# ボックスフィルタユーティリティ
# ----------------------------------------------------------------------------

def _box_filter_sum(
    arr: NDArray[np.float32], 
    r: int, 
    axis: int
) -> NDArray[np.float32]:
    """指定軸に沿って幅(2r+1)のボックスフィルタ和を適用（純粋NumPy実装）。"""
    if r < 1:
        return arr.copy()
    
    # パディング設定
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    a = np.pad(arr, pad, mode="constant", constant_values=0)
    
    # 累積和を使用した高速ボックスフィルタ
    cs = np.cumsum(a, axis=axis, dtype=np.float64)
    
    # スライシング用のインデックス設定
    slice_hi = [slice(None)] * arr.ndim
    slice_lo = [slice(None)] * arr.ndim
    slice_hi[axis] = slice(2 * r, None)
    slice_lo[axis] = slice(None, -2 * r)
    
    return (cs[tuple(slice_hi)] - cs[tuple(slice_lo)]).astype(arr.dtype, copy=False)


def _box_blur_nan_aware(
    arr: NDArray[np.float32], 
    r: int, 
    passes: int = 3,
    progress: Optional["ProgressReporter"] = None
) -> NDArray[np.float32]:
    """NaN対応マルチパスボックスブラーによるガウシアン近似。"""
    if r < 1:
        return arr.copy()

    # NaNマスクを事前計算
    nan_mask = np.isnan(arr)
    has_nan = np.any(nan_mask)
    
    if has_nan:
        # NaNがある場合は重み付き処理
        val = np.where(nan_mask, 0.0, arr).astype(np.float32, copy=False)
        wgt = (~nan_mask).astype(np.float32)
    else:
        # NaNがない場合は単純処理（高速化）
        val = arr.copy()
        wgt = None

    for pass_idx in range(passes):
        if progress:
            progress.advance(1, f"ボックスブラー {pass_idx + 1}/{passes}パス目 - 水平方向")
        
        # 水平方向処理
        val = _box_filter_sum(val, r, axis=1)
        if has_nan:
            wgt = _box_filter_sum(wgt, r, axis=1)
        
        if progress:
            progress.advance(1, f"ボックスブラー {pass_idx + 1}/{passes}パス目 - 垂直方向")
        
        # 垂直方向処理
        val = _box_filter_sum(val, r, axis=0)
        if has_nan:
            wgt = _box_filter_sum(wgt, r, axis=0)

    # 正規化処理
    if has_nan:
        out = np.empty_like(arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(val, wgt, out=out, where=(wgt > 0))
        out[wgt == 0] = np.nan
        return out
    else:
        # NaNがない場合は単純な正規化
        filter_size = (2 * r + 1) ** 2  # 2次元フィルタサイズ
        return val / (filter_size ** passes)


def _preprocess_input(
    dem: NDArray[np.float32],
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    dtype: type = np.float32
) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """
    入力DEMの前処理を行う。
    
    Returns
    -------
    processed_dem : NDArray[np.float32]
        前処理済みDEM
    original_nan_mask : NDArray[np.bool_]
        元のNaN位置のマスク
    """
    # 型変換
    arr = dem.astype(dtype, copy=True)
    
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(arr)
    
    # set_nan処理：指定値をNaNに変換
    if set_nan is not None:
        set_nan_mask = np.isclose(arr, set_nan, equal_nan=False)
        arr[set_nan_mask] = np.nan
        original_nan_mask |= set_nan_mask
    
    # replace_nan処理：NaNを指定値に置換
    if replace_nan is not None:
        current_nan_mask = np.isnan(arr)
        arr[current_nan_mask] = replace_nan
    
    return arr, original_nan_mask

# ----------------------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------------------

def topo_boxgauss(
    dem: NDArray[np.float32],
    radius: int = 8,
    *,
    passes: int = 3,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    dtype: type = np.float32,
    progress: Optional["ProgressReporter"] = None,
    _stream_state: Optional[NDArray] = None,  # stream.py用（未使用）
) -> NDArray[np.float32]:
    """
    ボックスフィルタによるガウシアン近似を用いた地形アンシャープマスキング。
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        入力2次元標高モデル
    radius : int, default=8
        フィルタ半径（ピクセル単位）（ガウシアン近似でσ ≈ radius/2）
    passes : int, default=3
        ガウシアン近似のためのボックスフィルタパス数
    replace_nan : float or None, default=None
        NaN値をこの値で置換。Noneの場合、NaN値は出力で保持される
    set_nan : float or None, default=None
        この値をNaNに設定
    dtype : type, default=np.float32
        出力データ型
    progress : ProgressReporter or None, default=None
        計算進捗を追跡するための進捗レポーター
    _stream_state : NDArray or None, default=None
        ストリーム処理用（この関数では未使用）
        
    Returns
    -------
    NDArray[np.float32]
        アンシャープマスク結果（元画像 - ブラー画像）
        
    Raises
    ------
    ValueError
        入力DEMが2次元でない場合、または半径が1未満の場合
    """
    # 入力検証
    if dem.ndim != 2:
        raise ValueError("入力DEMは2次元配列である必要があります")
    if radius < 1:
        raise ValueError("半径は1ピクセル以上である必要があります")
    if passes < 1:
        raise ValueError("パス数は1以上である必要があります")

    # 進捗初期化：前処理(1) + ブラー処理(passes*2) + 後処理(1)
    total_steps = 1 + passes * 2 + 1
    if progress:
        progress.set_range(total_steps)
        progress.advance(1, "入力データの準備中")

    # 入力前処理
    processed_dem, original_nan_mask = _preprocess_input(
        dem, replace_nan=replace_nan, set_nan=set_nan, dtype=dtype
    )

    # ボックスブラー処理
    blur = _box_blur_nan_aware(processed_dem, radius, passes=passes, progress=progress)

    if progress:
        progress.advance(1, "アンシャープマスクの計算中")

    # アンシャープマスク計算
    usm = processed_dem - blur

    # 元のNaN位置を復元（replace_nanが指定されていても）
    if replace_nan is None or np.any(original_nan_mask):
        usm[original_nan_mask] = np.nan

    if progress:
        progress.done()

    return usm.astype(dtype, copy=False)


def multi_scale_boxgauss(
    dem: NDArray[np.float32],
    radii: Union[Sequence[int], int] = (4, 16, 64, 256),
    weights: Union[Sequence[float], None] = None,
    *,
    passes: int = 3,
    normalize: bool = True,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    dtype: type = np.float32,
    progress: Optional["ProgressReporter"] = None,
) -> Tuple[NDArray[np.float32], List[NDArray[np.float32]]]:
    """
    スタック化ボックスフィルタレイヤーを用いたマルチスケール地形アンシャープマスキング。
    
    Parameters
    ----------
    dem : NDArray[np.float32]
        入力2次元標高モデル
    radii : Sequence[int] or int, default=(4, 16, 64, 256)
        各スケールのフィルタ半径
    weights : Sequence[float] or None, default=None
        スケール合成用重み。Noneの場合、半径を重みとして使用
    passes : int, default=3
        ガウシアン近似のためのボックスフィルタパス数
    normalize : bool, default=True
        99パーセンタイルを使用して出力を[-1, 1]範囲に正規化するかどうか
    replace_nan : float or None, default=None
        NaN値をこの値で置換。Noneの場合、NaN値は保持される
    set_nan : float or None, default=None
        この値をNaNに設定
    dtype : type, default=np.float32
        出力データ型
    progress : ProgressReporter or None, default=None
        計算進捗を追跡するための進捗レポーター
        
    Returns
    -------
    Tuple[NDArray[np.float32], List[NDArray[np.float32]]]
        合成されたマルチスケール結果と個別スケールレイヤーのリスト
        
    Raises
    ------
    ValueError
        いずれかの半径が1未満の場合、または重み/半径の長さが不一致の場合
    """
    # 入力検証
    if isinstance(radii, int):
        radii = (radii,)
    radii = list(radii)
    if any(r < 1 for r in radii):
        raise ValueError("すべての半径は1ピクセル以上である必要があります")

    if weights is None:
        weights = list(radii)
    if len(weights) != len(radii):
        raise ValueError("重みと半径は同じ長さである必要があります")

    # 進捗初期化：各スケール処理 + 合成処理(1) + 正規化処理(1 if normalize else 0)
    total_steps = len(radii) + 1 + (1 if normalize else 0)
    if progress:
        progress.set_range(total_steps)

    layers: List[NDArray[np.float32]] = []
    
    # 各スケールの処理
    for i, r in enumerate(radii):
        if progress:
            progress.advance(0, f"スケール {i + 1}/{len(radii)} (半径={r}) の処理中")
        
        # 個別のスケール処理では進捗レポーターを無効化
        layer = topo_boxgauss(
            dem,
            radius=r,
            passes=passes,
            replace_nan=replace_nan,
            set_nan=set_nan,
            dtype=dtype,
            progress=None,  # 個別進捗は無効化
        )
        layers.append(layer)
        
        if progress:
            progress.advance(1, f"スケール {i + 1}/{len(radii)} 完了")

    if progress:
        progress.advance(1, "マルチスケールレイヤーの合成中")

    # レイヤー合成（float64で計算して精度を確保）
    comp = np.zeros_like(layers[0], dtype=np.float64)
    total_weight = sum(weights)
    
    for w, lay in zip(weights, layers):
        comp += w * lay.astype(np.float64)
    comp /= total_weight

    # 正規化処理
    if normalize:
        if progress:
            progress.advance(0, "出力の正規化中")
        
        # NaNを除いた99パーセンタイルで正規化
        abs_comp = np.abs(comp)
        abs99 = float(np.nanpercentile(abs_comp, 99.0))
        if abs99 <= 0:
            abs99 = 1e-9  # ゼロ除算回避
        comp = np.clip(comp / abs99, -1.0, 1.0)
        
        if progress:
            progress.advance(1, "正規化完了")

    if progress:
        progress.done()

    return comp.astype(dtype, copy=False), layers