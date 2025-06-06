"""
FujiShader.shader.curvature
===========================

プロファイル、平面、および全曲率（2次微分）
-------------------------------------------
Zevenbergen & Thorne (1987) で概説されているアルゴリズムを3×3
ウィンドウ係数を用いて実装。曲率は1/mで返され、符号規則はGDALに従う
（+ 値 = プロファイル曲率では凸）。

実装では有限差分法を用いて標高データの1次および2次偏微分を計算し、
Zevenbergen & Thorne (1987) に従って曲率式を適用します。

公開関数
~~~~~~~~
* `profile_curvature` - 最大傾斜方向の曲率
* `plan_curvature` - 最大傾斜方向に垂直な曲率  
* `total_curvature` - 平均曲率（ラプラシアン）

参考文献
~~~~~~~~
Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land 
surface topography. Earth surface processes and landforms, 12(1), 47-56.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional, Protocol
import numpy as np
from numpy.typing import NDArray

# 進捗レポータープロトコルの型ヒント定義
class ProgressReporterProtocol(Protocol):
    """進捗レポーターのプロトコル。"""
    def set_range(self, maximum: int) -> None: ...
    def advance(self, step: int = 1, text: Optional[str] = None) -> None: ...
    def done(self) -> None: ...


# 進捗レポートのインポート - モジュール構造に応じてパスを調整
try:
    from ..core.progress import ProgressReporter, NullProgress
except ImportError:
    # スタンドアロン使用時のフォールバック
    class NullProgress:
        def set_range(self, maximum: int) -> None: pass
        def advance(self, step: int = 1, text: Optional[str] = None) -> None: pass
        def done(self) -> None: pass

__all__ = [
    "profile_curvature",
    "plan_curvature", 
    "total_curvature",
]


def _validate_dem(dem: NDArray[np.float32]) -> None:
    """DEM入力配列を検証する。
    
    Args:
        dem: デジタル標高モデル配列
        
    Raises:
        ValueError: DEMが小さすぎるか無効な形状の場合
    """
    if dem.ndim != 2:
        raise ValueError("DEMは2次元配列である必要があります")
    
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        raise ValueError("曲率計算にはDEMは最低3x3ピクセル必要です")


def _preprocess_dem(
    dem: NDArray[np.float32],
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None
) -> NDArray[np.float32]:
    """DEM前処理：NaN処理とデータクリーニング。
    
    Args:
        dem: デジタル標高モデル配列
        replace_nan: NaN値をこの値で置換（None の場合は置換しない）
        set_nan: この値をNaNに設定（None の場合は設定しない）
        
    Returns:
        前処理されたDEM配列
    """
    # コピーを作成して元のデータを保護
    processed = dem.astype(np.float32, copy=True)
    
    # 特定の値をNaNに設定
    if set_nan is not None:
        processed[processed == set_nan] = np.nan
    
    # NaN値を特定の値で置換
    if replace_nan is not None:
        processed[np.isnan(processed)] = replace_nan
    
    return processed


def _gradients_2nd(
    dem: NDArray[np.float32], 
    dy: float, 
    dx: float,
    progress: Optional[ProgressReporterProtocol] = None
) -> Tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], 
    NDArray[np.float32], NDArray[np.float32]
]:
    """有限差分を用いて1次および2次偏微分を計算する。
    
    1次微分には中央差分近似、2次微分には2次有限差分を使用。
    
    Args:
        dem: デジタル標高モデル配列
        dy: y方向のセルサイズ（通常、北向きグリッドでは負値）
        dx: x方向のセルサイズ
        progress: オプショナルな進捗レポーター
        
    Returns:
        (dz/dx, dz/dy, d²z/dx², d²z/dy², d²z/dxdy) のタプル
        すべての配列の形状は (height-2, width-2)
    """
    if progress is None:
        progress = NullProgress()
    
    z = dem.astype(np.float32, copy=False)
    
    # 中央差分を用いた1次偏微分
    # すべての結果配列は (height-2, width-2) になる
    progress.advance(1, "1次偏微分を計算中 (dz/dx)")
    dzdx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2.0 * dx)
    
    progress.advance(1, "1次偏微分を計算中 (dz/dy)")
    dzdy = (z[2:, 1:-1] - z[:-2, 1:-1]) / (2.0 * dy)
    
    # 有限差分を用いた2次偏微分
    progress.advance(1, "2次偏微分を計算中 (d²z/dx²)")
    d2zdx2 = (z[1:-1, 2:] - 2.0 * z[1:-1, 1:-1] + z[1:-1, :-2]) / (dx * dx)
    
    progress.advance(1, "2次偏微分を計算中 (d²z/dy²)")
    d2zdy2 = (z[2:, 1:-1] - 2.0 * z[1:-1, 1:-1] + z[:-2, 1:-1]) / (dy * dy)
    
    # 混合偏微分（交差微分）
    progress.advance(1, "交差偏微分を計算中 (d²z/dxdy)")
    d2zdxdy = (
        z[2:, 2:] + z[:-2, :-2] - z[2:, :-2] - z[:-2, 2:]
    ) / (4.0 * dx * dy)
    
    return dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy


def _curvature_impl(
    dem: NDArray[np.float32],
    *,
    cellsize: Union[Tuple[float, float], float] = 1.0,
    kind: str,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    progress: Optional[ProgressReporterProtocol] = None,
) -> NDArray[np.float32]:
    """曲率計算の内部実装。
    
    Args:
        dem: デジタル標高モデル配列
        cellsize: (dy, dx) タプルまたは正方セル用の単一値としてのセルサイズ
        kind: 曲率の種類 - 'profile', 'plan', または 'total'
        replace_nan: NaN値をこの値で置換（None の場合は置換しない）
        set_nan: この値をNaNに設定（None の場合は設定しない）
        progress: オプショナルな進捗レポーター
        
    Returns:
        入力DEMと同じ形状の曲率配列、境界部分はNaN
        
    Raises:
        ValueError: kindが認識されないかDEMが無効な場合
    """
    _validate_dem(dem)
    
    # cellsizeパラメータの処理
    if np.isscalar(cellsize):
        dy, dx = float(cellsize), float(cellsize)
    else:
        dy, dx = float(cellsize[0]), float(cellsize[1])
    
    # 進捗レポートの初期化
    if progress is None:
        progress = NullProgress()
    
    # プログレス範囲を設定（前処理1 + 偏微分計算5 + 曲率計算1 + 出力配列作成1）
    progress.set_range(8)
    
    progress.advance(1, f"{kind}曲率の計算を開始")
    
    # DEM前処理
    processed_dem = _preprocess_dem(dem, replace_nan, set_nan)
    
    # 偏微分を計算
    dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = _gradients_2nd(processed_dem, dy, dx, progress)
    
    # Zevenbergen & Thorne記法に従った共通変数名を割り当て
    p = dzdx  # x方向1次微分
    q = dzdy  # y方向1次微分  
    a = d2zdx2  # x方向2次微分
    b = d2zdxdy  # 混合2次微分
    c = d2zdy2  # y方向2次微分
    
    # 種類に基づいて曲率を計算
    progress.advance(1, f"{kind}曲率を計算中")
    
    if kind == "profile":
        # プロファイル曲率：最大傾斜方向の曲率
        # 公式: -(a*p² + 2*b*p*q + c*q²) / (1 + p² + q²)^(3/2)
        denom = (1.0 + p**2 + q**2) ** 1.5
        # 平坦地域での0除算を防ぐため、小さなイプシロンを追加
        denom = np.maximum(denom, 1e-10)
        curv = -((a * p**2 + 2.0 * b * p * q + c * q**2) / denom)
        
    elif kind == "plan":
        # 平面曲率：最大傾斜方向に垂直な曲率
        # 公式: -(a*q² - 2*b*p*q + c*p²) / (1 + p² + q²)^(1/2)
        denom = (1.0 + p**2 + q**2) ** 0.5
        # 平坦地域での0除算を防ぐため、小さなイプシロンを追加
        denom = np.maximum(denom, 1e-10)
        curv = -((a * q**2 - 2.0 * b * p * q + c * p**2) / denom)
        
    elif kind == "total":
        # 全曲率（平均曲率）：標高のラプラシアン
        # 公式: a + c = d²z/dx² + d²z/dy²
        curv = a + c
        
    else:
        raise ValueError(f"不明な曲率種別 '{kind}'。'profile', 'plan', または 'total' である必要があります")
    
    # NaN境界を持つ出力配列を作成
    progress.advance(1, "出力配列を作成中")
    out = np.full_like(dem, np.nan, dtype=np.float32)
    
    # NaNを適切に処理：入力にNaNがある場合、出力もNaNにする
    valid_mask = ~np.isnan(processed_dem[1:-1, 1:-1])
    out[1:-1, 1:-1] = np.where(valid_mask, curv, np.nan)
    
    progress.done()
    
    return out


def profile_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """デジタル標高モデルからプロファイル曲率を計算する。
    
    プロファイル曲率は最大傾斜方向における傾斜の変化率を測定する。
    正の値は凸領域（尾根）を示し、負の値は凹領域（谷）を示す。
    
    Args:
        dem: 2次元配列としてのデジタル標高モデル（最低3x3必要）
        cellsize: (dy, dx) タプルまたは正方セル用の単一値としてのセルサイズ。
                 単位はDEM標高単位と一致する必要がある。
        replace_nan: NaN値をこの値で置換（None の場合は置換しない）
        set_nan: この値をNaNに設定（None の場合は設定しない）
        progress: 計算進捗を追跡するオプショナルな進捗レポーター
    
    Returns:
        入力DEMと同じ形状のプロファイル曲率配列。
        境界ピクセルはNaNに設定される。単位は1/[長さ単位]。
        
    Raises:
        ValueError: DEMが小さすぎるか無効な次元を持つ場合
        
    Example:
        >>> import numpy as np
        >>> # シンプルな尾根を作成
        >>> dem = np.array([[1, 2, 1], [1, 3, 1], [1, 2, 1]], dtype=np.float32)
        >>> prof_curv = profile_curvature(dem, cellsize=1.0)
        >>> # 中央ピクセルは正の曲率（凸尾根）を示すはず
    """
    return _curvature_impl(
        dem, 
        cellsize=cellsize, 
        kind="profile", 
        replace_nan=replace_nan,
        set_nan=set_nan,
        progress=progress
    )


def plan_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """デジタル標高モデルから平面曲率を計算する。
    
    平面曲率は方位（傾斜方向）の変化率を測定する。
    最大傾斜方向に垂直な曲率を表す。
    正の値は発散流領域を示し、負の値は収束流領域を示す。
    
    Args:
        dem: 2次元配列としてのデジタル標高モデル（最低3x3必要）
        cellsize: (dy, dx) タプルまたは正方セル用の単一値としてのセルサイズ。
                 単位はDEM標高単位と一致する必要がある。
        replace_nan: NaN値をこの値で置換（None の場合は置換しない）
        set_nan: この値をNaNに設定（None の場合は設定しない）
        progress: 計算進捗を追跡するオプショナルな進捗レポーター
    
    Returns:
        入力DEMと同じ形状の平面曲率配列。
        境界ピクセルはNaNに設定される。単位は1/[長さ単位]。
        
    Raises:
        ValueError: DEMが小さすぎるか無効な次元を持つ場合
        
    Example:
        >>> import numpy as np
        >>> # シンプルな谷を作成
        >>> dem = np.array([[2, 1, 2], [2, 1, 2], [2, 1, 2]], dtype=np.float32)
        >>> plan_curv = plan_curvature(dem, cellsize=1.0)
        >>> # 中央ピクセルは負の曲率（収束）を示すはず
    """
    return _curvature_impl(
        dem, 
        cellsize=cellsize, 
        kind="plan", 
        replace_nan=replace_nan,
        set_nan=set_nan,
        progress=progress
    )


def total_curvature(
    dem: NDArray[np.float32], 
    *, 
    cellsize: Union[Tuple[float, float], float] = 1.0,
    replace_nan: Optional[float] = None,
    set_nan: Optional[float] = None,
    progress: Optional[ProgressReporterProtocol] = None
) -> NDArray[np.float32]:
    """デジタル標高モデルから全曲率（平均曲率）を計算する。
    
    全曲率は標高面のラプラシアンであり、各点における平均曲率を表す。
    主曲率の和であり、ほとんどの場合プロファイル曲率と平面曲率の和に等しい。
    
    Args:
        dem: 2次元配列としてのデジタル標高モデル（最低3x3必要）
        cellsize: (dy, dx) タプルまたは正方セル用の単一値としてのセルサイズ。
                 単位はDEM標高単位と一致する必要がある。
        replace_nan: NaN値をこの値で置換（None の場合は置換しない）
        set_nan: この値をNaNに設定（None の場合は設定しない）
        progress: 計算進捗を追跡するオプショナルな進捗レポーター
    
    Returns:
        入力DEMと同じ形状の全曲率配列。
        境界ピクセルはNaNに設定される。単位は1/[長さ単位]。
        
    Raises:
        ValueError: DEMが小さすぎるか無効な次元を持つ場合
        
    Example:
        >>> import numpy as np
        >>> # シンプルなドームを作成
        >>> dem = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.float32)
        >>> total_curv = total_curvature(dem, cellsize=1.0)
        >>> # 中央ピクセルは正の曲率（凸ドーム）を示すはず
    """
    return _curvature_impl(
        dem, 
        cellsize=cellsize, 
        kind="total", 
        replace_nan=replace_nan,
        set_nan=set_nan,
        progress=progress
    )