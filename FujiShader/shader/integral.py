"""
FujiShader.shader.integral
==============================

超大 DEM 向け **Summed‑Area Table (積分画像) ベース** 
----------------------------------------------------------------------
* FFT を一切使わず、2 回の累積和と差分だけで **O(N)** にスケール。
* 円盤平均の完全再現ではなく **正方窓** (square box) 平均を採用
  ⇒ 尾根/谷の符号表現は同じで、計算は桁違いに高速。
* SciPy も Numba も不要。NumPy さえあれば QGIS コンソールで動作。
* 元 ``topo_usm`` / ``multi_scale_usm`` とシグネチャ互換。

制限事項
~~~~~~~~
* 正方窓なので半径 r に対応する *円盤* 平均とスペクトルが若干異なります
  （視覚的にはほぼ問題ないことを確認）。
* 複数スケール合成を行う際に、円盤版と全く同じウェイトを使うと
  コントラストがわずかに変わることがあります。必要ならウェイトを微調整してください。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple, Union
import warnings

import numpy as np
from numpy.typing import NDArray

# プログレス管理のインポート
from ..core.progress import ProgressReporter, NullProgress

__all__ = [
    "topo_integral",
    "multi_scale_integral",
    "integral_image",
]

# 型エイリアス（明確性のため）
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

# ----------------------------------------------------------------------------
# 内部ヘルパー関数
# ----------------------------------------------------------------------------

def _validate_input_array(arr: np.ndarray, name: str = "array") -> None:
    """入力配列のプロパティを検証する。"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} は numpy 配列である必要があります")
    if arr.ndim != 2:
        raise ValueError(f"{name} は 2D 配列である必要があります。{arr.ndim}D が指定されました")
    if arr.size == 0:
        raise ValueError(f"{name} は空にできません")


def _preprocess_nan_values(
    arr: np.ndarray, 
    set_nan: float | None = None, 
    replace_nan: float | None = None,
    progress: ProgressReporter = None
) -> Float32Array:
    """
    NaN値の前処理を行う。
    
    Args:
        arr: 入力配列
        set_nan: この値をNaNに設定する
        replace_nan: NaN値をこの値で置換する
        progress: 進捗レポーター
        
    Returns:
        前処理済みの配列
    """
    if progress is None:
        progress = NullProgress()
    
    progress.advance(1, "NaN値の前処理を開始")
    
    # float32に変換
    result = arr.astype(np.float32, copy=True)
    
    # 特定の値をNaNに設定
    if set_nan is not None:
        if not np.isfinite(set_nan):
            raise ValueError("set_nan は有限の数値である必要があります")
        progress.advance(1, f"値 {set_nan} をNaNに設定中")
        mask = np.isclose(result, set_nan, equal_nan=False)
        if np.any(mask):
            result[mask] = np.nan
            progress.advance(1, f"{np.sum(mask)} ピクセルをNaNに設定しました")
        else:
            progress.advance(1, "設定対象の値が見つかりませんでした")
    
    # NaN値を置換
    if replace_nan is not None:
        if not np.isfinite(replace_nan):
            raise ValueError("replace_nan は有限の数値である必要があります")
        progress.advance(1, f"NaN値を {replace_nan} で置換中")
        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            result[nan_mask] = replace_nan
            progress.advance(1, f"{np.sum(nan_mask)} 個のNaN値を置換しました")
        else:
            progress.advance(1, "置換対象のNaN値が見つかりませんでした")
    
    return result


def _moving_sum_axis(
    arr: Float32Array, 
    r: int, 
    axis: int,
    progress: ProgressReporter = None
) -> Float32Array:
    """積分画像を使用して指定軸に沿って (2r + 1) 幅の移動**合計**を計算する。

    DEM は両端で r セル分ゼロパディングされ、その後累積和が計算される。
    プレフィックス和に*ゼロスライスを前置*することで、ウィンドウ合計を
    `csum[i + w] − csum[i]` として取得でき、**元の入力と全く同じ形状**の
    出力配列が得られる（オフバイワンエラーなし）。
    
    Args:
        arr: 入力配列
        r: 半径（>= 0である必要がある）
        axis: 移動合計を計算する軸（0または1）
        progress: 進捗レポーターインスタンス
        
    Returns:
        入力と同じ形状で移動合計が格納された配列
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"半径は >= 0 である必要があります。{r} が指定されました")
    if axis not in (0, 1):
        raise ValueError(f"軸は 0 または 1 である必要があります。{axis} が指定されました")
    if r == 0:
        # ブラーが要求されていない - コピーを返す
        progress.advance(1, f"軸 {axis} でブラー不要（r=0）")
        return arr.astype(np.float32, copy=True)

    # 指定軸の配列形状を検証
    if arr.shape[axis] == 0:
        raise ValueError("サイズゼロの軸に移動合計を適用できません")

    axis_name = "Y" if axis == 0 else "X"
    progress.advance(1, f"{axis_name}軸ブラー用に配列をパディング中（r={r}）")

    # パディング仕様を作成
    pad_spec = [(0, 0)] * arr.ndim
    pad_spec[axis] = (r, r)
    
    try:
        padded = np.pad(arr, pad_spec, mode="constant", constant_values=0.0)
    except Exception as e:
        raise ValueError(f"配列のパディングに失敗しました: {e}")

    # 数値的安定性のためfloat64でプレフィックス和を計算
    progress.advance(1, f"{axis_name}軸に沿って累積和を計算中")
    try:
        csum = np.cumsum(padded, axis=axis, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"累積和の計算に失敗しました: {e}")

    # 積分軸に沿って単一のゼロスライスを前置
    progress.advance(1, f"{axis_name}軸のウィンドウ抽出を準備中")
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    try:
        zero_slice = np.zeros(zero_shape, dtype=np.float64)
        csum = np.concatenate((zero_slice, csum), axis=axis)
    except Exception as e:
        raise ValueError(f"ゼロスライスの連結に失敗しました: {e}")

    # ウィンドウ合計を抽出
    n = arr.shape[axis]
    w = 2 * r + 1  # ウィンドウ幅

    # 差分演算用のスライスオブジェクトを作成
    sl_hi = [slice(None)] * arr.ndim
    sl_lo = [slice(None)] * arr.ndim
    sl_hi[axis] = slice(w, w + n)
    sl_lo[axis] = slice(0, n)

    progress.advance(1, f"{axis_name}軸ウィンドウ合計を抽出中")
    try:
        result = csum[tuple(sl_hi)] - csum[tuple(sl_lo)]
        return result.astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"ウィンドウ差分の計算に失敗しました: {e}")


def _square_sum(
    arr: Float32Array, 
    r: int, 
    progress: ProgressReporter = None
) -> Float32Array:
    """すべてのピクセルで (2r + 1)×(2r + 1) の**合計**を返す。
    
    Args:
        arr: 入力配列（NaNは事前処理済みであること）
        r: ピクセル単位の半径
        progress: 進捗レポーターインスタンス
        
    Returns:
        正方ウィンドウ合計が格納された同形状配列
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"半径は >= 0 である必要があります。{r} が指定されました")
    if r == 0:
        progress.advance(2, "正方ブラー不要（r=0）")
        return arr.astype(np.float32, copy=True)
        
    # 両次元で移動合計を適用
    try:
        tmp = _moving_sum_axis(arr, r, axis=1, progress=progress)  # X パス
        return _moving_sum_axis(tmp, r, axis=0, progress=progress)  # Y パス
    except Exception as e:
        raise ValueError(f"正方合計の計算に失敗しました: {e}")


def _square_blur_nan(
    arr: Float32Array, 
    r: int, 
    progress: ProgressReporter = None
) -> Float32Array:
    """半径 r のNaN対応正方ウィンドウ**平均**。
    
    Args:
        arr: 入力配列
        r: ピクセル単位の半径
        progress: 進捗レポーターインスタンス
        
    Returns:
        NaN対応正方ウィンドウ平均が格納された同形状配列
    """
    if progress is None:
        progress = NullProgress()
        
    if r < 0:
        raise ValueError(f"半径は >= 0 である必要があります。{r} が指定されました")
    
    # 自明なケースを処理
    if r == 0:
        progress.advance(1, "ブラー不要（r=0）")
        return arr.astype(np.float32, copy=True)
    
    # NaN処理を分離
    progress.advance(1, f"ブラー用のNaN分布を解析中（r={r}）")
    nan_mask = np.isnan(arr)
    has_nan = np.any(nan_mask)
    
    if not has_nan:
        # NaNなし - シンプルなケース
        progress.advance(1, "シンプルな正方ブラーを計算中（NaNなし）")
        window_area = (2 * r + 1) ** 2
        total_sum = _square_sum(arr, r, progress)
        return total_sum / window_area
    
    # NaN対応計算
    progress.advance(1, "NaN対応ブラー計算を準備中")
    arr_filled = np.where(nan_mask, 0.0, arr).astype(np.float32, copy=False)
    weight = (~nan_mask).astype(np.float32, copy=False)

    try:
        progress.advance(1, "分子を計算中（埋込値）")
        numerator = _square_sum(arr_filled, r, progress)
        progress.advance(1, "分母を計算中（有効重み）")
        denominator = _square_sum(weight, r, progress)
    except Exception as e:
        raise ValueError(f"NaN対応合計の計算に失敗しました: {e}")

    # 適切なNaN処理で平均を計算
    progress.advance(1, "最終的なNaN対応平均を計算中")
    result = np.empty_like(arr, dtype=np.float32)
    
    # 除算にnumpyのエラー処理を使用
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = denominator > 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        result[~valid_mask] = np.nan
    
    return result


def integral_image(
    img: np.ndarray, 
    progress: ProgressReporter = None
) -> Float32Array:
    """入力の積分画像（加算面積表）を計算する。
    
    Args:
        img: 入力2D配列
        progress: 進捗レポーターインスタンス
        
    Returns:
        入力と同形状の積分画像
        
    Note:
        これはコア機能に焦点を当てた簡略版です。
        明確性と保守性のためストリーミング機能は削除されています。
    """
    if progress is None:
        progress = NullProgress()
        
    _validate_input_array(img, "img")
    
    progress.set_range(3)
    progress.advance(1, "計算用にfloat64に変換中")
    
    try:
        # 計算用にfloat64に変換し、その後float32に戻す
        img_f64 = img.astype(np.float64, copy=False)
        progress.advance(1, "累積和を計算中（軸0）")
        result = img_f64.cumsum(axis=0)
        progress.advance(1, "累積和を計算中（軸1）")
        result = result.cumsum(axis=1)
        return result.astype(np.float32, copy=False)
    except Exception as e:
        raise ValueError(f"積分画像の計算に失敗しました: {e}")
    finally:
        progress.done()


# ----------------------------------------------------------------------------
# 公開API
# ----------------------------------------------------------------------------

def topo_integral(
    dem: Union[np.ndarray, Float32Array],
    radius: int = 8,
    *,
    kernel: str = "square",  # シグネチャ互換性のため保持
    use_fft: bool | None = None,  # 無視 – SATは常に空間的
    set_nan: float | None = None,
    replace_nan: float | None = None,
    dtype=np.float32,
    progress: ProgressReporter = None,
    # 後方互換性のため
    treat_nan: float | None = None,
) -> Float32Array:
    """**正方ウィンドウ** SATブラーを使用して符号付き局所起伏レイヤを計算する。
    
    Args:
        dem: 2D配列としての入力DEM
        radius: ピクセル単位のブラー半径（>= 1である必要がある）
        kernel: カーネルタイプ（"square"のみサポート、互換性のため保持）
        use_fft: 無視される（互換性のため保持）
        set_nan: この値をNaNに設定する
        replace_nan: NaN値をこの値で置換する
        dtype: 出力データ型（float32または互換型である必要がある）
        progress: 進捗レポーターインスタンス
        treat_nan: 後方互換性のため（replace_nanと同じ）
        
    Returns:
        アンシャープマスク結果（元画像 - ブラー画像）
        
    Raises:
        ValueError: 無効な入力に対して
        TypeError: 間違った入力型に対して
    """
    if progress is None:
        progress = NullProgress()
    
    # 後方互換性：treat_nanをreplace_nanに変換
    if treat_nan is not None and replace_nan is None:
        replace_nan = treat_nan
    
    # 進捗追跡の設定 - 必要なステップ数を推定
    total_steps = 12  # セットアップ、検証、ブラー、ファイナライゼーション用の基本ステップ
    if set_nan is not None or (np.any(np.isnan(dem)) and replace_nan is None):
        total_steps += 6  # NaN処理用の追加ステップ
    
    progress.set_range(total_steps)
    
    try:
        # 入力検証
        progress.advance(1, f"入力DEMを検証中（形状: {dem.shape}）")
        _validate_input_array(dem, "dem")
        
        if radius < 1:
            raise ValueError(f"半径は >= 1 ピクセルである必要があります。{radius} が指定されました")
        
        if kernel != "square":
            warnings.warn(f"「square」カーネルのみサポートされています。「{kernel}」が指定されました。「square」を使用します。")
        
        # 作業データ型に変換
        progress.advance(1, f"作業データ型（{dtype}）に変換中")
        try:
            arr_orig = dem.astype(dtype, copy=False)
        except Exception as e:
            raise ValueError(f"DEMをデータ型 {dtype} に変換できませんでした: {e}")

        # NaN前処理
        progress.advance(1, "NaN値を前処理中")
        arr_processed = _preprocess_nan_values(
            arr_orig, 
            set_nan=set_nan, 
            replace_nan=replace_nan, 
            progress=progress
        )
        
        # 元のNaN位置を保存（後で復元用）
        original_nan_mask = np.isnan(arr_orig)
        preserve_nan = (replace_nan is None)

        # ブラーとアンシャープマスクを計算
        progress.advance(1, f"正方ブラー計算を開始中（半径={radius}）")
        try:
            blur = _square_blur_nan(arr_processed, radius, progress)
            progress.advance(1, "アンシャープマスクを計算中（元画像 - ブラー画像）")
            usm = arr_orig - blur
        except Exception as e:
            raise ValueError(f"地形アンシャープマスクの計算に失敗しました: {e}")

        # 要求された場合、元のNaN位置を保持
        if preserve_nan:
            progress.advance(1, "元のNaN位置を保持中")
            if np.any(original_nan_mask):
                usm[original_nan_mask] = np.nan
        else:
            progress.advance(1, "結果を確定中")

        progress.advance(1, f"結果を出力データ型（{dtype}）に変換中")
        result = usm.astype(dtype, copy=False)
        
        progress.advance(1, "地形積分計算完了")
        return result
        
    except Exception as e:
        progress.advance(1, f"エラーが発生しました: {str(e)}")
        raise
    finally:
        progress.done()


def multi_scale_integral(
    dem: Union[np.ndarray, Float32Array],
    radii: Union[Sequence[int], int] = (4, 16, 64, 256),
    weights: Sequence[float] | None = None,
    *,
    kernel: str = "square",
    use_fft: bool | None = None,
    normalize: bool = True,
    set_nan: float | None = None,
    replace_nan: float | None = None,
    dtype=np.float32,
    progress: ProgressReporter = None,
    # 後方互換性のため
    treat_nan: float | None = None,
) -> Tuple[Float32Array, List[Float32Array]]:
    """複数のSATベースUSMレイヤを合成起伏画像にスタックする。
    
    Args:
        dem: 2D配列としての入力DEM
        radii: ブラー半径のシーケンス、または単一半径
        weights: レイヤ合成用の重み（デフォルト：radiiを重みとして使用）
        kernel: カーネルタイプ（"square"のみサポート）
        use_fft: 無視される（互換性のため保持）
        normalize: 出力を[-1, 1]範囲に正規化するかどうか
        set_nan: この値をNaNに設定する
        replace_nan: NaN値をこの値で置換する
        dtype: 出力データ型
        progress: 進捗レポーターインスタンス
        treat_nan: 後方互換性のため（replace_nanと同じ）
        
    Returns:
        (合成結果, 個別レイヤ)のタプル
        
    Raises:
        ValueError: 無効な入力に対して
    """
    if progress is None:
        progress = NullProgress()
    
    # 後方互換性：treat_nanをreplace_nanに変換
    if treat_nan is not None and replace_nan is None:
        replace_nan = treat_nan
    
    # 入力検証
    _validate_input_array(dem, "dem")
    
    # radiiパラメータを処理
    if isinstance(radii, int):
        if radii < 1:
            raise ValueError(f"半径は >= 1 である必要があります。{radii} が指定されました")
        radii = (radii,)
    else:
        radii = list(radii)
        if not radii:
            raise ValueError("radiiは空にできません")
        if any(r < 1 for r in radii):
            raise ValueError(f"すべての半径は >= 1 px である必要があります。{radii} が指定されました")

    # weightsパラメータを処理
    if weights is None:
        weights = list(radii)  # デフォルト重みとしてradiiを使用
    else:
        weights = list(weights)
        if len(weights) != len(radii):
            raise ValueError(
                f"長さの不一致: weights（{len(weights)}）vs radii（{len(radii)}）"
            )
        if any(w <= 0 for w in weights):
            raise ValueError("すべての重みは正の値である必要があります")

    if kernel != "square":
        warnings.warn(f"「square」カーネルのみサポートされています。「{kernel}」が指定されました。「square」を使用します。")

    # 進捗追跡の設定
    num_layers = len(radii)
    total_steps = num_layers + 5  # レイヤ + セットアップ + 合成 + 正規化 + ファイナライゼーション
    progress.set_range(total_steps)
    
    try:
        progress.advance(1, f"{num_layers}レイヤでマルチスケール計算を開始")
        progress.advance(1, f"半径: {radii}, 重み: {weights}")

        # 個別レイヤを計算
        layers: List[Float32Array] = []
        for i, r in enumerate(radii):
            try:
                progress.advance(1, f"レイヤ {i+1}/{num_layers} を計算中（半径={r}）")
                
                # 個別レイヤ計算用のサブ進捗レポーターを作成
                # 注意：これは簡略化したアプローチです。より洗練された実装では、
                # 予想される計算時間に比例して進捗を割り当てるかもしれません
                layer = topo_integral(
                    dem,
                    radius=r,
                    set_nan=set_nan,
                    replace_nan=replace_nan,
                    dtype=dtype,
                    progress=None,  # ネストした進捗レポートを避けるためNoneを使用
                )
                layers.append(layer)
            except Exception as e:
                raise ValueError(f"レイヤ {i}（半径={r}）の計算に失敗しました: {e}")

        # 重みでレイヤを合成
        progress.advance(1, "重みでレイヤを合成中")
        try:
            # 精度損失を避けるため累積にfloat64を使用
            composite = np.zeros_like(layers[0], dtype=np.float64)
            weight_sum = 0.0
            
            for w, layer in zip(weights, layers):
                composite += w * layer.astype(np.float64, copy=False)
                weight_sum += w
            
            # 重み合計で正規化
            if weight_sum > 0:
                composite /= weight_sum
            else:
                raise ValueError("総重み合計がゼロです")
                
        except Exception as e:
            raise ValueError(f"レイヤの合成に失敗しました: {e}")

        # オプションの出力正規化
        if normalize:
            progress.advance(1, "出力を[-1, 1]範囲に正規化中")
            try:
                # ロバストな正規化のため99パーセンタイルを使用
                abs_values = np.abs(composite)
                abs_values_finite = abs_values[np.isfinite(abs_values)]
                
                if len(abs_values_finite) > 0:
                    abs99 = float(np.percentile(abs_values_finite, 99.0))
                    abs99 = max(abs99, 1e-9)  # ゼロ除算を回避
                    composite = np.clip(composite / abs99, -1.0, 1.0)
                else:
                    # すべての値が非有限
                    composite.fill(0.0)
                    
            except Exception as e:
                raise ValueError(f"出力の正規化に失敗しました: {e}")
        else:
            progress.advance(1, "正規化をスキップ")

        # 出力データ型に変換
        progress.advance(1, f"結果を出力データ型（{dtype}）に変換中")
        try:
            composite_result = composite.astype(dtype, copy=False)
        except Exception as e:
            raise ValueError(f"結果をデータ型 {dtype} に変換できませんでした: {e}")

        progress.advance(1, "マルチスケール積分計算完了")
        return composite_result, layers
        
    except Exception as e:
        progress.advance(1, f"エラーが発生しました: {str(e)}")
        raise
    finally:
        progress.done()