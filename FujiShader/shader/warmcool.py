"""
FujiShader.shader.warmcool
==========================

人間の視覚科学に基づいた *暖色尾根 / 寒色谷* パレットによる地形の色付け。
完全に方位角非依存で、ギガバイトサイズのCOGにも対応。

関数 ``warmcool_map`` は **TopoUSM** (符号付き局所起伏) と
オプションで **slope** (表面粗さ) と **sky-view factor** (環境遮蔽) を受け取ります。
出力は ``[0, 1]`` 範囲の ``(H, W, 3)`` RGB配列です。

アルゴリズムは以下の設計規則に従います
----------------------------------------
* **色相 – TopoUSM符号**: 正 → 暖色 (赤-黄), 負 → 寒色 (シアン-青)
* **明度 – 輝度ベース**: ``1`` から開始し、slope と SVF の重み付きで減算して
  急斜面や遮蔽された領域を暗く表現
* **彩度 – |TopoUSM|**: 局所起伏が大きいほど色の飽和度が高くなる

すべての数学処理はベクトル化されたNumPy演算。NumPy以外の外部依存関係なし。
ラスタの読み書きは :pymod:`FujiShader.core.io` で処理されます。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from ..core.progress import ProgressReporter, NullProgress

__all__ = ["warmcool_map"]

# -----------------------------------------------------------------------------
# ヘルパーユーティリティ
# -----------------------------------------------------------------------------

_P99_EPS: float = 1e-6  # 平坦なデータでのゼロ除算を防ぐ


def _normalize_signed(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """符号付きデータを *pct* パーセンタイルで ±1 に正規化する"""
    hi = float(np.nanpercentile(np.abs(arr), pct))
    if hi < _P99_EPS:
        hi = 1.0  # ほぼ平坦なラスタのフォールバック
    return np.clip(arr / hi, -1.0, 1.0)


def _normalize_scalar(arr: np.ndarray, ref: float) -> np.ndarray:
    """正のスカラーフィールドを *ref* に対して 0-1 に正規化する"""
    return np.clip(arr / ref, 0.0, 1.0)


def _process_nan_values(
    arr: np.ndarray,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    配列のNaN値処理を行う
    
    パラメータ
    ----------
    arr : ndarray
        入力配列
    set_nan : float, optional
        この値をNaNに設定（処理開始時）
    replace_nan : float, optional
        NaN値をこの値で一時的に置換（計算用、結果では元NaN位置を復元）
        
    戻り値
    -------
    processed_arr : ndarray
        処理済み配列
    original_nan_mask : ndarray
        元のNaN位置のマスク（結果復元用）
    """
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(arr)
    
    # コピーを作成して処理
    processed_arr = arr.copy()
    
    # set_nan処理：指定値をNaNに設定
    if set_nan is not None:
        set_nan_mask = np.isclose(processed_arr, set_nan, equal_nan=False)
        if np.any(set_nan_mask):
            processed_arr[set_nan_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | set_nan_mask
    
    # replace_nan処理：NaN値を一時的に置換（計算用）
    if replace_nan is not None:
        current_nan_mask = np.isnan(processed_arr)
        if np.any(current_nan_mask):
            processed_arr[current_nan_mask] = replace_nan
    
    return processed_arr, original_nan_mask


# -----------------------------------------------------------------------------
# 公開API
# -----------------------------------------------------------------------------

def warmcool_map(
    usm: np.ndarray,
    *,
    slope: Optional[np.ndarray] = None,
    svf: Optional[np.ndarray] = None,
    slope_ref: float = 45.0,
    slope_weight: float = 0.4,
    svf_weight: float = 0.3,
    warm_gain: float = 0.5,
    cool_gain: float = 0.5,
    set_nan: Optional[float] = None,
    replace_nan: Optional[float] = None,
    dtype=np.float32,
    progress: Optional[ProgressReporter] = None,
) -> np.ndarray:
    """地形レイヤーから暖色-寒色RGB起伏画像を生成する

    パラメータ
    ----------
    usm : ndarray (H, W)
        TopoUSMからの符号付き局所起伏。正 → 尾根、負 → 谷
    slope : ndarray (H, W), optional
        度またはパーセントでの傾斜。急斜面を暗くする。``None`` でスキップ
    svf : ndarray (H, W), optional
        天空率 [0–1]。遮蔽された地形を暗くする。``None`` でスキップ
    slope_ref : float, default 45.0
        最大暗色化に対応する傾斜値（度）
    slope_weight : float, default 0.4
        輝度項への *slope* の寄与 (0–1)
    svf_weight : float, default 0.3
        輝度項への *svf* の寄与 (0–1)
    warm_gain : float, default 0.5
        尾根（正のUSM）に適用される色ゲイン
    cool_gain : float, default 0.5
        谷（負のUSM）に適用される色ゲイン
    set_nan : float, optional
        この値をNaNに設定（処理開始時）
    replace_nan : float, optional
        NaN値をこの値で置換（計算用、結果では元のNaN位置を復元）
    dtype : numpy dtype, default ``np.float32``
        出力配列のデータ型
    progress : ProgressReporter, optional
        計算進捗を追跡する進捗レポーター

    戻り値
    -------
    rgb : ndarray (H, W, 3)
        範囲 ``[0, 1]`` のRGB画像
    """
    if usm.ndim != 2:
        raise ValueError("usm は 2次元配列（高さ × 幅）である必要があります")

    # 進捗レポーターの初期化
    if progress is None:
        progress = NullProgress()
    
    # オプション入力に基づいて総ステップ数を計算
    total_steps = 8  # ベースステップ: USM正規化、NaN処理、輝度初期化、暖色マスク、寒色マスク、スタック、クランプ、完了
    if slope is not None:
        total_steps += 1  # slope処理
    if svf is not None:
        total_steps += 1  # svf処理
    
    progress.set_range(total_steps)
    current_step = 0

    # ------------------------------------------------------------------
    # 1. 入力レイヤーの正規化とNaN処理
    # ------------------------------------------------------------------
    progress.advance(1, "USMデータの正規化中...")
    current_step += 1
    
    # USMのNaN処理
    usm_processed, usm_nan_mask = _process_nan_values(usm.astype(dtype), set_nan, replace_nan)
    usm_n = _normalize_signed(usm_processed)  # ±1
    
    # 元のNaN位置を復元（正規化後）
    usm_n[usm_nan_mask] = np.nan

    # ベース輝度を1（白）から開始
    progress.advance(1, "輝度の初期化中...")
    current_step += 1
    L = np.ones_like(usm_n, dtype=dtype)

    # slope寄与（オプション）
    if slope is not None:
        progress.advance(1, "傾斜データの処理中...")
        current_step += 1
        
        # slopeのNaN処理
        slope_processed, slope_nan_mask = _process_nan_values(slope.astype(dtype), set_nan, replace_nan)
        slope_n = _normalize_scalar(slope_processed, slope_ref)  # 0–1
        
        # 元のNaN位置を復元
        slope_n[slope_nan_mask] = np.nan
        
        # NaN値のある場所では寄与を0にする
        slope_contribution = slope_weight * slope_n
        slope_contribution = np.nan_to_num(slope_contribution, nan=0.0)
        L = L - slope_contribution

    # 天空率寄与（オプション）
    if svf is not None:
        progress.advance(1, "天空率の処理中...")
        current_step += 1
        
        # SVFのNaN処理
        svf_processed, svf_nan_mask = _process_nan_values(svf.astype(dtype), set_nan, replace_nan)
        svf_n = np.clip(svf_processed, 0.0, 1.0)
        
        # 元のNaN位置を復元
        svf_n[svf_nan_mask] = np.nan
        
        # NaN値のある場所では寄与を0にする
        svf_contribution = svf_weight * (1.0 - svf_n)
        svf_contribution = np.nan_to_num(svf_contribution, nan=0.0)
        L = L - svf_contribution

    # 輝度を [0,1] にクランプ
    L = np.clip(L, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 2. RGBチャンネルの構築
    # ------------------------------------------------------------------
    progress.advance(1, "RGBチャンネルの初期化中...")
    current_step += 1
    R = L.copy()
    G = L.copy()
    B = L.copy()

    # 暖色尾根（USM > 0）
    progress.advance(1, "暖色尾根の処理中...")
    current_step += 1
    warm_mask = (usm_n > 0) & (~np.isnan(usm_n))
    if warm_mask.any():
        usm_pos = usm_n[warm_mask]
        R[warm_mask] += warm_gain * usm_pos
        G[warm_mask] += warm_gain * 0.5 * usm_pos
        B[warm_mask] -= warm_gain * 0.5 * usm_pos

    # 寒色谷（USM < 0）
    progress.advance(1, "寒色谷の処理中...")
    current_step += 1
    cool_mask = (usm_n < 0) & (~np.isnan(usm_n))
    if cool_mask.any():
        usm_neg_abs = np.abs(usm_n[cool_mask])
        R[cool_mask] -= cool_gain * 0.5 * usm_neg_abs
        G[cool_mask] += cool_gain * 0.25 * usm_neg_abs
        B[cool_mask] += cool_gain * usm_neg_abs

    # ------------------------------------------------------------------
    # 3. スタックとクランプ
    # ------------------------------------------------------------------
    progress.advance(1, "RGB画像の最終化中...")
    current_step += 1
    
    # RGBをスタック
    rgb = np.stack((R, G, B), axis=-1)
    
    # 値を [0, 1] にクランプ
    rgb = np.clip(rgb, 0.0, 1.0).astype(dtype)
    
    # 元のUSMでNaNだった位置をRGBでもNaNにする
    nan_mask_3d = np.repeat(usm_nan_mask[..., np.newaxis], 3, axis=-1)
    rgb[nan_mask_3d] = np.nan
    
    progress.done()
    return rgb