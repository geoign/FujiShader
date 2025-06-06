"""
FujiShader.core.stream
----------------------

巨大なGeoTIFF / COGを「タイル単位で読み込み→処理→書き込み」する
共通ユーティリティ。進捗レポーター統合付き。

Parameters
----------
in_path      : 入力ラスタ (GeoTIFF / COG / GDALが読めるもの)
out_path     : 出力GeoTIFF
algo         : コールバック関数 (tile, **kwargs) -> ndarray もしくは (ndarray, carry)
kwargs       : algo へ渡す追加キーワード
tile_size    : 1辺のピクセル数 (COGなら512, 1024, 2048など)
halo         : 各タイルの四方に余分に読み込むピクセル幅
carry_mode   : True にするとalgoは(result, carry)を返す
               y方向ストライプ境界でcarryを受け渡す (integral用)
gdt_out      : 出力GDALデータ型 (既定: gdal.GDT_Float32)
progress     : 進捗レポーター (オプション)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Union, Tuple

import numpy as np
from osgeo import gdal, gdalconst

from FujiShader.core.progress import ProgressReporter, NullProgress


def _process_tile_nan_values(
    tile: np.ndarray,
    nodata_value: float | None,
    set_nan: float | None = None,
    replace_nan: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    タイルのNaN値処理を行う
    
    パラメータ
    ----------
    tile : ndarray
        入力タイル
    nodata_value : float or None
        GDALのNoDataValue
    set_nan : float, optional
        この値をNaNに設定（処理開始時）
    replace_nan : float, optional
        NaN値をこの値で一時的に置換（計算用、結果では元NaN位置を復元）
        
    戻り値
    -------
    processed_tile : ndarray
        処理済みタイル
    original_nan_mask : ndarray
        元のNaN位置のマスク（結果復元用）
    """
    # float32に変換
    processed_tile = tile.astype(np.float32, copy=True)
    
    # 元のNaN位置を記録
    original_nan_mask = np.isnan(processed_tile)
    
    # NoDataValueをNaNに変換
    if nodata_value is not None and not np.isnan(nodata_value):
        nodata_mask = np.isclose(processed_tile, nodata_value, equal_nan=False)
        if np.any(nodata_mask):
            processed_tile[nodata_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | nodata_mask
    
    # set_nan処理：指定値をNaNに設定
    if set_nan is not None:
        set_nan_mask = np.isclose(processed_tile, set_nan, equal_nan=False)
        if np.any(set_nan_mask):
            processed_tile[set_nan_mask] = np.nan
            # マスクを更新
            original_nan_mask = original_nan_mask | set_nan_mask
    
    # replace_nan処理：NaN値を一時的に置換（計算用）
    if replace_nan is not None:
        current_nan_mask = np.isnan(processed_tile)
        if np.any(current_nan_mask):
            processed_tile[current_nan_mask] = replace_nan
    
    return processed_tile, original_nan_mask


def stream_tiles(
    in_path: str | bytes,
    out_path: str | bytes,
    algo: Callable[..., Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    kwargs: Dict[str, Any] | None = None,
    *,
    tile_size: int = 1024,
    halo: int = 0,
    carry_mode: bool = False,
    gdt_out: int = gdal.GDT_Float32,
    progress: ProgressReporter | None = None,
) -> None:
    """
    in_path : 読み込むラスタファイルパス (GeoTIFF / COG / GDAL対応ラスタ)
    out_path: 出力先GeoTIFFパス
    algo     : (tile: np.ndarray, ..., _stream_state: np.ndarray | None) -> np.ndarray もしくは (np.ndarray, np.ndarray)
               ※carry_mode=False のときは (_stream_stateは渡されず、返り値は単一のnp.ndarray)
               ※carry_mode=True のときは  
                  algo(tile, _stream_state=prev_prefix, **kwargs)  
                  が  (res_tile: np.ndarray, next_prefix: np.ndarray) を返す想定
    kwargs   : algoに追加で渡すキーワード引数
    tile_size: タイルサイズ（ピクセル単位）
    halo     : タイルごとに「上下左右に何ピクセル余分に読み込むか」
    carry_mode: True のときアルゴリズムに縦ストライプのcarryを渡す
    gdt_out  : 出力データ型（GDALのGDT_*定数）
    progress : 進捗レポーター実装（オプション）
    """
    kwargs = kwargs or {}
    progress = progress or NullProgress()

    # パラメータ検証
    if halo >= tile_size:
        raise ValueError(f"halo ({halo}) はtile_size ({tile_size})より小さくする必要があります")
    
    if tile_size <= 0:
        raise ValueError(f"tile_size ({tile_size})は正の値である必要があります")

    # NaN処理パラメータを抽出
    set_nan = kwargs.get('set_nan')
    replace_nan = kwargs.get('replace_nan')

    # ---------------- GDAL I/O 初期化 ----------------
    ds_in = None
    ds_out = None
    
    try:
        ds_in = gdal.Open(str(in_path), gdalconst.GA_ReadOnly)
        if ds_in is None:
            raise RuntimeError(f"[stream] {in_path}を開けません")

        w, h = ds_in.RasterXSize, ds_in.RasterYSize
        gt, proj = ds_in.GetGeoTransform(), ds_in.GetProjection()
        band_in = ds_in.GetRasterBand(1)

        if band_in is None:
            raise RuntimeError(f"[stream] {in_path}のバンド1にアクセスできません")

        # NoDataValueを取得
        nodata_value = band_in.GetNoDataValue()
        
        # 出力ファイルをGeoTIFF (タイル付き)として作成
        drv = gdal.GetDriverByName("GTiff")
        if drv is None:
            raise RuntimeError("[stream] GTiffドライバーが利用できません")
            
        ds_out = drv.Create(
            str(out_path),
            w,
            h,
            1,
            gdt_out,
            options=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER",
                f"BLOCKXSIZE={min(tile_size, 512)}",  # ブロックサイズを制限
                f"BLOCKYSIZE={min(tile_size, 512)}",
            ],
        )
        if ds_out is None:
            raise RuntimeError(f"[stream] 出力ファイル{out_path}を作成できません")
            
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(proj)
        band_out = ds_out.GetRasterBand(1)

        if band_out is None:
            raise RuntimeError(f"[stream] 出力バンドにアクセスできません")

        # 出力にもNoDataValueを設定（NaNを使用）
        band_out.SetNoDataValue(np.nan)

        # carry_mode=Trueのとき、xタイルごとに「縦方向の累積結果(1D配列)」を保持
        carry_state: Dict[int, np.ndarray] = {}

        # 総タイル数を計算し、進捗レポーターを設定
        total_tiles = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
        progress.set_range(total_tiles)
        processed_tiles = 0

        # ---------------- メインループ ----------------
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                win_w = min(tile_size, w - x0)
                win_h = min(tile_size, h - y0)

                # --- haloを含めた読み取り位置を計算 ---
                rx0 = max(0, x0 - halo)
                ry0 = max(0, y0 - halo)
                rx1 = min(w, x0 + win_w + halo)
                ry1 = min(h, y0 + win_h + halo)
                r_w = rx1 - rx0
                r_h = ry1 - ry0

                # タイル＋halo部分を一括読み込み
                tile = band_in.ReadAsArray(rx0, ry0, r_w, r_h)
                if tile is None:
                    raise RuntimeError(f"[stream] ({rx0}, {ry0}, {r_w}, {r_h})のタイル読み込みに失敗")
                
                # NaN値処理
                tile_processed, original_nan_mask = _process_tile_nan_values(
                    tile, nodata_value, set_nan, replace_nan
                )

                ix = x0 // tile_size  # x方向のタイル番号 (0, 1, 2, ...)
                iy = y0 // tile_size  # y方向のタイル番号

                # 進捗表示用テキスト
                progress_text = f"タイル処理中: 行{iy + 1}列{ix + 1} [{processed_tiles + 1}/{total_tiles}]"

                # --- アルゴリズムを呼び出す (carry_modeの有無で分岐) ---
                try:
                    if carry_mode:
                        prev_prefix = carry_state.get(ix)
                        # 進捗レポーターとNaN処理パラメータをalgoに渡さない（既に前処理済み）
                        algo_kwargs = {k: v for k, v in kwargs.items() 
                                      if k not in ('progress', 'set_nan', 'replace_nan')}
                        # algoは(res_tile, next_prefix)を返す想定
                        result = algo(tile_processed, _stream_state=prev_prefix, **algo_kwargs)
                        if not isinstance(result, tuple) or len(result) != 2:
                            raise ValueError("[stream] carry_modeでalgoは(result, carry)タプルを返す必要があります")
                        res, next_prefix = result
                        carry_state[ix] = next_prefix
                    else:
                        # 進捗レポーターとNaN処理パラメータをalgoに渡さない（既に前処理済み）
                        algo_kwargs = {k: v for k, v in kwargs.items() 
                                      if k not in ('progress', 'set_nan', 'replace_nan')}
                        res = algo(tile_processed, **algo_kwargs)
                except Exception as e:
                    raise RuntimeError(f"[stream] タイル(行{iy+1},列{ix+1})でアルゴリズムが失敗: {e}") from e

                # multi_scale系は(composite, layers)を返すので、composite (=0番目)を使う
                if isinstance(res, tuple):
                    res = res[0]

                # 結果がnumpy配列であることを確認
                if not isinstance(res, np.ndarray):
                    raise TypeError(f"[stream] アルゴリズムの結果はnumpy配列である必要があります。{type(res)}が返されました")

                # --- halo部分を切り落とし、中心のtile_size×tile_size部分だけ書き戻す ---
                cx0 = x0 - rx0  # tileの左端にあたるインデックス (haloがあると0 > cx0)
                cy0 = y0 - ry0  # tileの上端にあたるインデックス
                
                # 境界チェック
                if cy0 + win_h > res.shape[0] or cx0 + win_w > res.shape[1]:
                    raise RuntimeError(f"[stream] タイル(行{iy+1},列{ix+1})で結果配列サイズが不一致")
                
                core = res[cy0 : cy0 + win_h, cx0 : cx0 + win_w]
                
                # 元のNaN位置を復元（haloサイズを考慮）
                original_nan_core = original_nan_mask[cy0 : cy0 + win_h, cx0 : cx0 + win_w]
                core[original_nan_core] = np.nan
                
                # 書き込み実行
                write_result = band_out.WriteArray(core, xoff=x0, yoff=y0)
                if write_result != 0:  # GDALでは0が成功
                    raise RuntimeError(f"[stream] タイル(行{iy+1},列{ix+1})の書き込みに失敗")

                processed_tiles += 1
                progress.advance(step=1, text=progress_text)

        # キャッシュをフラッシュ
        band_out.FlushCache()
        progress.done()

    finally:
        # リソースを確実に解放
        if ds_in is not None:
            ds_in = None
        if ds_out is not None:
            ds_out = None