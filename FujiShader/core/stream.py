"""
FujiShader.core.stream
----------------------

巨大な GeoTIFF / COG を「タイル単位で読み込み→処理→書き込み」する
共通ユーティリティ。

Parameters
----------
in_path      : 入力ラスタ (GeoTIFF / COG / 何でも GDAL が読めれば可)
out_path     : 出力 GeoTIFF
algo         : コールバック関数 (tile, **kwargs) -> ndarray もしくは (ndarray, carry)
kwargs       : algo へ渡す追加キーワード
tile_size    : 1 辺のピクセル数 (COG なら 512, 1024, 2048 など)
halo         : 各タイルの四方に余分に読み込むピクセル幅
carry_mode   : True にすると algo は (result, carry) を返す
               y 方向ストライプ境界で carry を受け渡す (integral 用)
gdt_out      : 出力 GDAL データ型 (既定: gdal.GDT_Float32)
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Union, Tuple

import numpy as np
from osgeo import gdal, gdalconst


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
) -> None:
    """
    in_path : 読み込むラスタファイルパス (GeoTIFF / COG / GDAL 対応ラスタ)
    out_path: 出力先 GeoTIFF パス
    algo     : (tile: np.ndarray, ..., _stream_state: np.ndarray | None) -> np.ndarray もしくは (np.ndarray, np.ndarray)
               ※carry_mode=False のときは (_stream_state は渡されず、返り値は単一の np.ndarray)
               ※carry_mode=True のときは  
                  algo(tile, _stream_state=prev_prefix, **kwargs)  
                  が  (res_tile: np.ndarray, next_prefix: np.ndarray) を返す想定
    kwargs   : algo に追加で渡すキーワード引数
    tile_size: タイルサイズ（ピクセル単位）
    halo     : タイルごとに「上下左右に何ピクセル余分に読み込むか」
    carry_mode: True のときアルゴリズムに縦ストライプの carry を渡す
    gdt_out  : 出力データ型（GDAL の GDT_* 定数）
    """
    kwargs = kwargs or {}

    # パラメータ検証
    if halo >= tile_size:
        raise ValueError(f"halo ({halo}) should be smaller than tile_size ({tile_size})")
    
    if tile_size <= 0:
        raise ValueError(f"tile_size ({tile_size}) must be positive")

    # ---------------- GDAL I/O 初期化 ----------------
    ds_in = None
    ds_out = None
    
    try:
        ds_in = gdal.Open(str(in_path), gdalconst.GA_ReadOnly)
        if ds_in is None:
            raise RuntimeError(f"[stream] Cannot open {in_path}")

        w, h = ds_in.RasterXSize, ds_in.RasterYSize
        gt, proj = ds_in.GetGeoTransform(), ds_in.GetProjection()
        band_in = ds_in.GetRasterBand(1)

        if band_in is None:
            raise RuntimeError(f"[stream] Cannot access band 1 of {in_path}")

        # 出力ファイルを GeoTIFF (タイル付き) として作成
        drv = gdal.GetDriverByName("GTiff")
        if drv is None:
            raise RuntimeError("[stream] GTiff driver not available")
            
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
                f"BLOCKXSIZE={tile_size}",
                f"BLOCKYSIZE={tile_size}",
            ],
        )
        if ds_out is None:
            raise RuntimeError(f"[stream] Cannot create output {out_path}")
            
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(proj)
        band_out = ds_out.GetRasterBand(1)

        if band_out is None:
            raise RuntimeError(f"[stream] Cannot access output band")

        # carry_mode=True のとき、x タイルごとに「縦方向の累積結果 (1D 配列)」を保持
        carry_state: Dict[int, np.ndarray] = {}

        # ---------------- メインループ ----------------
        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                win_w = min(tile_size, w - x0)
                win_h = min(tile_size, h - y0)

                # --- halo を含めた読み取り位置を計算 ---
                rx0 = max(0, x0 - halo)
                ry0 = max(0, y0 - halo)
                rx1 = min(w, x0 + win_w + halo)
                ry1 = min(h, y0 + win_h + halo)
                r_w = rx1 - rx0
                r_h = ry1 - ry0

                # タイル＋halo 部分を一括読み込み
                tile = band_in.ReadAsArray(rx0, ry0, r_w, r_h)
                if tile is None:
                    raise RuntimeError(f"[stream] Failed to read tile at ({rx0}, {ry0}, {r_w}, {r_h})")
                
                # float32 変換
                tile = tile.astype(np.float32, copy=False)

                ix = x0 // tile_size  # x 方向のタイル番号 (0, 1, 2, ...)

                # --- アルゴリズムを呼び出す (carry_mode の有無で分岐) ---
                try:
                    if carry_mode:
                        prev_prefix = carry_state.get(ix)
                        # algo は (res_tile, next_prefix) を返す想定
                        result = algo(tile, _stream_state=prev_prefix, **kwargs)
                        if not isinstance(result, tuple) or len(result) != 2:
                            raise ValueError("[stream] algo must return (result, carry) tuple in carry_mode")
                        res, next_prefix = result
                        carry_state[ix] = next_prefix
                    else:
                        res = algo(tile, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"[stream] Algorithm failed at tile ({x0}, {y0}): {e}") from e

                # multi_scale 系は (composite, layers) を返すので、composite (=0番目) を使う
                if isinstance(res, tuple):
                    res = res[0]

                # 結果が numpy 配列であることを確認
                if not isinstance(res, np.ndarray):
                    raise TypeError(f"[stream] Algorithm result must be numpy array, got {type(res)}")

                # --- halo 部分を切り落とし、中心の tile_size×tile_size 部分だけ書き戻す ---
                cx0 = x0 - rx0  # tile の左端にあたるインデックス (halo があると 0 > cx0)
                cy0 = y0 - ry0  # tile の上端にあたるインデックス
                
                # 境界チェック
                if cy0 + win_h > res.shape[0] or cx0 + win_w > res.shape[1]:
                    raise RuntimeError(f"[stream] Result array size mismatch at tile ({x0}, {y0})")
                
                core = res[cy0 : cy0 + win_h, cx0 : cx0 + win_w]
                
                # 書き込み実行
                write_result = band_out.WriteArray(core, xoff=x0, yoff=y0)
                if write_result != 0:  # GDAL では 0 が成功
                    raise RuntimeError(f"[stream] Failed to write tile at ({x0}, {y0})")

        # キャッシュをフラッシュ
        band_out.FlushCache()

    finally:
        # リソースを確実に解放
        if ds_in is not None:
            ds_in = None
        if ds_out is not None:
            ds_out = None