"""

FujiShader用の *最小限でありながら柔軟な* コマンドライン・インターフェース

"""
from __future__ import annotations

import argparse, json, math, sys, re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from osgeo import gdal, gdalconst

import FujiShader.shader as fs

# -----------------------------------------------------------------------------
# ラスタI/Oヘルパー（GDALの薄いラッパー）
# -----------------------------------------------------------------------------

def read_raster(path: str | Path) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    """適切なエラーハンドリング付きでラスタファイルを読み込む"""
    try:
        ds = gdal.Open(str(path), gdalconst.GA_ReadOnly)
        if ds is None:
            sys.exit(f"[エラー] {path} を開けません - ファイルが存在しないか、有効なラスタではありません")
        
        band = ds.GetRasterBand(1)
        if band is None:
            sys.exit(f"[エラー] {path} からバンド1を読み込めません")
            
        arr = band.ReadAsArray()
        if arr is None:
            sys.exit(f"[エラー] {path} から配列データを読み込めません")
        
        # NoDataValueの処理
        nodata = band.GetNoDataValue()
        if nodata is not None:
            # NoDataValueをNaNに置換
            mask = np.isclose(arr, nodata, equal_nan=False)
            if np.any(mask):
                arr = arr.astype(np.float32)
                arr[mask] = np.nan
            
        arr = arr.astype(np.float32)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()
        
        # 地理変換の検証
        if gt is None or len(gt) != 6:
            print(f"[警告] {path} で無効または欠落した地理変換")
            gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # デフォルト地理変換
            
        return arr, gt, proj
    except Exception as e:
        sys.exit(f"[エラー] {path} の読み込みに失敗: {e}")


def write_raster(path: str | Path, data: np.ndarray, gt: Tuple[float, ...], proj: str) -> None:
    """適切なエラーハンドリング付きでラスタファイルを書き込む"""
    try:
        bands = data.shape[2] if data.ndim == 3 else 1
        h, w = data.shape[:2]
        
        drv = gdal.GetDriverByName("GTiff")
        if drv is None:
            sys.exit("[エラー] GTiffドライバーを取得できません")
            
        # 適切なデータ型を選択
        if data.ndim == 2:
            gdal_dtype = gdal.GDT_Float32
            options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
        else:
            gdal_dtype = gdal.GDT_Byte
            options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "PHOTOMETRIC=RGB"]
        
        ds = drv.Create(str(path), w, h, bands, gdal_dtype, options=options)
        if ds is None:
            sys.exit(f"[エラー] 出力ファイル {path} を作成できません")
            
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        
        if data.ndim == 2:
            band_out = ds.GetRasterBand(1)
            # NaN値に対するNoDataValueを設定
            band_out.SetNoDataValue(np.nan)
            result = band_out.WriteArray(data)
            if result != 0:
                sys.exit(f"[エラー] {path} へのデータ書き込みに失敗")
        else:  # RGB 0‑1 → 0‑255 uint8
            data_u8 = (np.clip(data, 0, 1) * 255).astype(np.uint8)
            for i in range(3):
                result = ds.GetRasterBand(i + 1).WriteArray(data_u8[:, :, i])
                if result != 0:
                    sys.exit(f"[エラー] {path} のバンド{i+1}書き込みに失敗")
        
        ds.FlushCache()
        ds = None
        
    except Exception as e:
        sys.exit(f"[エラー] {path} の書き込みに失敗: {e}")


# ------------------------------------------------------------------
# COGをストリーミング処理する汎用ヘルパー
# ------------------------------------------------------------------
def process_in_tiles(
    input_path: str | Path,
    output_path: str | Path,
    algo_func,
    kwargs: Dict[str, Any],
    tile_size: int = 1024,
    halo: int = 0,
) -> None:
    """DEMを ``tile_size × tile_size`` 窓で順繰りに読み込み、結果を同窓で書き出す"""
    
    try:
        ds_in = gdal.Open(str(input_path), gdalconst.GA_ReadOnly)
        if ds_in is None:
            sys.exit(f"[エラー] {input_path} を開けません")
            
        gt = ds_in.GetGeoTransform()
        proj = ds_in.GetProjectionRef()
        w, h = ds_in.RasterXSize, ds_in.RasterYSize
        band_in = ds_in.GetRasterBand(1)
        
        # NoDataValueの取得
        nodata = band_in.GetNoDataValue()
        
        # 寸法の検証
        if w <= 0 or h <= 0:
            sys.exit(f"[エラー] 無効なラスタ寸法: {w}x{h}")

        drv = gdal.GetDriverByName("GTiff")
        ds_out = drv.Create(
            str(output_path),
            w, h, 1, gdal.GDT_Float32,
            options=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER",
                f"BLOCKXSIZE={min(tile_size, 512)}",  # ブロックサイズを制限
                f"BLOCKYSIZE={min(tile_size, 512)}",
            ],
        )
        if ds_out is None:
            sys.exit(f"[エラー] 出力ファイル {output_path} を作成できません")
            
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(proj)

        b_out = ds_out.GetRasterBand(1)
        b_out.SetNoDataValue(np.nan)  # 出力のNoDataValueを設定
        
        total_tiles = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
        processed_tiles = 0

        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                try:
                    win_w, win_h = min(tile_size, w - x0), min(tile_size, h - y0)

                    # ハロー分拡張して読み込む
                    rx0 = max(0, x0 - halo)
                    ry0 = max(0, y0 - halo)
                    rx1 = min(w, x0 + win_w + halo)
                    ry1 = min(h, y0 + win_h + halo)
                    r_w, r_h = rx1 - rx0, ry1 - ry0

                    dem_tile = band_in.ReadAsArray(rx0, ry0, r_w, r_h)
                    if dem_tile is None:
                        print(f"[警告] ({x0}, {y0})のタイル読み込みに失敗、スキップします")
                        continue
                    
                    # NoDataValueをNaNに変換
                    if nodata is not None:
                        mask = np.isclose(dem_tile, nodata, equal_nan=False)
                        if np.any(mask):
                            dem_tile = dem_tile.astype(np.float32)
                            dem_tile[mask] = np.nan
                        
                    dem_tile = dem_tile.astype(np.float32)

                    # NaN処理オプションの適用（set_nan, replace_nan）
                    set_nan_val = kwargs.get('set_nan')
                    replace_nan_val = kwargs.get('replace_nan')
                    
                    if set_nan_val is not None:
                        set_mask = np.isclose(dem_tile, set_nan_val, equal_nan=False)
                        dem_tile[set_mask] = np.nan
                    
                    # replace_nanは各アルゴリズム内で処理されるため、ここでは処理しない
                    # （アルゴリズムが元のNaN位置を復元するため）

                    res = algo_func(dem_tile, **kwargs)
                    if isinstance(res, tuple):
                        res = res[0]

                    # ハローを除去してタイル中心部だけを書き込む
                    cx0 = x0 - rx0            # 左側ハロー幅（0またはhalo）
                    cy0 = y0 - ry0            # 上側ハロー幅
                    core = res[cy0 : cy0 + win_h, cx0 : cx0 + win_w]

                    result = b_out.WriteArray(core, xoff=x0, yoff=y0)
                    if result != 0:
                        print(f"[警告] ({x0}, {y0})のタイル書き込みに失敗")
                        
                    processed_tiles += 1
                    if processed_tiles % 10 == 0:
                        print(f"[進捗] {processed_tiles}/{total_tiles} タイルを処理済み")
                        
                except Exception as e:
                    print(f"[警告] ({x0}, {y0})のタイル処理でエラー: {e}")
                    continue

        b_out.FlushCache()
        ds_in = ds_out = None
        
    except Exception as e:
        sys.exit(f"[エラー] タイル処理に失敗: {e}")


# ------------------------------------------------------------------
# 文字列を自動でint/float/bool/Noneに変換するユーティリティ
# ------------------------------------------------------------------
def _autocast(val: str) -> Any:
    """文字列を適切な型（int、float、bool、None、またはstr）に変換"""
    if not isinstance(val, str):
        return val
        
    val_lower = val.lower().strip()
    if val_lower in {"true", "false"}:
        return val_lower == "true"
    if val_lower in {"null", "none", ""}:
        return None
    try:
        # 最初にintを試す
        if "." not in val and "e" not in val_lower:
            return int(val)
    except ValueError:
        pass
    try:
        # floatを試す
        return float(val)
    except ValueError:
        return val
        
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_kv_list(s: str) -> Dict[str, Any]:
    """``key=value,key2=value2`` をJSON風の型キャストでdictに解析"""
    out: Dict[str, Any] = {}
    if not s:
        return out
        
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"[警告] 不正なパラメータを無視: {pair}")
            continue
            
        k, _, v = pair.partition("=")
        k = k.strip()
        v = v.strip()
        
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = _autocast(v)  # フォールバックとしてautocastを使用
    return out


def calculate_required_halo(algo: str, kwargs: Dict[str, Any]) -> int:
    """指定されたアルゴリズムとパラメータに必要なハローサイズを計算"""
    if algo == "slope":
        return 1
    elif algo in {"profile_curvature", "plan_curvature", "total_curvature"}:
        # 曲率計算は3x3ウィンドウを使用するため1ピクセルのハローが必要
        return 1
    elif algo in {"metallic_shade", "specular"}:
        # スペキュラ計算では表面法線計算で勾配を使うため1ピクセルのハローが必要
        return 1
    elif algo in {"swiss_shade", "swiss_shade_classic"}:
        # スイスシェーディングはヒルシェード計算で勾配を使うため1ピクセルのハローが必要
        return 1
    elif algo in {"topo_integral", "integral"}:
        return kwargs.get("radius", 8)  # デフォルト8
    elif algo == "multi_scale_integral":
        radii = kwargs.get("radii", [4, 16, 64, 256])  # デフォルト値
        if isinstance(radii, (list, tuple)) and len(radii) > 0:
            return max(radii)
        elif isinstance(radii, int):
            return radii
        else:
            return 64  # フォールバック値
    elif algo in {"topo_boxgauss", "boxgauss"}:
        return kwargs.get("radius", 8)  # デフォルト8
    elif algo == "multi_scale_boxgauss":
        radii = kwargs.get("radii", [4, 16, 64, 256])  # デフォルト値
        if isinstance(radii, (list, tuple)) and len(radii) > 0:
            return max(radii)
        elif isinstance(radii, int):
            return radii
        else:
            return 64  # フォールバック値
    elif algo == "topo_usm":
        # 単一半径のTopoUSMの場合
        radius = kwargs.get("radius", 8)  # デフォルト8
        return radius
    elif algo == "multi_scale_usm":
        # マルチスケールTopoUSMの場合
        radii = kwargs.get("radii", [4, 16, 64, 256])  # デフォルト値
        if isinstance(radii, (list, tuple)) and len(radii) > 0:
            return max(radii)
        elif isinstance(radii, int):
            return radii
        else:
            return 64  # フォールバック値
    elif algo == "warmcool_map":
        # 暖色-寒色マッピングはピクセル単位の処理なのでハロー不要
        return 0
    elif algo in {"skyview_factor", "ambient_occlusion", "positive_openness", "negative_openness"}:
        max_radius = kwargs.get("max_radius")
        cellsize = kwargs.get("cellsize")
        
        if max_radius is None or cellsize is None:
            sys.exit(f"[エラー] {algo} には --max-radius と --cellsize の両方のパラメータが必要です")
        
        # スカラーとタプルのcellsizeの両方を処理
        if isinstance(cellsize, (list, tuple)):
            min_cs = min(cellsize)
        else:
            min_cs = float(cellsize)
            
        return int(np.ceil(max_radius / min_cs))
    else:
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    algos = sorted(fs.list_algorithms().keys())
    p = argparse.ArgumentParser(
        prog="fujishader",
        description="コマンドラインからFujiShaderアルゴリズムを適用します。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"利用可能なアルゴリズム: {', '.join(algos)}"
    )
    p.add_argument("input", help="入力DEM GeoTIFF")
    p.add_argument("output", help="出力ラスタ")
    p.add_argument(
        "--algo",
        choices=algos,
        default="slope",
        help=f"適用するアルゴリズム (デフォルト: slope)",
    )
    p.add_argument(
        "--params",
        metavar="key=val,...",
        help="アルゴリズムに転送される追加のキーワードパラメータ",
    )
    
    # 共通の便利オプション
    p.add_argument("--slope", type=str, help="必要に応じて傾斜ラスタを読み込み")
    p.add_argument("--svf", type=str, help="必要に応じて天空率ラスタを読み込み")
    p.add_argument("--radius", type=int, help="アルゴリズムの半径（ピクセル）")
    p.add_argument("--radii", help="マルチスケールアルゴリズムのカンマ区切りリスト")
    p.add_argument("--weights", help="マルチスケールアルゴリズムのカンマ区切りリスト")
    p.add_argument("--kernel", choices=["disc", "gauss"], help="TopoUSMで使用するカーネルタイプ")
    p.add_argument("--use-fft", action="store_true", dest="use_fft", 
                   help="FFT畳み込みを強制使用（TopoUSM用）")
    p.add_argument("--no-fft", action="store_true", dest="no_fft", 
                   help="空間畳み込みを強制使用（TopoUSM用）")
    p.add_argument("--passes", type=int, help="ボックスフィルタのパス数（デフォルト: 3）")
    p.add_argument("--cellsize", type=float, help="ピクセルサイズ（svf、slope、opennessなど用）")
    p.add_argument("--max-radius", type=float, dest="max_radius", help="svf/openness/aoのスキャン距離")
    p.add_argument("--n-directions", type=int, dest="n_directions", help="svf/opennessの方位セクター数")
    p.add_argument("--n-rays", type=int, dest="n_rays", help="ambient_occlusionのレイ数")
    p.add_argument("--stride", type=int, help="ambient_occlusionのサンプリング間隔")
    p.add_argument("--unit", choices=["degree", "percent"], help="傾斜の単位")
    p.add_argument("--set-nan", type=float, dest="set_nan", 
                   help="この値をNaNに設定（処理開始時）")
    p.add_argument("--replace-nan", type=float, dest="replace_nan", 
                   help="NaN値をこの値で置換（計算用、結果では元のNaN位置を復元）")
    p.add_argument("--normalize", action="store_true", help="マルチスケール結果を正規化")
    p.add_argument("--memory-efficient", action="store_true", dest="memory_efficient",
                   help="メモリ効率モードを有効化（大容量DEM用）")
    
    # warmcool_map専用オプション
    p.add_argument("--slope-ref", type=float, dest="slope_ref", default=None,
                   help="warmcool_map: 最大暗色化に対応する傾斜値（度、デフォルト: 45.0）")
    p.add_argument("--slope-weight", type=float, dest="slope_weight", default=None,
                   help="warmcool_map: 輝度項への傾斜の寄与（0-1、デフォルト: 0.4）")
    p.add_argument("--svf-weight", type=float, dest="svf_weight", default=None,
                   help="warmcool_map: 輝度項への天空率の寄与（0-1、デフォルト: 0.3）")
    p.add_argument("--warm-gain", type=float, dest="warm_gain", default=None,
                   help="warmcool_map: 尾根（正のUSM）に適用される色ゲイン（デフォルト: 0.5）")
    p.add_argument("--cool-gain", type=float, dest="cool_gain", default=None,
                   help="warmcool_map: 谷（負のUSM）に適用される色ゲイン（デフォルト: 0.5）")
    
    p.add_argument("--tile", type=int, metavar="N", help="大容量ファイル用のN×Nピクセルタイルで処理")
    p.add_argument("--verbose", "-v", action="store_true", help="詳細出力を有効化")
    
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)

    # 入力ファイルの検証
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[エラー] 入力ファイルが存在しません: {input_path}")

    # アルゴリズム関数を取得
    try:
        algo_func = fs.list_algorithms()[args.algo]
    except KeyError:
        sys.exit(f"[エラー] アルゴリズム '{args.algo}' はFujiShaderに登録されていません。")

    # アルゴリズム用のkwargsを構築
    kwargs: Dict[str, Any] = {}

    # 補助ラスタを読み込み
    try:
        if args.slope:
            kwargs["slope"], *_ = read_raster(args.slope)
        if args.svf:
            kwargs["svf"], *_ = read_raster(args.svf)
    except Exception as e:
        sys.exit(f"[エラー] 補助ラスタの読み込みに失敗: {e}")

    # --paramsを解析
    if args.params:
        try:
            kwargs.update(parse_kv_list(args.params))
        except Exception as e:
            sys.exit(f"[エラー] --paramsの解析に失敗: {e}")

    # 便利オプション
    if args.radius is not None:
        kwargs["radius"] = args.radius
    if args.radii is not None:
        try:
            kwargs["radii"] = [int(r.strip()) for r in args.radii.split(",") if r.strip()]
        except ValueError as e:
            sys.exit(f"[エラー] 無効な--radii形式: {e}")
    if args.weights is not None:
        try:
            kwargs["weights"] = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
        except ValueError as e:
            sys.exit(f"[エラー] 無効な--weights形式: {e}")
    if args.passes is not None:
        kwargs["passes"] = args.passes
    if args.cellsize is not None:
        kwargs["cellsize"] = args.cellsize
    if args.max_radius is not None:
        kwargs["max_radius"] = args.max_radius
    if args.n_directions is not None:
        kwargs["n_directions"] = args.n_directions
    if args.n_rays is not None:
        kwargs["n_rays"] = args.n_rays
    if args.stride is not None:
        kwargs["stride"] = args.stride
    if args.unit is not None:
        kwargs["unit"] = args.unit
    if args.set_nan is not None:
        kwargs["set_nan"] = args.set_nan
    if args.replace_nan is not None:
        kwargs["replace_nan"] = args.replace_nan
    if args.normalize:
        kwargs["normalize"] = True
    if args.memory_efficient:
        kwargs["memory_efficient"] = True

    # warmcool_map専用オプション
    if hasattr(args, 'slope_ref') and args.slope_ref is not None:
        kwargs["slope_ref"] = args.slope_ref
    if hasattr(args, 'slope_weight') and args.slope_weight is not None:
        kwargs["slope_weight"] = args.slope_weight
    if hasattr(args, 'svf_weight') and args.svf_weight is not None:
        kwargs["svf_weight"] = args.svf_weight
    if hasattr(args, 'warm_gain') and args.warm_gain is not None:
        kwargs["warm_gain"] = args.warm_gain
    if hasattr(args, 'cool_gain') and args.cool_gain is not None:
        kwargs["cool_gain"] = args.cool_gain

    # 未知のオプションを処理
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            parser.error(f"認識されない引数 {tok!r}")
        key = tok.lstrip("-")
        
        # --foo=bar形式
        if "=" in key:
            k, v = key.split("=", 1)
            kwargs[k.replace("-", "_")] = _autocast(v)
            i += 1
        # --foo 123形式
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            kwargs[key.replace("-", "_")] = _autocast(unknown[i + 1])
            i += 2
        # --flag形式
        else:
            kwargs[key.replace("-", "_")] = True
            i += 1

    # 必要なハローを計算
    halo = calculate_required_halo(args.algo, kwargs)
    
    if args.verbose:
        print(f"[情報] アルゴリズム: {args.algo}")
        print(f"[情報] パラメータ: {kwargs}")
        print(f"[情報] 必要なハロー: {halo}")
        if args.tile:
            print(f"[情報] タイルサイズ: {args.tile}x{args.tile}")
        if args.memory_efficient:
            print(f"[情報] メモリ効率モード: 有効")

    # タイルサイズ vs ハローの検証
    if args.tile and args.tile <= 2 * halo:
        sys.exit(f"[エラー] タイルサイズ（{args.tile}）は 2 * ハロー（{2 * halo}）より大きくする必要があります")

    # 進捗レポーターの設定
    if args.verbose:
        try:
            from FujiShader.core.progress import TqdmProgress
            progress_reporter = TqdmProgress()
        except ImportError:
            from FujiShader.core.progress import NullProgress
            progress_reporter = NullProgress()
            print("[警告] tqdmが利用できません。進捗表示は無効です。")
    else:
        from FujiShader.core.progress import NullProgress
        progress_reporter = NullProgress()

    # データを処理
    try:
        if args.tile:
            if args.verbose:
                print("[情報] タイルベース処理を使用")
            from FujiShader.core.stream import stream_tiles
            
            # stream_tilesに進捗レポーターを追加
            stream_tiles(
                args.input,
                args.output,
                algo_func,
                kwargs,
                tile_size=args.tile,
                halo=halo,
                progress=progress_reporter,
            )
        else:
            if args.verbose:
                print("[情報] ファイル全体をメモリに読み込み")
            
            # 進捗レポーターをkwargsに追加
            kwargs["progress"] = progress_reporter
            
            dem, gt, proj = read_raster(args.input)
            result = algo_func(dem, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
            write_raster(args.output, result, gt, proj)
        
        print(f"[OK] 保存完了 -> {args.output}")
        
    except Exception as e:
        sys.exit(f"[エラー] 処理に失敗: {e}")

    if args.kernel is not None:
        kwargs["kernel"] = args.kernel
    if args.use_fft:
        kwargs["use_fft"] = True
    if args.no_fft:
        kwargs["use_fft"] = False
    if args.use_fft and args.no_fft:
        sys.exit("[エラー] --use-fftと--no-fftは同時に指定できません")

if __name__ == "__main__":  # pragma: no cover
    main()