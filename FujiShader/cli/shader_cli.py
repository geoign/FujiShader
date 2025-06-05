"""

A *minimal yet flexible* command‑line interface for FujiShader.

"""
from __future__ import annotations

import argparse, json, math, sys, re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from osgeo import gdal, gdalconst

import FujiShader.shader as fs

# -----------------------------------------------------------------------------
# Raster I/O helpers (thin wrapper around GDAL)
# -----------------------------------------------------------------------------

def read_raster(path: str | Path) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    """Read raster file with proper error handling."""
    try:
        ds = gdal.Open(str(path), gdalconst.GA_ReadOnly)
        if ds is None:
            sys.exit(f"[Error] Cannot open {path} - file may not exist or is not a valid raster")
        
        band = ds.GetRasterBand(1)
        if band is None:
            sys.exit(f"[Error] Cannot read band 1 from {path}")
            
        arr = band.ReadAsArray()
        if arr is None:
            sys.exit(f"[Error] Cannot read array data from {path}")
            
        arr = arr.astype(np.float32)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()
        
        # Validate geotransform
        if gt is None or len(gt) != 6:
            print(f"[Warning] Invalid or missing geotransform in {path}")
            gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)  # Default geotransform
            
        return arr, gt, proj
    except Exception as e:
        sys.exit(f"[Error] Failed to read {path}: {e}")


def write_raster(path: str | Path, data: np.ndarray, gt: Tuple[float, ...], proj: str) -> None:
    """Write raster file with proper error handling."""
    try:
        bands = data.shape[2] if data.ndim == 3 else 1
        h, w = data.shape[:2]
        
        drv = gdal.GetDriverByName("GTiff")
        if drv is None:
            sys.exit("[Error] Cannot get GTiff driver")
            
        # Choose appropriate data type
        if data.ndim == 2:
            gdal_dtype = gdal.GDT_Float32
            options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
        else:
            gdal_dtype = gdal.GDT_Byte
            options = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "PHOTOMETRIC=RGB"]
        
        ds = drv.Create(str(path), w, h, bands, gdal_dtype, options=options)
        if ds is None:
            sys.exit(f"[Error] Cannot create output file {path}")
            
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        
        if data.ndim == 2:
            result = ds.GetRasterBand(1).WriteArray(data)
            if result != 0:
                sys.exit(f"[Error] Failed to write data to {path}")
        else:  # RGB 0‑1 → 0‑255 uint8
            data_u8 = (np.clip(data, 0, 1) * 255).astype(np.uint8)
            for i in range(3):
                result = ds.GetRasterBand(i + 1).WriteArray(data_u8[:, :, i])
                if result != 0:
                    sys.exit(f"[Error] Failed to write band {i+1} to {path}")
        
        ds.FlushCache()
        ds = None
        
    except Exception as e:
        sys.exit(f"[Error] Failed to write {path}: {e}")


# ------------------------------------------------------------------
# ここから追加：COG をストリーミング処理する汎用ヘルパー
# ------------------------------------------------------------------
def process_in_tiles(
    input_path: str | Path,
    output_path: str | Path,
    algo_func,
    kwargs: Dict[str, Any],
    tile_size: int = 1024,
    halo: int = 0,
) -> None:
    """DEM を ``tile_size × tile_size`` 窓で順繰りに読み込み、結果を同窓で書き出す。"""
    
    try:
        ds_in = gdal.Open(str(input_path), gdalconst.GA_ReadOnly)
        if ds_in is None:
            sys.exit(f"[Error] Cannot open {input_path}")
            
        gt = ds_in.GetGeoTransform()
        proj = ds_in.GetProjectionRef()
        w, h = ds_in.RasterXSize, ds_in.RasterYSize
        
        # Validate dimensions
        if w <= 0 or h <= 0:
            sys.exit(f"[Error] Invalid raster dimensions: {w}x{h}")

        drv = gdal.GetDriverByName("GTiff")
        ds_out = drv.Create(
            str(output_path),
            w, h, 1, gdal.GDT_Float32,
            options=[
                "COMPRESS=LZW",
                "TILED=YES",
                "BIGTIFF=IF_SAFER",
                f"BLOCKXSIZE={min(tile_size, 512)}",  # Limit block size
                f"BLOCKYSIZE={min(tile_size, 512)}",
            ],
        )
        if ds_out is None:
            sys.exit(f"[Error] Cannot create output file {output_path}")
            
        ds_out.SetGeoTransform(gt)
        ds_out.SetProjection(proj)

        b_in, b_out = ds_in.GetRasterBand(1), ds_out.GetRasterBand(1)
        
        total_tiles = ((h + tile_size - 1) // tile_size) * ((w + tile_size - 1) // tile_size)
        processed_tiles = 0

        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                try:
                    win_w, win_h = min(tile_size, w - x0), min(tile_size, h - y0)

                    # ── ハローぶん拡張して読み込む ───────────────────
                    rx0 = max(0, x0 - halo)
                    ry0 = max(0, y0 - halo)
                    rx1 = min(w, x0 + win_w + halo)
                    ry1 = min(h, y0 + win_h + halo)
                    r_w, r_h = rx1 - rx0, ry1 - ry0

                    dem_tile = b_in.ReadAsArray(rx0, ry0, r_w, r_h)
                    if dem_tile is None:
                        print(f"[Warning] Failed to read tile at ({x0}, {y0}), skipping")
                        continue
                        
                    dem_tile = dem_tile.astype(np.float32)

                    res = algo_func(dem_tile, **kwargs)
                    if isinstance(res, tuple):
                        res = res[0]

                    # ── ハローを除去してタイル中心部だけを書き込む ──
                    cx0 = x0 - rx0            # 左側ハロー幅（0 または halo）
                    cy0 = y0 - ry0            # 上側ハロー幅
                    core = res[cy0 : cy0 + win_h, cx0 : cx0 + win_w]

                    result = b_out.WriteArray(core, xoff=x0, yoff=y0)
                    if result != 0:
                        print(f"[Warning] Failed to write tile at ({x0}, {y0})")
                        
                    processed_tiles += 1
                    if processed_tiles % 10 == 0:
                        print(f"[Progress] Processed {processed_tiles}/{total_tiles} tiles")
                        
                except Exception as e:
                    print(f"[Warning] Error processing tile at ({x0}, {y0}): {e}")
                    continue

        b_out.FlushCache()
        ds_in = ds_out = None
        
    except Exception as e:
        sys.exit(f"[Error] Tile processing failed: {e}")


# ------------------------------------------------------------------
# 追加：文字列を自動で int/float/bool/None に変換するユーティリティ
# ------------------------------------------------------------------
def _autocast(val: str) -> Any:
    """Convert string to appropriate type (int, float, bool, None, or str)."""
    if not isinstance(val, str):
        return val
        
    val_lower = val.lower().strip()
    if val_lower in {"true", "false"}:
        return val_lower == "true"
    if val_lower in {"null", "none", ""}:
        return None
    try:
        # Try int first
        if "." not in val and "e" not in val_lower:
            return int(val)
    except ValueError:
        pass
    try:
        # Try float
        return float(val)
    except ValueError:
        return val
        
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_kv_list(s: str) -> Dict[str, Any]:
    """Parse ``key=value,key2=value2`` into dict with JSON‑like type casting."""
    out: Dict[str, Any] = {}
    if not s:
        return out
        
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"[Warning] Ignoring malformed parameter: {pair}")
            continue
            
        k, _, v = pair.partition("=")
        k = k.strip()
        v = v.strip()
        
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = _autocast(v)  # Use autocast as fallback
    return out


def calculate_required_halo(algo: str, kwargs: Dict[str, Any]) -> int:
    """Calculate required halo size for given algorithm and parameters."""
    if algo == "slope":
        return 1
    elif algo == "boxgauss":
        return kwargs.get("radius", 1)
    elif algo in {"topo_usm", "multi_scale_usm"}:
        radius = kwargs.get("radius", 1)
        radii = kwargs.get("radii", [1])
        return max(radius, max(radii) if radii else 1)
    elif algo == "multi_scale_boxgauss":
        radii = kwargs.get("radii", [1])
        return max(radii) if radii else 1
    elif algo in {"skyview_factor", "openness", "ambient_occlusion"}:
        max_radius = kwargs.get("max_radius")
        cellsize = kwargs.get("cellsize")
        
        if max_radius is None or cellsize is None:
            sys.exit(f"[Error] {algo} requires both --max-radius and --cellsize parameters")
        
        # Handle both scalar and tuple cellsize
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
        description="Apply FujiShader algorithms from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available algorithms: {', '.join(algos)}"
    )
    p.add_argument("input", help="input DEM GeoTIFF")
    p.add_argument("output", help="output raster")
    p.add_argument(
        "--algo",
        choices=algos,
        default="slope",
        help=f"algorithm to apply (default: slope)",
    )
    p.add_argument(
        "--params",
        metavar="key=val,...",
        help="extra keyword parameters forwarded to the algorithm",
    )
    
    # Common convenience options
    p.add_argument("--slope", type=str, help="Read slope raster if needed")
    p.add_argument("--svf", type=str, help="Read sky-view factor raster if needed")
    p.add_argument("--radius", type=int, help="radius (px) for algorithms")
    p.add_argument("--radii", help="comma‑separated list for multi-scale algorithms")
    p.add_argument("--weights", help="comma‑separated list for multi-scale algorithms")
    p.add_argument("--cellsize", type=float, help="pixel size (for svf, slope)")
    p.add_argument("--max-radius", type=float, dest="max_radius", help="scan distance for svf/openness/ao")
    p.add_argument("--n-directions", type=int, dest="n_directions", help="azimuth sectors for svf/openness")
    p.add_argument("--unit", choices=["degree", "percent"], help="unit for slope")
    p.add_argument("--tile", type=int, metavar="N", help="Process in N×N pixel tiles for large files")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[Error] Input file does not exist: {input_path}")

    # Get algorithm function
    try:
        algo_func = fs.list_algorithms()[args.algo]
    except KeyError:
        sys.exit(f"[Error] Algorithm '{args.algo}' is not registered in FujiShader.")

    # Build kwargs for algorithm
    kwargs: Dict[str, Any] = {}

    # Load auxiliary rasters
    try:
        if args.slope:
            kwargs["slope"], *_ = read_raster(args.slope)
        if args.svf:
            kwargs["svf"], *_ = read_raster(args.svf)
    except Exception as e:
        sys.exit(f"[Error] Failed to load auxiliary raster: {e}")

    # Parse --params
    if args.params:
        try:
            kwargs.update(parse_kv_list(args.params))
        except Exception as e:
            sys.exit(f"[Error] Failed to parse --params: {e}")

    # Convenience options
    if args.radius is not None:
        kwargs["radius"] = args.radius
    if args.radii is not None:
        try:
            kwargs["radii"] = [int(r.strip()) for r in args.radii.split(",") if r.strip()]
        except ValueError as e:
            sys.exit(f"[Error] Invalid --radii format: {e}")
    if args.weights is not None:
        try:
            kwargs["weights"] = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
        except ValueError as e:
            sys.exit(f"[Error] Invalid --weights format: {e}")
    if args.cellsize is not None:
        kwargs["cellsize"] = args.cellsize
    if args.max_radius is not None:
        kwargs["max_radius"] = args.max_radius
    if args.n_directions is not None:
        kwargs["n_directions"] = args.n_directions
    if args.unit is not None:
        kwargs["unit"] = args.unit

    # Process unknown options
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            parser.error(f"Unrecognised argument {tok!r}")
        key = tok.lstrip("-")
        
        # --foo=bar format
        if "=" in key:
            k, v = key.split("=", 1)
            kwargs[k.replace("-", "_")] = _autocast(v)
            i += 1
        # --foo 123 format
        elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            kwargs[key.replace("-", "_")] = _autocast(unknown[i + 1])
            i += 2
        # --flag format
        else:
            kwargs[key.replace("-", "_")] = True
            i += 1

    # Calculate required halo
    halo = calculate_required_halo(args.algo, kwargs)
    
    if args.verbose:
        print(f"[Info] Algorithm: {args.algo}")
        print(f"[Info] Parameters: {kwargs}")
        print(f"[Info] Required halo: {halo}")
        if args.tile:
            print(f"[Info] Tile size: {args.tile}x{args.tile}")

    # Validate tile size vs halo
    if args.tile and args.tile <= 2 * halo:
        sys.exit(f"[Error] Tile size ({args.tile}) must be > 2 * halo ({2 * halo})")

    # Process the data
    try:
        if args.tile:
            if args.verbose:
                print("[Info] Using tile-based processing")
            process_in_tiles(
                args.input,
                args.output,
                algo_func,
                kwargs,
                tile_size=args.tile,
                halo=halo,
            )
        else:
            if args.verbose:
                print("[Info] Loading entire file into memory")
            dem, gt, proj = read_raster(args.input)
            result = algo_func(dem, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
            write_raster(args.output, result, gt, proj)
        
        print(f"[OK] Saved -> {args.output}")
        
    except Exception as e:
        sys.exit(f"[Error] Processing failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()