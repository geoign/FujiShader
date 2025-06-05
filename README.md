# FujiShader
Advanced DEM visualization library.
- Fast
- Capable to process GIGABYTES of COG (Cloud Optimized GeoTIFF).
- Some original shading methods

***!!! Currently in ALPHA version !!!***

## Installation
```bash
pip install git+https://github.com/geoign/FujiShader.git
```

The development of QGIS module is in progress.

## Shaders
Note that none of them are compatible to LatLon Grid... yet.

### Slope
Classic slope map.
```bash
FujiShader DEM.tif Slope.tif --algo slope
```
    Optional arguments:
    --cellsize: 1.0 by default
    --unit: degrees by default (or percent)
    --treat_nan: None by default

### SkyView Factor
Classic skyview factor.
```bash
FujiShader DEM.tif SVF.tif --algo skyview_factor
```
    Optional arguments:
    --cellsize 1.0 (meter/pixel)
    --max_radius 100.0 (= 100 meters)
    --n_directions 16 (= 16 directions)
    --tile_size None (Use 1024 or so for COG)
    --memory_efficient: False (Usually don't need)

## SkyView Factor Fast
Pseudo SVF. Not based on the original implementation.
```bash
FujiShader DEM.tif SVF.tif --algo skyview_factor_fast
```
    Optional arguments:
    --cellsize 1.0 (meter/pixel)
    --max_radius 100.0 (= 100 meters)
    --n_directions 16 (= 16 directions)
    --n_samples 8 (Reduced sampling steps)
    --tile_size None (Use 1024 or so for COG)

## Curvature
Classic curvature map.
```bash
FujiShader DEM.tif CURVATURE.tif --algo profile_curvature
FujiShader DEM.tif CURVATURE.tif --algo plan_curvature
FujiShader DEM.tif CURVATURE.tif --algo total_curvature
```
    Optional arguments:
    --cellsize 1.0 (meter/pixel)

## Openness
Classic openness map based on Yokoyama et al. (2002).
```bash
FujiShader DEM.tif CURVATURE.tif --algo positive_openness
FujiShader DEM.tif CURVATURE.tif --algo negative_openness
```
    Optional arguments:
    --cellsize 1.0 (meter/pixel)
    --max_radius 100.0 (= 100 meters)
    --n_directions 16 (= 16 directions)

## RidgeVolley - Integral method
Highlights ridges and shadows volleys. Fastest implementation.
```bash
FujiShader DEM.tif RV.tif --algo multi_scale_integral
```
    Optional arguments:
    --radii 4,16,64,256 (pixels range)
    --normalize True (Normalize to -1 ~ +1)
    --treat_nan NaN (Replace NaN to this if set)

## RidgeVolley - BoxGauss method
Highlights ridges and shadows volleys. Faster implementation.
```bash
FujiShader DEM.tif CURVATURE.tif --algo multi_scale_boxgauss
```
    Optional arguments:
    --cellsize 1.0 (meter/pixel)
    
### WarmCool
Original shading method based on vision science.
```bash
FujiShader RidgeVolley.tif WarmCool.tif --algo warmcool_map --slope slope.tif --svf svf.tif`
```
