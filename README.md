# FujiShader
Advanced DEM visualization library.
- Fast
- Capable to process GIGABYTES of COG (Cloud Optimized GeoTIFF).
- Some original shading methods

## Installation
```bash
pip install git+https://github.com/geoign/FujiShader.git
```

The development of QGIS module is in progress.

## Shaders
### Slope
Classic slope map. Do not use for LatLon Grid.
```bash
FujiShader DEM.tif Slope.tif --algo slope`
```

### WarmCool
Original shading method based on vision science.
- `FujiShader RidgeVolley.tif WarmCool.tif --algo warmcool_map --slope slope.tif --svf svf.tif`
