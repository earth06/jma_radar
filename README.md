# jma_radar

## Overview
気象業務支援センターより配信されている気象庁全国合成レーダ―(grib2形式)を読み込むモジュール

## Description
気象庁が独自に定義している、気象庁全国合成レーダーGPV(girb2)形式でpythonでデコードし、numpy配列または、xarray.Datasetに変換するモジュール

## Demo

```python
ds=JMA_Radar.read_radar("Z__C_RJTD_YYYYMMDDhhmmss_RDR_JMAGPV_Ggis1km_Prr10lv_ANAL_grib2.bin").to_xarray()
fig=plt.figure(figsize=(9,9),facecolor="w")
ax=fig.add_subplot(1,1,1,projection=ccrs.AzimuthalEquidistant(central_longitude=140))
ax.coastlines()
ax.gridlines(draw_labels=True)
cf=ax.pcolormesh(ds["lon"],ds["lat"],ds["rr"].isel(time=0),transform=ccrs.PlateCarree())
fig.colorbar(cf,orientation="horizontal",extend="max",shrink=0.8)
ax.set_extent([128,145,30,45])
```



![sample](C:\Users\kjkrs\OneDrive\Document\GitHub\jma_radar\sample.png)

## Requirement

| package | version |
| ------- | ------- |
| numpy   |         |
| pandas  | 1.0     |
| numba   |         |
| xarray  |         |
| netCDF4 |         |

よほど古くなければバージョンは気にしなくても大丈夫のはずです。

## Usage 

* girb2ファイルの読み込み

```python
#[1]
from jma_radar import JMA_Radar
data=JMA_Radar.read_radar("file.grib2")
```

* 緯度経度の取得

```python
#[2]
xx,yy=ds.latlon()
```

* 値の取得

```python
#[3]
val=data.values
```

* xarray.Datasetに変換

```python
#[4]
ds=data.to_xarray()
ds
```

>- Dimensions:
>  - **lat**: 3360
>  - **lon**: 2560
>  - **time**: 1
>- Coordinates:
>  - **time** (time) datetime64[ns] 2021-05-21
>  - **lat** (lat) float64 20.0 20.01 20.02 ... 47.99 48.0
>  - **lon** (lon) float64 118.0 118.0 118.0 ... 150.0 150.0
>- Data variables:
>  - rr (time, lat, lon) float32 nan nan nan nan ... nan nan nan nan

## Author
earth06
