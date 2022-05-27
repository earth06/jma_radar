#! /home/oonishi/miniconda3/envs/ml/bin/python

import xarray as xr
import numpy as np
import pandas as pd
import glob
import pygrib
from datetime import timedelta
import os
import argparse
import json
import yaml
import sqlite3
import sys

ROOTDIR=os.path.dirname(__file__)
root_dir = [ROOTDIR , os.pardir]
sys.path.append(os.path.join(*root_dir))


def _get_param_GSM(var0, var1, var2):
    # 放射・降水以外は初期値があるので除外する
    idx = 0
    tmax = len(var0) + len(var1) + len(var2)
    alldata = np.zeros((tmax, 151, 121), dtype="float32")
    # 1H~84H
    for var in [var0, var1, var2]:
        for d in var:
            alldata[idx, :, :] = d.values
            idx += 1
    return alldata


def _get_precip_GSM(precip0, precip1, precip2):
    idx = 0
    alldata = np.zeros(
        (len(precip0)+len(precip1)+len(precip2), 151, 121), dtype="float32")
    datetimes = []

    # 1H
    rr = precip0[0]
    alldata[idx, :, :] = rr.values
    datetimes.append(rr.validDate+timedelta(hours=1))

    # 2H~84H
    for i in range(1, len(precip0)):
        idx += 1
        rr0 = precip0[i]
        datetimes.append(rr.validDate+timedelta(hours=(i+1)))
        tmp = rr0.values - precip0[i-1].values
        tmp[tmp < 0] = 0
        alldata[idx, :, :] = tmp
    lats, lons = rr.latlons()
    lat, lon = lats[:, 0], lons[0, :]

    # 87H
    idx += 1
    rr1 = precip1[0]
    datetimes.append(rr0.validDate+timedelta(hours=84+3))  # i=83
    alldata[idx, :, :] = (rr1.values - rr0.values)/3.0

    # 90~132H
    for j in range(1, len(precip1)):
        idx += 1
        rr1 = precip1[j]
        datetimes.append(rr1.validDate+timedelta(hours=84+3*(j+1)))
        tmp = (rr1.values-precip1[j-1].values)/3.0
        tmp[tmp < 0] = 0
        alldata[idx, :, :] = tmp
    elapsetime = 84+3*(j+1)

    # 135~264H
    idx += 1
    rr2 = precip2[0]
    datetimes.append(rr1.validDate+timedelta(hours=elapsetime+3))
    alldata[idx, :, :] = (rr2.values-rr1.values)/3.0
    for k in range(1, len(precip2)):
        idx += 1
        rr2 = precip2[k]
        datetimes.append(rr2.validDate+timedelta(hours=elapsetime+3*(k+1)))
        tmp = (rr2.values-precip2[k-1].values)/3.0
        tmp[tmp < 0] = 0
        alldata[idx, :, :] = tmp
    return alldata, datetimes


# 格子情報定義を読み込む
grid = pygrib.open(f"{ROOTDIR}/master/gridGSM.bin")
rr = grid.select(parameterName="Total precipitation")[0]
lats, lons = rr.latlons()
lat = lats[:, 0]
lon = lons[0, :]
grid.close()
##
encoding = {"precip": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "T2m": {"zlib": True, "complevel": 5, "dtype": "float32"},
            #"solar_rad_flux": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "ch_cover":{"zlib":True, "dtype":"float32"},
            "cm_cover":{"zlib":True, "dtype":"float32"},
            "cl_cover":{"zlib":True, "dtype":"float32"},
            "c_tot_cover":{"zlib":True, "dtype":"float32"},
            "u2m": {"zlib": True, "dtype": "float32"},
            "v2m": {"zlib": True, "dtype": "float32"},
            "rh": {"zlib": True, "dtype": "float32"},
            }

fct_variables_names = [
    "Total precipitation",
    "Relative humidity",
    "2 metre temperature",
    #"Downward short-wave radiation flux",
    "High cloud cover",
    "Medium cloud cover",
    "Low cloud cover",
    "Total cloud cover",
    "10 metre U wind component",
    "10 metre V wind component",
]

paramerter_names=[
    "Total precipitation", "Relative humidity",
    "Low cloud cover", "Medium cloud cover","High cloud cover", "Total cloud cover"
]


def grib2_to_netcdf(init_time, is_output=False, GSMDIR="/home/takato/Data/JMA/GSM/jp/1200UTC"
    , NCDIR="/home/takato/Data/JMA/GSM/jpnc/1200UTC"):
    datetime = pd.to_datetime(init_time, format="%Y%m%d%H%M")
    iyy = datetime.year
    im = datetime.month
    iday = datetime.day
    hour = datetime.hour
    minute = datetime.minute

    filename = \
        f"{GSMDIR}/{iyy:04d}/{im:02d}/Z__C_RJTD_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00_GSM_GPV_Rjp_Lsurf"
    gribfile0 = filename+"_FD0000-0312_grib2.bin"
    gribfile1 = filename+"_FD0315-0800_grib2.bin"
    gribfile2 = filename+"_FD0803-1100_grib2.bin"
    grib0 = pygrib.open(gribfile0)
    grib1 = pygrib.open(gribfile1)
    grib2 = pygrib.open(gribfile2)
    fct_variables_dict = {}
    for var in fct_variables_names:
        # 降水量と湿度はparameterName参照なので
        if var in paramerter_names:
            buf0 = grib0.select(parameterName=var)
            buf1 = grib1.select(parameterName=var)
            buf2 = grib2.select(parameterName=var)
        else:
            buf0 = grib0.select(name=var)
            buf1 = grib1.select(name=var)
            buf2 = grib2.select(name=var)

        # 初期値があるかないかで場合分け
        if var == "Total precipitation":
            fct_variables_dict[var], datetimes = _get_precip_GSM(
                buf0, buf1, buf2)
        elif var == "Downward short-wave radiation flux":
            fct_variables_dict[var] = _get_param_GSM(buf0, buf1, buf2)
        else:
            # 初期値はskip
            fct_variables_dict[var] = _get_param_GSM(buf0[1:], buf1, buf2)

    # netcdf準備
    coords = {"time": pd.to_datetime(datetimes),
              "lat": ("lat", lat, {"units": "degrees_north"}),
              "lon": ("lon", lon, {"units": "degrees_east"})}
    values = {
        "precip": (["time", "lat", "lon"], fct_variables_dict["Total precipitation"], {"units": "[mm/hr]", "long name": "total precipitaion in last hour"}),
        "T2m": (["time", "lat", "lon"], fct_variables_dict["2 metre temperature"], {"units": "[K]", "long name": "2m temperature"}),
#        "solar_rad_flux": (["time", "lat", "lon"], fct_variables_dict["Downward short-wave radiation flux"], {"units": "[W*m^-2]", "long name": "downward short wave radiation flux"}),
        "ch_cover": (["time", "lat", "lon"], fct_variables_dict["High cloud cover"], {"units": "%", "long name": "high cloud cover"}),
        "cm_cover": (["time", "lat", "lon"], fct_variables_dict["Medium cloud cover"], {"units": "%", "long name": "middle cloud cover"}),
        "cl_cover": (["time", "lat", "lon"], fct_variables_dict["Low cloud cover"], {"units": "%", "long name": "low cloud cover"}),
        "c_tot_cover": (["time", "lat", "lon"], fct_variables_dict["Total cloud cover"], {"units": "%", "long name": "total cloud cover"}),
        "u2m": (["time", "lat", "lon"], fct_variables_dict["10 metre U wind component"], {"units": "m/s", "long name": "2m u wind"}),
        "v2m": (["time", "lat", "lon"], fct_variables_dict["10 metre V wind component"], {"units": "m/s", "long name": "2m v wind"}),
        "rh": (["time", "lat", "lon"], fct_variables_dict["Relative humidity"], {"units": "%", "long name": "relative humidity"}),
    }
    attrs = {"title": "JMA GSM GPV",
             "range": "1H-264H forecast", "init_time": "1200UTC"}
    ds = xr.Dataset(values, coords, attrs)
    if is_output:
        filename = f"GSM_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00UTC_all_1H_264H_Lsurf_GPV.nc"
        os.makedirs(f"{NCDIR}/{iyy}/{im:02d}", exist_ok=True)
        ds.to_netcdf(f"{NCDIR}/{iyy}/{im:02d}/{filename}", encoding=encoding)
    # clean up
    grib0.close()
    grib1.close()
    grib2.close()
    print(init_time,"end")
    return ds


def _calc_di(ds):
    Tc = ds["T2m"]-273.15
    di = 0.81*Tc+0.010*ds["rh"]*(0.99*Tc-14.3)+46.3
    return di


def _calc_wbgt(ds):
    Tc = ds["T2m"]-273.15
    SR = ds["solar_rad_flux"]*1e-3
    wbgt = 0.735*Tc+0.0374*ds["rh"]+0.00292*Tc*ds["rh"]\
        + 7.619*SR**2 \
        - 0.0572*ds["wind"]
    return wbgt
