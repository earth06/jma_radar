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

root_dir = [os.path.dirname(__file__), os.pardir]
sys.path.append(os.path.join(*root_dir))

from Common.database_upsert import upsert  # noqa: E402
from Common.logger import logger  # noqa: E402

with open("./gpvpath.yaml") as f:
    CONFIG = yaml.safe_load(f)


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
grid = pygrib.open("./master/gridGSM.bin")
rr = grid.select(parameterName="Total precipitation")[0]
lats, lons = rr.latlons()
lat = lats[:, 0]
lon = lons[0, :]
grid.close()
##
encoding = {"precip": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "T2m": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "solar_rad_flux": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "u2m": {"zlib": True, "dtype": "float32"},
            "v2m": {"zlib": True, "dtype": "float32"},
            "rh": {"zlib": True, "dtype": "float32"},
            }

fct_variables_names = [
    "Total precipitation",
    "Relative humidity",
    "2 metre temperature",
    "Downward short-wave radiation flux",
    "10 metre U wind component",
    "10 metre V wind component",
]


def grib2_to_netcdf(init_time, is_output=False):
    datetime = pd.to_datetime(init_time, format="%Y%m%d%H%M")
    iyy = datetime.year
    im = datetime.month
    iday = datetime.day
    hour = datetime.hour
    minute = datetime.minute

    filename = CONFIG["DATADIR"] + \
        f"/GSM/1200UTC/{iyy:04d}/{im:02d}/Z__C_RJTD_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00_GSM_GPV_Rjp_Lsurf"
    gribfile0 = filename+"_FD0000-0312_grib2.bin"
    gribfile1 = filename+"_FD0315-0800_grib2.bin"
    gribfile2 = filename+"_FD0803-1100_grib2.bin"
    logger.info("decode grib2 start")
    grib0 = pygrib.open(gribfile0)
    grib1 = pygrib.open(gribfile1)
    grib2 = pygrib.open(gribfile2)
    fct_variables_dict = {}
    for var in fct_variables_names:
        # 降水量と湿度はparameterName参照なので
        if var in ["Total precipitation", "Relative humidity"]:
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
        "solar_rad_flux": (["time", "lat", "lon"], fct_variables_dict["Downward short-wave radiation flux"], {"units": "[W*m^-2]", "long name": "downward short wave radiation flux"}),
        "u2m": (["time", "lat", "lon"], fct_variables_dict["10 metre U wind component"], {"units": "m/s", "long name": "2m u wind"}),
        "v2m": (["time", "lat", "lon"], fct_variables_dict["10 metre V wind component"], {"units": "m/s", "long name": "2m v wind"}),
        "rh": (["time", "lat", "lon"], fct_variables_dict["Relative humidity"], {"units": "%", "long name": "relative humidity"}),
    }
    attrs = {"title": "JMA GSM GPV",
             "range": "1H-264H forecast", "init_time": "1200UTC"}
    ds = xr.Dataset(values, coords, attrs)
    if is_output:
        filename = f"GSM_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00UTC_all_1H_264H_Lsurf_GPV.nc"
        os.makedirs(CONFIG["GSM_NCDIR"], exist_ok=True)
        ds.to_netcdf(CONFIG["GSM_NCDIR"]+f"/{filename}", encoding=encoding)
        logger.info(f"save to {CONFIG['GSM_NCDIR']}/{filename}")
    # clean up
    grib0.close()
    grib1.close()
    grib2.close()
    logger.info("decode end correctly")
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


def netcdf_to_dataframe(ds, init_time):
    logger.info("Extract station point data")
    with open("./master/pref_jp2en.json", "tr") as f:
        jp2en_dict = json.load(f)
    tdfk_list = list(jp2en_dict.keys())
    mask = xr.open_dataset("./master/GSM_MASK_TDFK.nc")
    focus_tdfk = ["北海道", "東京都", "大阪府", "愛知県", "福岡県", "広島県",
                  "静岡県", "宮城県", "愛媛県", "岐阜県", "三重県", "長野県"]
    weight_cols = ["w_precip", "w_T2m", "w_solar_rad_flux",
                   "w_u2m", "w_v2m", "w_rh",
                   "w_wind", "w_DI", "w_WBGT"]
    # 変数に使っている都道府県だけで処理を行う
    mask = mask.sel(tdfk=focus_tdfk)

    tdfk_area = mask["tdfk_area"].sum(dim=["lat", "lon"])
    ds["time"] = pd.to_datetime(ds["time"].values) + \
        pd.offsets.Hour(9)  # UTC->JST
    ds1yr = ds.isel(time=slice(0, 144))
    # 風速.不快指数とwbgtを算出
    #ds1yr["wind"] = xr.ufuncs.sqrt(ds1yr["u2m"]**2 + ds1yr["v2m"]**2)
    ds1yr["wind"] = np.sqrt(ds1yr["u2m"]**2 + ds1yr["v2m"]**2)
    ds1yr["DI"] = _calc_di(ds1yr)
    ds1yr["WBGT"] = _calc_wbgt(ds1yr)

    # 都道府県別に重みをつける
    for wcol in weight_cols:
        org = wcol[2:]
        ds1yr[wcol] = ds1yr[org]*mask["tdfk_area"]
    dstdfk = ds1yr[weight_cols].sum(dim=["lat", "lon"])
    dsout = dstdfk/tdfk_area
    dfout = dsout.to_dataframe().reset_index()
    dfout["tdfk"] = dfout["tdfk"].apply(lambda x: jp2en_dict[x])
    # 予報初期時刻をつけておく
    # 初期時刻＋３H=00:00JSTを予報の開始としてフラグを立てる用にする
    init_time = pd.to_datetime(
        init_time, format="%Y%m%d%H%M")+pd.offsets.Hour(9) + pd.offsets.Hour(1)
    dfout["init_time"] = init_time
    col_dict = {'w_precip': 'precip',
                'w_solar_rad_flux': "solar_rad_flux",
                "w_WBGT": "WBGT",
                'w_T2m': 'T2m',
                'w_u2m': 'u2m',
                'w_v2m': 'v2m',
                'w_rh': 'rh',
                'w_ps': 'ps',
                'w_pmsl': 'pmsl',
                'w_tot_cloud_cover': 'tot_cloud_cover',
                'w_wind': 'wind',
                'w_DI': 'DI',
                'init_time': 'it_time'}
    dfout.rename(columns=col_dict, inplace=True)
    return dfout


def split_to_each_prefecture(dfout, init_time):
    init_time = pd.to_datetime(
        init_time, format="%Y%m%d%H%M")+pd.offsets.Hour(9) + pd.offsets.Hour(1)
    sdate = init_time.strftime("%Y%m%d%H%M00JST")
    dfdict = {}
    for t in dfout["tdfk"].unique():
        tmp = dfout[dfout["tdfk"] == t].copy()
        tmp["time"] = pd.to_datetime(tmp["time"])
        tmp.set_index("time", inplace=True)
        tmp["T2m"] = tmp["T2m"] - 273.15
        diff = tmp.index - (pd.to_datetime(tmp["it_time"]))
        tmp["elapse_days"] = diff.dt.days
        tmp["elapse_hours"] = diff.dt.total_seconds()/3600.0
        tmp["elapse_hours"] = tmp["elapse_hours"].apply(lambda x: round(x))
        tmp.index.name = "date_time"
        # 初期時刻を0000JSTにする
        tmp = tmp[tmp["elapse_hours"] >= 2]
        tmp["it_time"] = (pd.to_datetime(tmp["it_time"]) +
                          pd.offsets.Hour(2)).dt.strftime("%Y-%m-%d")
        tmp.drop(columns=["elapse_hours"], inplace=True)
        dfdict[t] = tmp.copy()
    return dfdict


def upsert_gsm(df, CONNECTION_CONFIG):
    """dbにGSMデータを追加

    Args:
        df (pd.DataFrame): 都道府県別GSM
        CONNECTION_CONFIG (db): dbに接続

    Returns:
        int: 0
    """
    upsert(df, "gsm_area_mean_al", [
        "date_time", "tdfk", "elapse_days"], CONNECTION_CONFIG)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ex) ./convert_GSM_to_pred_weather_csv.py init_time --output_nc")
    parser.add_argument("init_time", help="GSM inital time YYYYMMDDhhmm")

    # ファイル出力計の引数
    parser.add_argument("--output_nc", action="store_true",
                        help="flag args, wheather output netcdf file or not")
    parser.add_argument("--output_csv", action="store_true")
    parser.add_argument("--output_db", action="store_true")

    # 処理の途中から始めるフラグ
    parser.add_argument("--from_nc", action="store_true",
                        help="netcdf-> csvの変換から処理を開始")
    parser.add_argument("--from_csv", action="store_true",
                        help="csv -> dbへの登録のみ実行")
    args = parser.parse_args()
    init_time = args.init_time

    # grib2 -> nc
    if args.from_nc:
        #すでにnetcdfファイルがある.
        filename = f"GSM_{init_time}00UTC_all_1H_264H_Lsurf_GPV.nc"
        filepath = CONFIG["GSM_NCDIR"]+f"/{filename}"
        logger.info(f"Loading {filepath} directly")
        try:
            ds = xr.open_dataset(filepath)
        except BaseException as e:
            logger.error("load netcdf file  failed")
            raise
    elif args.from_csv:
        #すでにcsvファイルがある
        logger.info("Skip extracting part")
        pass
    else:
        logger.info(f"Decode grib2 @{init_time}")
        try:
            ds = grib2_to_netcdf(init_time, args.output_nc)
        except BaseException as e:
            logger.error("Decode failed")
            raise

    # nc -> csv
    if args.from_csv:
        logger.info("Csv files are loaeded directly")
        init_time = pd.to_datetime(init_time, format="%Y%m%d%H%M") +\
            pd.offsets.Hour(9) + pd.offsets.Hour(1)
        sdate = init_time.strftime("%Y%m%d%H%M00JST")
        logger.info(f"Loading csv file @ {sdate}")
        files = glob.glob(
            CONFIG["GSM_CSVDIR"]+f"/*/{init_time.year:04d}/{init_time.month:02d}/GSM_*_{sdate}.csv")
        dflist = []
        for f in files:
            dflist.append(pd.read_csv(f))
        dfall = pd.concat(dflist)
    else:
        try:
            dfout = netcdf_to_dataframe(ds, init_time)
            dfdict = split_to_each_prefecture(dfout, init_time)
            # ファイル出力
            if args.output_csv:
                init_time = pd.to_datetime(init_time, format="%Y%m%d%H%M") +\
                    pd.offsets.Hour(9) + pd.offsets.Hour(1)
                sdate = init_time.strftime("%Y%m%d%H%M00JST")
                for tdfk in dfdict.keys():
                    tmp = dfdict[tdfk]
                    dirname = CONFIG["GSM_CSVDIR"] + \
                        f"/{tdfk}/{init_time.year:04d}/{init_time.month:02d}"
                    os.makedirs(dirname, exist_ok=True)
                    tmp.to_csv(f"{dirname}/GSM_{tdfk}_{sdate}.csv")
                    logger.info(f"save to {dirname}/GSM_{tdfk}_{sdate}.csv")
            dflist = list(dfdict.values())
            dfall = pd.concat(dflist)
        except BaseException as e:
            logger.error(f"Extract station point failed")
            raise

    # csv -> db
    if args.output_db:
        CONNECTION_CONFIG = sqlite3.connect(CONFIG["DB_DIR"])
        upsert_gsm(dfall,CONNECTION_CONFIG)
        CONNECTION_CONFIG.close()
