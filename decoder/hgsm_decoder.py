#! /home/oonishi/miniconda3/envs/ml/bin/python

import xarray as xr
import numpy as np
import pandas as pd
import pygrib
from datetime import timedelta
import os
import sys

ROOTDIR = os.path.dirname(__file__)
root_dir = [ROOTDIR, os.pardir]
sys.path.append(os.path.join(*root_dir))


class hGSMDecoder:
    def __init__(self):
        grid = xr.open_dataset(f"{ROOTDIR}/master/gridhGSM.nc")
        self.lat = grid["lat"].values
        self.lon = grid["lon"].values
        self.Nlat = len(self.lat)
        self.Nlon = len(self.lon)
        self.N_1H_STEP = 132
        ##
        self.encoding = {
            "precip": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "T2m": {"zlib": True, "complevel": 5, "dtype": "float32"},
            "solar_rad_flux": {"zlib": True, "complevel": 5, "dtype": "float32"},
            # "ch_cover": {"zlib": True, "dtype": "float32"},
            # "cm_cover": {"zlib": True, "dtype": "float32"},
            # "cl_cover": {"zlib": True, "dtype": "float32"},
            # "c_tot_cover": {"zlib": True, "dtype": "float32"},
            "p0": {"zlib": True, "dtype": "float32"},
            "u2m": {"zlib": True, "dtype": "float32"},
            "v2m": {"zlib": True, "dtype": "float32"},
            "rh": {"zlib": True, "dtype": "float32"},
        }

        self.fct_variables_names = [
            "Total precipitation",
            "Relative humidity",
            "2 metre temperature",
            "Downward short-wave radiation flux",
            "10 metre U wind component",
            "10 metre V wind component",
            "Pressure reduced to MSL",
        ]

        self.paramerter_names = [
            "Total precipitation",
            "Relative humidity",
            "Low cloud cover",
            "Medium cloud cover",
            "High cloud cover",
            "Total cloud cover",
        ]
        self.hgsm_suffix = [
            "0000-0100",
            "0101-0200",
            "0201-0300",
            "0301-0400",
            "0401-0500",
            "0501-0512",
            "0515-0700",
            "0703-0900",
            "0903-1100",
        ]
        self.N_grib = len(self.hgsm_suffix)

    def _get_param_GSM(self, vars, skip_init=False):
        # 放射・降水以外は初期値があるので除外する
        idx = 0
        if skip_init:
            tmax = len(vars) - 1
            begin_idx = 1
        else:
            tmax = len(vars)
            begin_idx = 0
        alldata = np.zeros((tmax, self.Nlat, self.Nlon), dtype="float32")
        # 1H~264
        for idx in range(tmax):
            alldata[idx, :, :] = vars[idx + begin_idx].values
        return alldata

    def _get_precip_GSM(self, precip_s):
        idx = 0
        tmax = len(precip_s)
        alldata = np.zeros((tmax, self.Nlat, self.Nlon), dtype="float32")
        prev_cum_precip = np.zeros((self.Nlat, self.Nlon), dtype="float32")
        datetimes = []

        # 1H
        rr = precip_s[0]
        alldata[idx, :, :] = rr.values
        datetimes.append(rr.validDate + timedelta(hours=1))
        prev_cum_precip[:, :] = rr.values
        # ~132H
        for idx in range(1, self.N_1H_STEP):
            print(idx)
            rr0 = precip_s[idx].values
            datetimes.append(rr.validDate + timedelta(hours=(idx + 1)))
            tmp = rr0 - prev_cum_precip
            tmp[tmp < 0] = 0
            alldata[idx, :, :] = tmp
            prev_cum_precip[:, :] = rr0

        # 135~264H
        for idx, timestep in enumerate(range(self.N_1H_STEP + 3, 264 + 1, 3)):
            print(idx + self.N_1H_STEP)
            rr0 = precip_s[(idx + self.N_1H_STEP)].values
            datetimes.append(rr.validDate + timedelta(hours=(timestep + 1)))
            tmp = (rr0 - prev_cum_precip) / 3.0
            tmp[tmp < 0] = 0
            alldata[(idx + self.N_1H_STEP), :, :] = tmp
            prev_cum_precip[:, :] = rr0
        return alldata, datetimes

    def grib2_to_netcdf(
        self,
        init_time,
        is_output=False,
        GSMDIR="/home/takato/Data/JMA/GSM/jp/1200UTC",
        NCDIR="/home/takato/Data/JMA/GSM/jpnc/1200UTC",
    ):
        datetime = pd.to_datetime(init_time, format="%Y%m%d%H%M")
        iyy = datetime.year
        im = datetime.month
        iday = datetime.day
        hour = datetime.hour
        minute = datetime.minute
        bufs = []
        for suffx in self.hgsm_suffix:
            filename = f"{GSMDIR}/Z__C_RJTD_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00_GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD{suffx}_grib2.bin"
            bufs.append(pygrib.open(filename))

        fct_variables_dict = {}
        for var in self.fct_variables_names:

            dats = []
            for buf in bufs:
                # 降水量と湿度はparameterName参照なので
                if var in self.paramerter_names:
                    dats.extend(buf.select(parameterName=var))
                else:
                    dats.extend(buf.select(name=var))

            # 初期値があるかないかで場合分け
            if var == "Total precipitation":
                fct_variables_dict[var], datetimes = self._get_precip_GSM(dats)
            elif var == "Downward short-wave radiation flux":
                fct_variables_dict[var] = self._get_param_GSM(dats)
            else:
                # 初期値はskip
                fct_variables_dict[var] = self._get_param_GSM(dats, skip_init=True)

        # netcdf準備
        coords = {
            "time": pd.to_datetime(datetimes),
            "lat": ("lat", self.lat, {"units": "degrees_north"}),
            "lon": ("lon", self.lon, {"units": "degrees_east"}),
        }
        values = {
            "precip": (
                ["time", "lat", "lon"],
                fct_variables_dict["Total precipitation"],
                {"units": "[mm/hr]", "long name": "total precipitaion in last hour"},
            ),
            "T2m": (
                ["time", "lat", "lon"],
                fct_variables_dict["2 metre temperature"],
                {"units": "[K]", "long name": "2m temperature"},
            ),
            "solar_rad_flux": (
                ["time", "lat", "lon"],
                fct_variables_dict["Downward short-wave radiation flux"],
                {
                    "units": "[W*m^-2]",
                    "long name": "downward short wave radiation flux",
                },
            ),
            # "ch_cover": (
            #     ["time", "lat", "lon"],
            #     fct_variables_dict["High cloud cover"],
            #     {"units": "%", "long name": "high cloud cover"},
            # ),
            # "cm_cover": (
            #     ["time", "lat", "lon"],
            #     fct_variables_dict["Medium cloud cover"],
            #     {"units": "%", "long name": "middle cloud cover"},
            # ),
            # "cl_cover": (
            #     ["time", "lat", "lon"],
            #     fct_variables_dict["Low cloud cover"],
            #     {"units": "%", "long name": "low cloud cover"},
            # ),
            # "c_tot_cover": (
            #     ["time", "lat", "lon"],
            #     fct_variables_dict["Total cloud cover"],
            #     {"units": "%", "long name": "total cloud cover"},
            # ),
            "u2m": (
                ["time", "lat", "lon"],
                fct_variables_dict["10 metre U wind component"],
                {"units": "m/s", "long name": "2m u wind"},
            ),
            "v2m": (
                ["time", "lat", "lon"],
                fct_variables_dict["10 metre V wind component"],
                {"units": "m/s", "long name": "2m v wind"},
            ),
            "rh": (
                ["time", "lat", "lon"],
                fct_variables_dict["Relative humidity"],
                {"units": "%", "long name": "relative humidity"},
            ),
            "ps": (
                ["time", "lat", "lon"],
                fct_variables_dict["Pressure reduced to MSL"],
                {"units": "hPa", "long name": "Pressure reduced to MSL"},
            )        
        }
        attrs = {
            "title": "JMA GSM GPV",
            "range": "1H-264H forecast",
            "init_time": "1200UTC",
        }
        ds = xr.Dataset(values, coords, attrs)
        if is_output:
            filename = f"GSM_{iyy:04d}{im:02d}{iday:02d}{hour:02d}{minute:02d}00UTC_all_1H_264H_Lsurf_GPV.nc"
            os.makedirs(f"{NCDIR}/{iyy}/{im:02d}", exist_ok=True)
            ds.to_netcdf(f"{NCDIR}/{iyy}/{im:02d}/{filename}", encoding=self.encoding)
        # clean up
        for buf in bufs:
            buf.close()
        print(init_time, "end")
        return ds
