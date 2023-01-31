import cartopy.crs as ccrs
from scipy.ndimage.filters import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
#import xarray.ufuncs as xufuncs
import pygrib
import metpy.calc as mcalc
from metpy.units import units
import json
import pathlib
import sys
import scipy.ndimage as ndimage
sys.path.append(pathlib.Path(__file__).resolve().parent)
import custom_colormap

jmacmap = custom_colormap.get_jmacmap2()


class GSMGlobal():
    @classmethod
    def add_dew_point_depreession_equ_potential_T(self, ds):
        dew_point = mcalc.dewpoint_from_relative_humidity(ds["T"], ds["rh"])
        ds["Tw"] = (["level", "lat", "lon"], dew_point.values+273.15)
        ds["Tw"].attrs["units"] = "kelvin"
        ds["T-Tw"] = ds["T"] - ds["Tw"]
        ds["theta_w"] = mcalc.equivalent_potential_temperature(
            85000*units.pascal, ds["T"].sel(level=850), ds["Tw"].sel(level=850))

    @classmethod
    def add_wind_vorticity_omega(self, ds):
        ds["vo"] = mcalc.vorticity(ds["u"], ds["v"])
        ds["wind"] = np.sqrt(ds["u"]**2 + ds["v"]**2)
        ds["vp"] = ds["w"]*1e-2*3600e0
        ds["vo"].attrs["units"] = "1/s"
        ds["vp"].attrs["units"] = "hPa/h"
        ds["wind"].attrs["units"] = "m/s"

    def __init__(self):
        module_dir = pathlib.Path(__file__).resolve().parent
        with open(f"{module_dir}/gsm_config.json") as f:
            config = json.load(f)
        self.SFC_CONFIG = config["GSM"]["SFC"]
        self.PLEV_CONFIG = config["GSM"]["PLEV"]
        self.plevels = config["GSM"]["PLEVELS"]

    def read_sfc(self, file, product_type="ANAL", crip="Asia",timestep=[0], prev_precip:xr.DataArray=None):
        """

        Parameters
        -----------
        file : str
        product_type : str
            "ANAL" or "FCT"
        crip : str:
        timestep : list
            if product_type="ANAL" then timestamp=[0]
            else timestampe=[1,2,3....]
        prev_precip: xr.DataArray
        """
        gsm = pygrib.open(file)
        param_names = list(self.SFC_CONFIG.keys())
        if product_type == "ANAL":
            param_names.remove("precip")
            timestep=[0]
        dslist=[]
        for t in timestep:
            data={}
            for key in param_names:
                namekey = self.SFC_CONFIG[key]["namekey"]
                name = self.SFC_CONFIG[key]["name"]
                if namekey == "name":
                    dat = gsm.select(name=name)[t]
                elif namekey == "parameterName":
                    dat = gsm.select(parameterName=name)[t]
                data[key] = (["lat", "lon"], dat.values,
                             {"title": self.SFC_CONFIG[key]["name"],
                              "units": self.SFC_CONFIG[key]["units"]})
                lat, lon = dat.latlons()
                lat = lat[:, 0]
                lon = lon[0, :]
                coords = {
                    "lat": (("lat", lat, {"units": "degrees_north"})),
                    "lon": (("lon", lon, {"units": "degrees_east"}))
                }
                ds = xr.Dataset(data, coords)
                if type(crip) is tuple:
                    lon1, lon2, lat1, lat2 = crip
                    ds = ds.sel(lon=slice(lon1, lon2), lat=slice(lat1, lat2))
                elif crip == "Asia":
                    ds = ds.sel(lat=slice(60, 10), lon=slice(100, 180))
            dslist.append(ds)
            del data
        gsm.close()
        if len(dslist)==1:
            dsout=dslist[0]
        else:
            dsout=xr.concat(dslist, dim="time")
        if prev_precip is not None:
            dsout["precip"]=dsout["precip"] - prev_precip
        del dslist
        return dsout

    def read_plev(self, file, crip="Asia", timestep=[0]):
        gsm = pygrib.open(file)
        param_names = list(self.PLEV_CONFIG.keys())
        dslist=[]
        for t in timestep:
            data={}
            for key in param_names:
                namekey = self.PLEV_CONFIG[key]["namekey"]
                name = self.PLEV_CONFIG[key]["name"]
                print(name)
                dat_list = []
                for lev in self.plevels:
                    if name == "Relative humidity" and lev < 300:
                        dat_list.append(np.zeros_like(dat.values))
                        continue
                    if namekey == "name":
                        dat = gsm.select(name=name, level=lev)[0]
                    elif namekey == "parameterName":
                        dat = gsm.select(parameterName=name, level=lev)[0]
                    dat_list.append(dat.values)
                data[key] = (["level", "lat", "lon"], np.array(dat_list),
                             {"title": self.PLEV_CONFIG[key]["name"],
                              "units": self.PLEV_CONFIG[key]["units"]})
                del dat_list
            lat, lon = dat.latlons()
            lat = lat[:, 0]
            lon = lon[0, :]
            coords = {
                "level": (("level", self.plevels, {"units": "hPa"})),
                "lat": (("lat", lat, {"units": "degrees_north"})),
                "lon": (("lon", lon, {"units": "degrees_east"}))
            }
            ds = xr.Dataset(data, coords)
            if type(crip) is tuple:
                lon1, lon2, lat1, lat2 = crip
                ds = ds.sel(lon=slice(lon1, lon2), lat=slice(lat1, lat2))
            elif crip == "Asia":
                ds = ds.sel(lat=slice(70, 0), lon=slice(60, 210))
            dslist.append(ds)
            del data
        gsm.close()
        if len(dslist)==1:
            dsout=dslist[0]
        else:
            dsout=xr.concat(dslist,dim="time")
        del  dslist
        return dsout