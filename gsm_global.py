from scipy.ndimage.filters import maximum_filter, minimum_filter
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
import re
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

    def set_timestep_dict(self,level="sfc",gsmtype="jp_high"):
        """_summary_

        Args:
            level (str, optional):{sfc, plev}. Defaults to "sfc".
            gsmtype (str, optional): {jp,jp_high,gl,gl_high}. Defaults to "jp_high".
        """
        self.timestep_dict_all={
            "sfc":{
                "jp":{
                    "FD000-0312":list(range(1,85)),
                    "FD0315-0512":list(range(0,16)),
                    "FD0515-1100":list(range(0,44))
                },
                "jp_high":{
                    "FD0000-0100" :list(range(1,24)),
                    "FD0101-0200" :list(range(24)),
                    "FD0201-0300" :list(range(24)),
                    "FD0301-0400" :list(range(24)),
                    "FD0401-0500" :list(range(24)),
                    "FD0501-0512" :list(range(12)),
                    "FD0515-0700" :list(range(12)),
                    "FD0703-0900" :list(range(16)),
                    "FD0903-1100" :list(range(16)),
                },
            }

        }
        self.timestep_dict=self.timestep_dict_all[level][gsmtype]

    def read_sfc(self, file, product_type="ANAL", crip="Asia",timestep=[0], last_prev_precip=0):
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
        fdcode=re.findall("FD\d{4}",file)[0]
        gsm = pygrib.open(file)
        param_names = list(self.SFC_CONFIG.keys())
        if product_type == "ANAL":
            param_names.remove("precip")
            timestep=[0]
        dslist=[]
        selected_dict={}
        #先にgrib2の検索はしておく
        for key in param_names:
            namekey = self.SFC_CONFIG[key]["namekey"]
            name = self.SFC_CONFIG[key]["name"]            
            if namekey == "name":
                tmp = gsm.select(name=name)
            elif namekey == "parameterName":
                tmp = gsm.select(parameterName=name)         
            selected_dict[key]=tmp
        prev_precip=last_prev_precip
        for it,t in enumerate(timestep):
            data={}
            for key in param_names:
                dat=selected_dict[key][t]
                val=dat.values[np.newaxis,:,:]
                if key=="T2m":
                    time=np.datetime64(dat.validDate)
                #降水量データのときのみ複数タイムステップの処理のとき、前タイムステップとの差分をとるようにする
                if key=="precip":#ANALのときはprecipはkeyから外されているので無視
                    if it==0:
                        if fdcode=="FD0000":
                            #FD0000の1時間目前は降水量をキャッシュさせるだけ
                            prev_precip=val
                        elif fdcode != "FD0000":
                            #それ以外は前のファイルの末尾の降水量データとの差分をとる
                            prev_precip=np.copy(val)
                            val=val-last_prev_precip
                            val[val<0]=0
                    else:
                        #それ以外のときは差分をとって,負になったら０に戻す
                        new_prev_precip=np.copy(val)
                        val=val - prev_precip
                        val[val<0]=0
                        prev_precip=new_prev_precip

                data[key] = (["time","lat", "lon"], val,
                             {"title": self.SFC_CONFIG[key]["name"],
                              "units": self.SFC_CONFIG[key]["units"]})
            #格子情報を定義する。
            lat, lon = dat.latlons()
            lat = lat[:, 0]
            lon = lon[0, :]
            coords = {
                "time":(("time",[time])),
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
        del dslist
        return dsout, prev_precip
    
    def read_sfc_all(self, files):
        ds_list=[]
        for i,f in enumerate(files):
            fdcode=re.findall("FD\d{4}-\d{4}",f)[0]
            if i==0:
                dstmp,prev_precip=self.read_sfc(f,product_type="FCT",timestep=self.timestep_dict[fdcode])
            else:
                dstmp,prev_precip=self.read_sfc(f,product_type="FCT",timestep=self.timestep_dict[fdcode],last_prev_precip=prev_precip)
            ds_list.append(dstmp) 
        dsall=xr.concat(ds_list,dim="time")
        del ds_list
        return dsall    

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
                data[key] = (["time","level", "lat", "lon"], np.array(dat_list)[np.newaxis,:,:,:],
                             {"title": self.PLEV_CONFIG[key]["name"],
                              "units": self.PLEV_CONFIG[key]["units"]})
                del dat_list
            lat, lon = dat.latlons()
            lat = lat[:, 0]
            lon = lon[0, :]
            time=np.datetime64(dat.validDate)
            coords = {
                "time":[time],
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
    
    def to_netcdf(self,ds:xr.Dataset,filename:str,compression=True):
        encoding={}
        for param in ds.data_vars:
            encoding[param]={"zlib":True, "complevel":5, "dtype":"float32"}
        ds.to_netcdf(filename, encoding=encoding)
        return 0