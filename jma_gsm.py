import cartopy.crs as ccrs
from scipy.ndimage.filters import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as xufuncs
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


class GSM_global():
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
        ds["wind"] = xufuncs.sqrt(ds["u"]**2 + ds["v"]**2)
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

    def read_sfc(self, file, product_type="ANAL", crip="Asia",timestep=[0]):
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
                    ds = ds.sel(lat=slice(70, 0), lon=slice(60, 210))
            dslist.append(ds)
            del data
        gsm.close()
        if len(dslist)==1:
            dsout=dslist[0]
        else:
            dsout=xr.concat(dslist, dim="time")
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


class Wheather_map():
    """
    ????????????????????????????????????
    """

    def __init__(self):
        self.levels = {
            "850hPa": {
                "base_height": 1500,
                "level1": np.arange(1500-600, 1500+601, 60),
                "level2": np.arange(1500-600, 1500+601, 300),
                "wetlevels": [-273.15, 3]
            },
            "700hPa": {
                "base_height": 3000,
                "level1": np.arange(3000-600, 3001+601, 60),
                "level2": np.arange(3000-600, 3001+601, 300),
                "wetlevels": [-273.15, 3]
            },
            "500hPa": {
                "base_height": 5500,
                "level1": np.arange(5700-600, 5700+601, 60),
                "level2": np.arange(5700-600, 5700+601, 300),
            },
            "300hPa": {
                "base_height": 9600,
                "level1": np.arange(9600-1440, 9600+1441, 120),
                "wind_level": np.arange(0, 161, 20)
            },
            "SFC_FCT": {
                "level1": np.arange(1000-200, 1000+201, 4),
                "level2": np.arange(1000-200, 1000+201, 20),
                "level3": np.arange(1000-200, 1000+201, 40)
            },
            "850hPa_w_T_750hPa_omega": {
                "level1": np.arange(-60, 40, 3),
                "level2": np.arange(-120, 121, 20)
            }
        }
        self.Tlevels = {
            "warm": np.arange(-100, 50.1, 3),
            "cool": np.arange(-100, 50.1, 6)
        }
        self.figsize = (20, 12)
        self.map_extent=[90,170,10,55]
        self.fct_extent=[110,160,20,55]

    def plot_850hPa_map(self, ds, season="warm", lev=850,fig=None,ax=None,smooth=False):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        # ax.set_extent([120,150,20,50],crs=ccrs.PlateCarree())
        ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())
        # ????????????
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["850hPa"]["level1"], colors="k")
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["850hPa"]["level2"], colors="k", linewidths=3)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [1500])

        # ?????????
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(
            level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels[season], colors="k", linestyles="dashed")
        ax.clabel(cs3, cs3.levels[::2])

        # ?????????
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=lev),
                          transform=ccrs.PlateCarree(), levels=[-273.15, 3], hatches=[".."], colors=None, alpha=0)
        ax.set_title("850hPa")
        # ?????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        return fig, ax

    def plot_700hPa_map(self, ds, lev=700,fig=None,ax=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        # ax.set_extent([120,150,20,50],crs=ccrs.PlateCarree())
        ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())
        # ????????????
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["700hPa"]["level1"], colors="k")
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["700hPa"]["level2"], colors="k", linewidths=3)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [2400, 3000, 3600])

        # ?????????
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels["cool"], colors="k",
                         linestyles="dashed")
        ax.clabel(cs3, cs3.levels)

        # ?????????
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=lev),
                          transform=ccrs.PlateCarree(), levels=[-273.15, 3], hatches=[".."], colors=None, alpha=0)
        ax.set_title("700hPa")
        # ?????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        return fig, ax

    def plot_500hPa_map(self, ds, season="warm", lev=500,fig=None, ax=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())
        # ????????????
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["500hPa"]["level1"], colors="k")
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["500hPa"]["level2"], colors="k", linewidths=3)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [5100, 5700, 6300])
        # ?????????
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels[season], colors="k",
                         linestyles="dashed")
        ax.clabel(cs3, cs3.levels)
        # ?????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        ax.set_title("500hPa")
        return fig, ax

    def plot_300hPa_map(self, ds, season="warm", lev=300,ax=None, fig=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.map_extent, crs=ccrs.PlateCarree())
        # ????????????
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["300hPa"]["level1"], colors="k")
        ax.clabel(cs, cs.levels[::2])
        # ????????????
        cs4 = ax.contour(ds["lon"], ds["lat"], ds["wind"].sel(level=lev), transform=ccrs.PlateCarree(), levels=np.arange(0, 100.1, 20), colors="k",
                         linestyles="dashed")
        ax.clabel(cs4, cs4.levels)
        # ?????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        ax.set_title("300hPa")
        return fig, ax

    def _make_label(self, x):
        if x > 0:
            return "+\n"+str(int(x))
        elif x < 0:
            return "-\n"+str(int(-x))
        else:
            return str(int(x))

    def _detect_peaks_v(self, dsinput, item, filter_size=3, order=0.3, factor=1.0e0):
        lon1,lon2,lat1,lat2=self.fct_extent
        ds = dsinput.sel(level=700, lat=slice(lat2, lat1), lon=slice(lon1,lon2))
        lon = ds["lon"].values
        lat = ds["lat"].values
        val = ds[item].values
        local_max = maximum_filter(val, footprint=np.ones(
            (filter_size, filter_size)), mode='constant')
        local_min = minimum_filter(val, footprint=np.ones(
            (filter_size, filter_size)), mode="constant")

        detected_peaks_max = np.ma.array(val, mask=~(val == local_max))
        detected_peaks_min = np.ma.array(val, mask=~(val == local_min))

        # ??????????????????????????????????????????????????????order???????????????????????????
        peak_values_max = np.ma.array(detected_peaks_max, mask=~(
            detected_peaks_max >= detected_peaks_max.max() * order))
        peaks_index_max = np.where((peak_values_max.mask != True))
        peak_values_min = np.ma.array(detected_peaks_min, mask=~(
            detected_peaks_min <= detected_peaks_min.min() * order))
        peaks_index_min = np.where((peak_values_min.mask != True))

        df_peak_max = pd.DataFrame(
            data={"x": peaks_index_max[1][:], "y": peaks_index_max[0][:]})
        df_peak_min = pd.DataFrame(
            data={"x": peaks_index_min[1][:], "y": peaks_index_min[0][:]})
        df_peak = pd.concat([df_peak_max, df_peak_min]).reset_index(drop=True)
        df_peak.sort_values("y", inplace=True)
        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            df_peak.loc[idx, "value"] = val[y, x]
        df_peak.loc[:, "value"] *= factor
        df_peak["label"] = df_peak["value"].apply(self._make_label)
        df_peak["lon"] = [lon[x] for x in df_peak["x"]]
        df_peak["lat"] = [lat[y] for y in df_peak["y"]]
        return df_peak

    def _plot_peak_v(self, ax, df_peak):

        # y???????????????????????????????????????
        peak_list = []
        prev_x, prev_y = None, None
        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list.append(idx)

        # x???????????????????????????????????????
        peak_list2 = []
        df_peak_sort = df_peak.loc[peak_list].sort_values("x")
        prev_x, prev_y = None, None
        for (idx, x, y) in zip(df_peak_sort.index, df_peak_sort["x"], df_peak_sort["y"]):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list2.append(idx)
        # ??????????????????
        for lon, lat, label in zip(df_peak_sort.loc[peak_list2, "lon"], df_peak_sort.loc[peak_list2, "lat"], df_peak_sort.loc[peak_list2, "label"]):
            #ax.scatter(lon, lat, color='black',transform=ccrs.PlateCarree(),marker=".",s=1)
            ax.text(lon, lat, "\n"+label, verticalalignment="center",
                    horizontalalignment="center", transform=ccrs.PlateCarree(), fontsize=14)

    def plot_500hPa_vo_map(self, ds, lev=500,ax=None, fig=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.fct_extent, crs=ccrs.PlateCarree())
        # ????????????
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["500hPa"]["level1"], colors="k")
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["500hPa"]["level2"], colors="k", linewidths=3)
        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [5100, 5700, 6300])
        hatches = ["||"]*6+[None]*5
        volevels = np.arange(-200, 200.1, 40)
        cs3 = ax.contourf(ds["lon"], ds["lat"], ds["vo"].sel(
            level=lev)*1e6, transform=ccrs.PlateCarree(), levels=volevels, cmap="bwr", hatches=hatches, extend="both")
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["vo"].sel(
            level=lev)*1e6, transform=ccrs.PlateCarree(), levels=volevels, colors="k", extend="both", linestyles="dashed")
        # ??????????????????????????????
        df_peak = self._detect_peaks_v(
            ds, "vo", filter_size=3, order=0.5, factor=1e6)
        self._plot_peak(ax, df_peak)
        ax.set_title("500hPa height_vorticity")
        return fig, ax

    def _detect_peaks_precip(self, dsinput, filter_size=3, order=0.3):
        lon1,lon2,lat1,lat2=self.fct_extent
        ds=dsinput.sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
        lon = ds["lon"].values
        lat = ds["lat"].values
        precip = ds["precip"].values
        local_max = maximum_filter(precip, footprint=np.ones(
            (filter_size, filter_size)), mode='constant')
        detected_peaks = np.ma.array(precip, mask=~(precip == local_max))
        # ??????????????????????????????????????????????????????order???????????????????????????
        peak_values = np.ma.array(detected_peaks, mask=~(
            detected_peaks >= detected_peaks.max() * order))
        peaks_index = np.where((peak_values.mask != True))
        df_peak = pd.DataFrame(
            data={"x": peaks_index[1][:], "y": peaks_index[0][:]})

        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            df_peak.loc[idx, "value"] = precip[y, x]
        df_peak["label"] = [str(int(x)) for x in df_peak["value"]]
        df_peak["lon"] = [lon[x] for x in df_peak["x"]]
        df_peak["lat"] = [lat[y] for y in df_peak["y"]]
        return df_peak

    def _plot_peak(self, ax, df_peak):
        # y???????????????????????????????????????
        peak_list = []
        prev_x, prev_y = None, None
        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list.append(idx)

        # x???????????????????????????????????????
        peak_list2 = []
        df_peak_sort = df_peak.loc[peak_list].sort_values("x")
        prev_x, prev_y = None, None
        for (idx, x, y) in zip(df_peak_sort.index, df_peak_sort["x"], df_peak_sort["y"]):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list2.append(idx)
        # ??????????????????
        for lon, lat, label in zip(df_peak_sort.loc[peak_list2, "lon"], df_peak_sort.loc[peak_list2, "lat"], df_peak_sort.loc[peak_list2, "label"]):
            ax.scatter(lon, lat, color='black',
                       transform=ccrs.PlateCarree(), marker="+", s=100)
            ax.text(lon, lat, "\n"+label, verticalalignment="center",
                    horizontalalignment="center", transform=ccrs.PlateCarree(), fontsize=14)

    def plot_surface_ps_wind_precip(self, ds, cmap=jmacmap, alpha=0.8, ax=None, fig=None):
        # ????????????????????????
        df_peak = self._detect_peaks_precip(ds)
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.fct_extent, crs=ccrs.PlateCarree())

        cs = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                        levels=self.levels["SFC_FCT"]["level1"], colors="k")
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                         levels=self.levels["SFC_FCT"]["level2"], colors="k", linewidths=3)
        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, self.levels["SFC_FCT"]["level3"])
        # ???????????????
        cs3 = ax.contourf(ds["lon"], ds["lat"], ds["precip"], transform=ccrs.PlateCarree(),
                          levels=[1, 10, 20, 30, 40, 50], extend="max", linestyles="dashed", cmap=cmap, alpha=alpha)
        # ?????????????????????????????????????????????
        self._plot_peak(ax, df_peak)

        # ????????????
        basewind = 20
        # ?????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u10"].values/0.51, ds["v10"].values /
                 0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        ax.set_title("surface precip_pressure_wind")
        return fig, ax

    def plot_850hPa_T_wind_700hPa_omega(self, ds, cmap="none", ax=None, fig=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.fct_extent, crs=ccrs.PlateCarree())

        # 850hPa??????
        cs = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=850)-273.15, transform=ccrs.PlateCarree(),
                        colors="k", levels=self.levels["850hPa_w_T_750hPa_omega"]["level1"])
        ax.clabel(cs, cs.levels[::2])

        # 700hPa??????p????????
        hatches = ["||"]*6+[None]*7
        linestyles = ["dashed"]*6 + ["solid"] + ["dashed"]*6
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["vp"].sel(level=700), transform=ccrs.PlateCarree(), colors="none"  # cmap="bwr"
                          , levels=np.arange(-120, 121, 20), hatches=hatches, linestyles=linestyles)
        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["vp"].sel(level=700), transform=ccrs.PlateCarree(), colors="k"  # cmap="bwr"
                           , levels=np.arange(-120, 121, 20), linestyles=linestyles)
        # ??????p??????????????????
        df_peak = self._detect_peaks_v(
            ds, "vp", filter_size=3, order=0.3, factor=1.0)
        self._plot_peak(ax, df_peak)

        # 850hPa?????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=850).values/0.51, ds["v"].sel(
            level=850).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        ax.set_title("850hPa temperature 700hPa omega")
        return fig, ax

    def plot_500hPa_T_700hPa_dew_point_depression(self, ds, cmap="none", ax=None, fig=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.fct_extent, crs=ccrs.PlateCarree())
        # 500hPa?????????
        Tlevels = np.arange(-60, 40, 3)
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(
            level=500)-273.15, transform=ccrs.PlateCarree(), colors="k", linewidths=3, linestyles="solid", levels=Tlevels)
        ax.clabel(cs3, cs3.levels[::2])
        # 700hPa??????
        x = [0, 3, 6, 12, 18, 24, 30, 36, 42, 48,
             54, 60, 66, 72, 78, 84, 90, 96, 102]
        h = ["||"]*1+[None]*17
        l = ["dashed"]*2+["solid"]*16
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=700),
                          transform=ccrs.PlateCarree(), colors="none", levels=x, hatches=h)

        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=700),
                           transform=ccrs.PlateCarree(), colors="k", levels=x, linestyles=l)
        ax.clabel(cs4_2, cs4_2.levels[2::2])
        ax.set_title("500hPa T 700hPa dew point depreesion")
        return fig, ax

    def plot_850hPa_wind_equ_potential_temperature(self, ds,ax=None, fig=None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=140))
        ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        ax.coastlines(resolution="50m")
        ax.set_extent(self.fct_extent, crs=ccrs.PlateCarree())

        # 850hPa????????????????????????
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=850).values/0.51, ds["v"].sel(
            level=850).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree())
        # 850hPa???????????????
        theta_levels = np.arange(300-90, 300+90, 3)
        cs4 = ax.contour(ds["lon"], ds["lat"], ds["theta_w"],
                         transform=ccrs.PlateCarree(), colors="k", levels=theta_levels)
        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["theta_w"], transform=ccrs.PlateCarree(
        ), colors="k", levels=theta_levels[::5], linewidths=1)
        ax.clabel(cs4, cs4.levels[::2])
        ax.clabel(cs4_2, [210, 240, 270, 300, 330, 360])
        ax.set_title("850hPa wind equ potential temperature")
        return fig, ax
