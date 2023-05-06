import cartopy.crs as ccrs
from scipy.ndimage.filters import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from metpy.units import units
import pathlib
import sys
import scipy.ndimage as ndimage
sys.path.append(pathlib.Path(__file__).resolve().parent)
import custom_colormap

jmacmap = custom_colormap.get_jmacmap2()


class WeatherMap():
    """
    各天気図を描画するクラス
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
        self.figsize = (9, 6)
        self.map_extents={
            "EastAsia":[115,164,20,55],
            "Asia":[90,170,10,55],
            "Japan":[120,150,20,50]
        }
        self.boldlinewidth=1
        self.linewidth=0.5
        self.barbwidth=0.25
        self.hatchwidth=0.5
        plt.rcParams['hatch.linewidth'] =self.hatchwidth
        plt.rcParams['contour.negative_linestyle'] = 'solid'
        self.peak_fontsize=9
        self.outdir="./"

    def _generate_figure(self,fig=None,ax=None,map="EastAsia"):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(
                1, 1, 1, projection=ccrs.LambertConformal(central_longitude=140,central_latitude=30))
        ax.coastlines(resolution="50m")
        gl=ax.gridlines(draw_labels=True, xlocs=plt.MultipleLocator(
            10), ylocs=plt.MultipleLocator(10))
        gl.right_labels=False
        ax.set_extent(self.map_extents[map], crs=ccrs.PlateCarree())

        return fig,ax

    def plot_surface_ps_wind(self, ds, ax=None, fig=None, map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        cs = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                        levels=self.levels["SFC_FCT"]["level1"], colors="k", linewidths=self.linewidth)
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                         levels=self.levels["SFC_FCT"]["level2"], colors="k", linewidths=self.boldlinewidth)
        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, self.levels["SFC_FCT"]["level3"])
        # 等風速線
        basewind = 20
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u10"].values/0.51, ds["v10"].values /
                 0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.barbwidth)
        ax.set_title("surface precip_pressure_wind")
        return fig, ax

    def plot_850hPa_map(self, ds, season="warm", lev=850,fig=None,ax=None,smooth=False, map="EastAsia"):
        
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        #等高度線
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["850hPa"]["level1"], colors="k",linewidths=self.linewidth)
        #太線
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["850hPa"]["level2"], colors="k", linewidths=self.boldlinewidth)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [1500])

        # 等温線
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(
            level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels[season], colors="k", linestyles="dashed",
            linewidths=self.linewidth)
        ax.clabel(cs3, cs3.levels[::2])

        # 湿り域
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=lev),
                          transform=ccrs.PlateCarree(), levels=[-273.15, 3], hatches=["..."], colors=None, alpha=0,
                          linewidths=self.linewidth)
        ax.set_title("850hPa")
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.linewidth)
        return fig, ax

    def plot_700hPa_map(self, ds, lev=700,fig=None,ax=None,map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 等高度線
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["700hPa"]["level1"], colors="k",
            linewidths=self.linewidth)
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["700hPa"]["level2"], colors="k", linewidths=self.boldlinewidth)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [2400, 3000, 3600])

        # 等温線
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels["cool"], colors="k",
                         linestyles="dashed", linewidths=self.boldlinewidth)
        ax.clabel(cs3, cs3.levels)

        # 湿り域
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=lev),
                          transform=ccrs.PlateCarree(), levels=[-273.15, 3], hatches=["..."]
                          , colors=None, alpha=0, linewidths=self.linewidth)
        ax.set_title("700hPa")
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree()
            ,linewidth=self.barbwidth)
        return fig, ax

    def plot_500hPa_map(self, ds, season="warm", lev=500,fig=None, ax=None, map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 等高度線
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["500hPa"]["level1"], colors="k", linewidths=self.linewidth)
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["500hPa"]["level2"], colors="k", linewidths=self.boldlinewidth)

        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [5100, 5700, 6300])
        # 等温線
        baseT = 0
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=lev)-273.15, transform=ccrs.PlateCarree(), levels=self.Tlevels[season], colors="k",
                         linestyles="dashed", linewidths=self.linewidth)
        ax.clabel(cs3, cs3.levels)
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.linewidth)
        ax.set_title("500hPa")
        return fig, ax

    def plot_300hPa_map(self, ds, season="warm", lev=300,ax=None, fig=None, map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 等高度線
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["300hPa"]["level1"], colors="k", linewidths=self.boldlinewidth)
        ax.clabel(cs, cs.levels[::2])
        # 等風速線
        cs4 = ax.contour(ds["lon"], ds["lat"], ds["wind"].sel(level=lev), transform=ccrs.PlateCarree(), levels=np.arange(0, 100.1, 20), colors="k",
                         linestyles="dashed",linewidths=self.linewidth)
        ax.clabel(cs4, cs4.levels)
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=lev).values/0.51, ds["v"].sel(
            level=lev).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.barbwidth)
        ax.set_title("300hPa")
        return fig, ax

    def _make_label(self, x):
        if x > 0:
            return "+\n"+str(int(x))
        elif x < 0:
            return "―\n"+str(int(-x))
        else:
            return str(int(x))

    def _detect_peaks_v(self, dsinput, item, filter_size=3, order=0.3, factor=1.0e0, map="EastAsia"):
        lon1,lon2,lat1,lat2=self.map_extents[map]
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

        # 小さいピーク値を排除（最大ピーク値のorder倍のピークは排除）
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

        # y方向で距離が近い点を間引く
        peak_list = []
        prev_x, prev_y = None, None
        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list.append(idx)

        # x方向で距離が近い点を間引く
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
        # プロットする
        for lon, lat, label in zip(df_peak_sort.loc[peak_list2, "lon"], df_peak_sort.loc[peak_list2, "lat"], df_peak_sort.loc[peak_list2, "label"]):
            #ax.scatter(lon, lat, color='black',transform=ccrs.PlateCarree(),marker=".",s=1)
            ax.text(lon, lat, label, verticalalignment="center",
                    horizontalalignment="center", transform=ccrs.PlateCarree(), fontsize=self.peak_fontsize, weight="bold")

    def plot_500hPa_vo_map(self, ds, lev=500,ax=None, fig=None,map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 等高度線
        cs = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(
            level=lev), transform=ccrs.PlateCarree(), levels=self.levels["500hPa"]["level1"], colors="k", linewidths=self.linewidth)
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["hgt"].sel(level=lev), transform=ccrs.PlateCarree(
        ), levels=self.levels["500hPa"]["level2"], colors="k", linewidths=self.boldlinewidth)
        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, [5100, 5700, 6300])
        hatches =[None]*6+["|||"]*5
        volevels = np.arange(-200, 200.1, 40)
        cs3 = ax.contourf(ds["lon"], ds["lat"], ds["vo"].sel(
            level=lev)*1e6, transform=ccrs.PlateCarree(), levels=volevels, colors="None",
            hatches=hatches, extend="both")
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["vo"].sel(
            level=lev)*1e6, transform=ccrs.PlateCarree(), levels=volevels,colors="k", extend="both", linestyles="dashed"
            ,linewidths=self.linewidth)
        # 渦度の極大値プロット
        # df_peak = self._detect_peaks_v(
        #     ds, "vo", filter_size=3, order=0.5, factor=1e6, map=map)
        # self._plot_peak_v(ax, df_peak)
        ax.set_title("500hPa height_vorticity")
        return fig, ax

    def _detect_peaks_precip(self, dsinput, filter_size=3, order=0.3,map="EastAsia"):
        lon1,lon2,lat1,lat2=self.map_extents[map]
        ds=dsinput.sel(lon=slice(lon1,lon2),lat=slice(lat2,lat1))
        lon = ds["lon"].values
        lat = ds["lat"].values
        precip = ds.values
        local_max = maximum_filter(precip, footprint=np.ones(
            (filter_size, filter_size)), mode='constant')
        detected_peaks = np.ma.array(precip, mask=~(precip == local_max))
        # 小さいピーク値を排除（最大ピーク値のorder倍のピークは排除）
        peak_values = np.ma.array(detected_peaks, mask=~(
            detected_peaks >= detected_peaks.max() * order))
        peaks_index = np.where((peak_values.mask != True))
        df_peak = pd.DataFrame(
            data={"x": peaks_index[1][:], "y": peaks_index[0][:]})

        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            df_peak.loc[idx, "value"] = precip[y, x]
        df_peak["label"] = ["+\n"+str(int(x)) for x in df_peak["value"]]
        df_peak["lon"] = [lon[x] for x in df_peak["x"]]
        df_peak["lat"] = [lat[y] for y in df_peak["y"]]
        return df_peak

    def _plot_peak(self, ax, df_peak):
        # y方向で距離が近い点を間引く
        peak_list = []
        prev_x, prev_y = None, None
        for idx, (x, y) in enumerate(zip(df_peak["x"], df_peak["y"])):
            if (prev_x is not None):
                r = np.sqrt((x-prev_x)**2 + (y-prev_y)**2)
                if r < 5:
                    continue
            prev_x, prev_y = x, y
            peak_list.append(idx)

        # x方向で距離が近い点を間引く
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
        # プロットする
        for lon, lat, label in zip(df_peak_sort.loc[peak_list2, "lon"], df_peak_sort.loc[peak_list2, "lat"], df_peak_sort.loc[peak_list2, "label"]):
            ax.scatter(lon, lat, color='black',
                       transform=ccrs.PlateCarree(), marker=".", s=1)
            ax.text(lon, lat, "\n"+label, verticalalignment="center",
                    horizontalalignment="center", transform=ccrs.PlateCarree(), fontsize=self.peak_fontsize)

    def plot_surface_ps_wind_precip(self, ds, cmap=jmacmap, alpha=0.8,ds_prev_precip=None ,ax=None, fig=None, map="EastAsia"):
        if ds_prev_precip is None:
            precip=ds["precip"]
        else:
            print("calc differential precip")
            precip=ds["precip"] - ds_prev_precip
        # 極大値を見つける
        df_peak = self._detect_peaks_precip(precip, map=map)
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        cs = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                        levels=self.levels["SFC_FCT"]["level1"], colors="k", linewidths=self.linewidth)
        cs2 = ax.contour(ds["lon"], ds["lat"], ds["pmsl"]*1e-2, transform=ccrs.PlateCarree(),
                         levels=self.levels["SFC_FCT"]["level2"], colors="k", linewidths=self.boldlinewidth)
        ax.clabel(cs, cs.levels[::2])
        ax.clabel(cs2, self.levels["SFC_FCT"]["level3"])

        # 等降水量線
        cs3 = ax.contourf(ds["lon"], ds["lat"], precip, transform=ccrs.PlateCarree(),
                          levels=[1, 10, 20, 30, 40, 50], extend="max", linestyles="dashed", cmap=cmap, alpha=alpha, linewidths=self.linewidth)
        # 降水量の極大点に数値を記入する
        self._plot_peak(ax, df_peak)

        # 等風速線
        basewind = 20
        # 矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u10"].values/0.51, ds["v10"].values /
                 0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.barbwidth)
        ax.set_title("surface precip_pressure_wind")
        return fig, ax

    def plot_850hPa_T_wind_700hPa_omega(self, ds, cmap="none", ax=None, fig=None, map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 850hPa気温
        cs = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(level=850)-273.15, transform=ccrs.PlateCarree(),
                        colors="k", levels=self.levels["850hPa_w_T_750hPa_omega"]["level1"], linewidths=self.boldlinewidth)
        ax.clabel(cs, cs.levels[::2])

        # 700hPa鉛直p速度ω
        hatches = ["|||"]*6+[None]*7
        linestyles = ["dashed"]*6 + ["solid"] + ["dashed"]*6
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["vp"].sel(level=700), transform=ccrs.PlateCarree(), colors="none"  # cmap="bwr"
                          , levels=np.arange(-120, 121, 20), hatches=hatches, linestyles=linestyles, linewidths=self.linewidth)
        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["vp"].sel(level=700), transform=ccrs.PlateCarree(), colors="k"  # cmap="bwr"
                           , levels=np.arange(-120, 121, 20), linestyles=linestyles, linewidths=self.linewidth)
        # 鉛直p速度の極大値
        df_peak = self._detect_peaks_v(
            ds, "vp", filter_size=3, order=0.3, factor=1.0)
        self._plot_peak(ax, df_peak)

        # 850hPa矢羽根
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=850).values/0.51, ds["v"].sel(
            level=850).values/0.51, length=6, regrid_shape=12, transform=ccrs.PlateCarree(), linewidth=self.barbwidth)
        ax.set_title("850hPa temperature 700hPa omega")
        return fig, ax

    def plot_500hPa_T_700hPa_dew_point_depression(self, ds, cmap="none", ax=None, fig=None,map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 500hPa等温線
        Tlevels = np.arange(-60, 40, 3)
        cs3 = ax.contour(ds["lon"], ds["lat"], ds["T"].sel(
            level=500)-273.15, transform=ccrs.PlateCarree(), colors="k", linewidths=self.boldlinewidth, linestyles="solid", levels=Tlevels)
        ax.clabel(cs3, cs3.levels[::2])
        # 700hPa湿数
        x = [0, 3, 6, 12, 18, 24, 30, 36, 42, 48,
             54, 60, 66, 72, 78, 84, 90, 96, 102]
        h = ["|||"]*1+[None]*17
        l = ["dashed"]*2+["solid"]*16
        cs4 = ax.contourf(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=700),
                          transform=ccrs.PlateCarree(), colors="none", levels=x, hatches=h, linewidths=self.linewidth)

        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["T-Tw"].sel(level=700),
                           transform=ccrs.PlateCarree(), colors="k", levels=x, linestyles=l, linewidths=self.linewidth)
        ax.clabel(cs4_2, cs4_2.levels[2::2])
        ax.set_title("500hPa T 700hPa dew point depreesion")
        return fig, ax

    def plot_850hPa_wind_equ_potential_temperature(self, ds,ax=None, fig=None, map="EastAsia"):
        fig,ax=self._generate_figure(fig=fig,ax=ax,map=map)
        # 850hPa風矢羽根プロット
        ax.barbs(ds["lon"], ds["lat"], ds["u"].sel(level=850).values/0.51, ds["v"].sel(
            level=850).values/0.51, length=6, regrid_shape=18, linewidth=self.barbwidth ,transform=ccrs.PlateCarree())
        # 850hPa相当温位線
        theta_levels = np.arange(300-90, 300+90, 3)
        cs4 = ax.contour(ds["lon"], ds["lat"], ds["theta_w"],
                         transform=ccrs.PlateCarree(), colors="k", levels=theta_levels, linewidths=self.linewidth)
        cs4_2 = ax.contour(ds["lon"], ds["lat"], ds["theta_w"], transform=ccrs.PlateCarree(
        ), colors="k", levels=theta_levels[::5], linewidths=self.boldlinewidth)
        ax.clabel(cs4, cs4.levels[::2])
        ax.clabel(cs4_2, [210, 240, 270, 300, 330, 360])
        ax.set_title("850hPa wind equ potential temperature")
        return fig, ax
