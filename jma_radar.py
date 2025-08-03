import xarray as xr
import numpy as np
import pandas as pd
from numba import jit
from jma_radar_core import decode_int16, decode_int32, unpack_runlength


class JMA_Radar:
    """
    define JMA Radar GPV data structure

    Attributes
    --------------
    values : numpy.ndarray
        precipitation intensity
    time : str
        timestamp of the data
    lat : numpy.ndarray
        latitude
    lon : numpy.ndarray
        longitude
    dx : float
        interval of longitude
    dy : float
        interval of latitude

    Methods
    ----------
    read_radar(filepath)
        read grib2 file of JMA Radar
    latlon()
        return meshed longitude and latitude
    to_xarray()
        convert this instance into xarray.Dataset
    """

    @classmethod
    def read_file(self, filepath):
        """
        read grib2 file of JMA Radar GPV

        Parameters
        ----------
        filepath :str
            file path of JMA Radar GPV

        Returns
        -----------
        jradar : JMA_Radar
        """

        with open(filepath, "br") as f:
            # section0
            buf0 = np.frombuffer(f.read(16), dtype=">u1")
            # section1
            buf1 = np.frombuffer(f.read(21), ">u1")
            year = decode_int16(buf1, 12)
            month, day, hour, minute, sec = buf1[14:19]
            timestamp = (
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}"
            )
            # section3
            buf3 = np.frombuffer(f.read(72), dtype=">u1")
            nx = decode_int32(buf3, 30)
            ny = decode_int32(buf3, 34)
            ye = decode_int32(buf3, 46) * 1e-6
            x0 = decode_int32(buf3, 50) * 1e-6
            y0 = decode_int32(buf3, 55) * 1e-6
            xe = decode_int32(buf3, 59) * 1e-6
            dx = decode_int32(buf3, 63) * 1e-6
            dy = decode_int32(buf3, 67) * 1e-6
            # section4
            buf4 = np.frombuffer(f.read(82), dtype="u1")
            # section5
            buf5 = np.frombuffer(f.read(519), dtype="u1")
            datanum = decode_int32(buf5, 5)
            Vbit = int(buf5[11])
            VMAX = int(decode_int16(buf5, 12))
            level = int(decode_int16(buf5, 14))
            scale_factor = int(buf5[16])
            RR = np.zeros(level, dtype="float32")
            for m in range(level):
                RR[m] = float(decode_int16(buf5, 15 + 2 * m)) / (10.0**scale_factor)
            # section6
            buf6 = np.frombuffer(f.read(6), dtype="u1")
            # section7
            buf7 = np.frombuffer(f.read(5), dtype=">u1")
            length = decode_int32(buf7, 0) - 5
            data = np.frombuffer(f.read(length), dtype=">u1")
            aa = 2**Vbit - 1 - VMAX
            import pdb

            pdb.set_trace()
            rrain = unpack_runlength(data, nx, ny, length, aa, VMAX, RR)
            rrain = rrain.reshape((ny, nx))[::-1, :]
            rrain[rrain < 0] = np.nan
            jradar = JMA_Radar(rrain, nx, ny, timestamp, x0, xe, dx, y0, ye, dy, RR)
            return jradar

    def __init__(self, rr, nx, ny, timestamp, x0, xe, dx, y0, ye, dy, RR):
        """
        Parameters
        -----------
        rr : numpy.ndarray
            precipitation intensity(1D)
        nx,ny : int
            the number of grid x and y
        timestamp : str
            timestamp of the data(YYYY-MM-DDThh:mm:ss)
        x0,xe,dx : float
            begin,end,interval of longitude(degrees)
        y0,ye,dy : float
            begin,end,interval of latitude(degrees)
        """
        self.nx = nx
        self.ny = ny
        self.values = rr
        self.time = timestamp
        self.lat = np.linspace(y0, ye, ny)
        self.lon = np.linspace(x0, xe, nx)
        self.dx = dx
        self.dy = dy
        self.RR = RR

    def latlon(self):
        xx, yy = np.meshgrid(self.lon, self.lat)
        return xx, yy

    def to_xarray(self):
        """
        Returns
        --------
        ds : xarray.Dataset
        """
        values = {
            "rr": (
                ["time", "lat", "lon"],
                self.values.reshape((1, self.ny, self.nx)),
                {"units": "[mm/hr]", "long name": "precipitation intensity"},
            )
        }
        coords = {
            "time": pd.to_datetime([self.time], format="%Y-%m-%dT%H:%M:%S"),
            "lat": (("lat", self.lat, {"units": "degrees_north"})),
            "lon": (("lon", self.lon, {"units": "degrees_east"})),
        }
        attrs = {"title": "JMA precipitation intensity", "timezone": "UTC"}
        ds = xr.Dataset(values, coords)
        return ds
