import xarray as xr
import numpy as np
import pandas as pd
from numba import jit
from jma_radar_core import decode_int16,decode_int32,unpack_runlength
from jma_srf import JMA_SRF
class JMA_SRF15H(JMA_SRF):
    @classmethod
    def read_file(self,filepath):
        with open(filepath,"br") as f:
            # section0
            buf0=np.frombuffer(f.read(16),dtype=">u1")
            
            # section1
            buf1=np.frombuffer(f.read(21),">u1")
            year=decode_int16(buf1,12)
            month,day,hour,minute,sec=buf1[14:19]
            timestamp=f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}"

            #section3
            buf3=np.frombuffer(f.read(72),dtype=">u1")
            nx=decode_int32(buf3,30)
            ny=decode_int32(buf3,34)
            ye=decode_int32(buf3,46)*1e-6
            x0=decode_int32(buf3,50)*1e-6
            y0=decode_int32(buf3,55)*1e-6
            xe=decode_int32(buf3,59)*1e-6
            dx=decode_int32(buf3,63)*1e-6
            dy=decode_int32(buf3,67)*1e-6
            values=np.zeros((9,ny,nx),dtype="float32")
            for ih in range(9):
                #section4
                buf4=np.frombuffer(f.read(66),dtype="u1")
                #section5　節の長さを取得するところから
                buf5_4=np.frombuffer(f.read(4),dtype=">u1")
                buf5_length=decode_int32(buf5_4,0)
                buf5=np.frombuffer(f.read(buf5_length-4),dtype=">u1")
                Vbit=int(buf5[7]) #12
                VMAX=decode_int16(buf5,8)#13~14 VMAX
                level=decode_int16(buf5,10)#15~16 level
                scale_factor=buf5[12] #17
                RR=np.zeros(level,dtype="float32")
                for m in range(level):
                    RR[m]=float(decode_int16(buf5, (16-4+1+2*m))) / (10.0**scale_factor)
                #section6
                buf6=np.frombuffer(f.read(6),dtype=">u1")
                ## section7
                buf7_head=np.frombuffer(f.read(5),dtype=">u1")
                length7=decode_int32(buf7_head,0)-5
                data=np.frombuffer(f.read(length7),dtype=">u1")
                aa=(2**Vbit -1-VMAX)
                rrain=unpack_runlength(data,nx,ny,length7,aa,VMAX,RR)
                values[ih,:,:]=rrain.reshape((ny,nx))[::-1,:]
            jmasrf=JMA_SRF15H(values,nx,ny,timestamp,x0,xe,dx,y0,ye,dy)
        return jmasrf
    
    def __init__(self,values,nx,ny,timestamp,x0,xe,dx,y0,ye,dy):
        self.nx=nx
        self.ny=ny
        values[values==-999.9]=np.nan
        self.values=values
        self.time=pd.to_datetime(timestamp)
        self.datetime=pd.date_range(self.time+pd.offsets.Hour(7)
                                   ,self.time+pd.offsets.Hour(15),freq="H")
        self.lat=np.linspace(y0,ye,ny)
        self.lon=np.linspace(x0,xe,nx)