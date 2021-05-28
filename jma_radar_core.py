import numpy as np
from numba import jit

def decode_int32(buf,begin,swap=False):
    if swap:
        val_int32=np.frombuffer(buf[begin+3].tobytes()+buf[begin+2].tobytes()+buf[begin+1].tobytes()+buf[begin].tobytes(),dtype=">u4")       
    else:
        val_int32=np.frombuffer(buf[begin].tobytes()+buf[begin+1].tobytes()+buf[begin+2].tobytes()+buf[begin+3].tobytes(),dtype=">u4")  
    return val_int32[0]

def decode_int16(buf,begin,swap=False):
    if swap:
        val_int16=np.frombuffer(buf[begin+1].tobytes()+buf[begin].tobytes(),dtype=">u2")
    else:
        val_int16=np.frombuffer(buf[begin].tobytes()+buf[begin+1].tobytes(),dtype=">u2")
    return val_int16[0]

@jit#('int64(uint8[:],int64,int64,int64,int64)')
def check_next(data,cur_pos2,n,length,VMAX):
    while True:
        if int(data[cur_pos2+1]) <= VMAX:
            return n
        n=n+1
        if cur_pos2+n > length:
            return n
        cur_pos2+=1

@jit#('float32[:](uint8[:],uint32,uint32,int64,int64,int64,float32[:])')
def unpack_runlength(data,nx,ny,length,aa,VMAX,RR):
    cur_pos=0
    ij=0
    rrain=np.zeros(nx*ny,dtype="float32")
    while(cur_pos < length):
        d1=int(data[cur_pos])
        n1=int(data[cur_pos+1])
        n=0
        if (n1 > VMAX):
            n=check_next(data,cur_pos+1,n,length,VMAX)
            total_num=0
            for nn in range(n+1):#1個多くしないとダメ
                total_num+=int((int(data[cur_pos+1+nn]) - VMAX-1)*(aa**nn))
            total_num+=1
            for m in range(total_num):
                if d1==0:
                    rrain[ij+m]=-999.9
                else:
                    rrain[ij+m]=RR[d1]
            ij+=total_num
            cur_pos+=(n+2)
        else:
            if(d1==0):
                rrain[ij]=-999.9
            else:
                rrain[ij]=RR[d1]
            ij+=1
            cur_pos+=1
    return rrain