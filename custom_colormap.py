from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
import numpy as np
"""
ex)普通のカラーマップとして扱う
plt.contourf(xx,yy,....
    ,cmap=custom_colorbar.get_gscmap()...)

ex) 等高線とその色を手動で与える(カラーマップを使わないver)
plt.contourf(xx,yy,...
    levels=[0,0.1,0.5,1,3,5,10,15,20,30,50],
    colors=custom_colorbar.get_gscolors())

ex) 手動で決めた等高線に対して色を適当に割り振る
from matplotlib.colors import BoundaryNorm
levels=[0,0.1,0.5,1,3,5,10,15,20,30,50]
norm=BoundaryNorm(levels,ncolors=256)
plt.contourf(xx,yy,
    levels=levels,norm=norm,cmap=custom_colorbar.get_gscmap()...)

"""
def get_gscolors():
    gscolors=np.array([
        [160,0 ,200,1],#purple
        [130,0 ,220,1],#dark purple
        [30 ,60,255,1],#dark blue
        [0, 160,255,1],#medium blue
        [0, 200,200,1],#light blue
        [0, 210,140,1],#aqua
        [0,220,0,1],   #green
        [160,230,50,1],#yellow/green
        [230,220,50,1],#yellow
        [230,175,45,1],#dark yellow
        [240,130,40,1],#orange
        [250,60,60,1], #red
        [240,0,130,1]  #magenta
        ],dtype=np.float32)
    gscolors[:,:3]/=256 #RGBを0-1の範囲に変換
    return gscolors

def get_jmacolors():
    jmacolors=np.array([
        [242,242,242,1],#white
        [160,210,255,1],#white blue
        [33 ,140,255,1],#lightblue
        [0  ,65 ,255,1],#blue
        [250,245,0,1],#yellow
        [255,153,0,1],#orange
        [255,40,0,1],#red
        [180,0,104,1]#magenta
    ],dtype=np.float32)
    jmacolors[:,:3] /=256
    return jmacolors

def get_jmacmap2():
    l1,l2,l3,l4,l5,l6,l7,l8=np.linspace(0,1,8)
    l4=0.5
    cdict={
    "red":[[l1,0.9453125,0.9453125],
            [l2,0.625,0.625 ],
            [l3,0.12890625,0.12890625],
            [l4,0,0.9765625],#
            [l5,0.9765625,0.9765625],
            [l6,0.99609375,0.99609375],
            [l7,0.99609375,0.99609375],
            [l8,0.703125,0.703125]]
    ,
    
    "green":[[l1,0.9453125,0.9453125],
            [l2,0.8203125,0.8203125],
            [l3,0.546875,0.546875 ],
            [l4,0.25390625,0.95703125],#
            [l5,0.95703125,0.95703125],
            [l6,0.59765625,0.59765625],
            [l7,0.15625 ,0.15625],
            [l8, 0. ,0 ]]
        
    ,
    "blue":[[l1,0.9453125,0.9453125 ],
            [l2,0.99609375,0.99609375],
            [l3,0.99609375,0.99609375],
            [l4,0.99609375,0],#
            [l5,0.   ,0],
            [l6,0.   ,0],
            [l7,0,0],
            [l8,0.40625,0.40625 ]]
    }
    cmap=LinearSegmentedColormap("jma",segmentdata=cdict,N=256)
    return cmap


def get_jwacolors():
    jwacolors=np.array([
        [0,255,253,1],#aqua
        [0,0,226,1],#blue
        [28,150,0,1],#green
        [254,255,5,1],#yellow
        [254,255,5,1],#orange
        [253,3,242,1],#pink
        [255,0,0,1],#red
    ],dtype=np.float32)
    jwacolors[:,:3] /=256
    return jwacolors

def get_jmalevels():
    return [0,1,5,10,20,30,50,80]

def get_jwalevels():
    return [0,1,5,10,15,25,30,50]

def get_gscmap():
    gscolors=get_gscolors()
    gscmap=LinearSegmentedColormap.from_list("gscmap",colors=gscolors)
    return gscmap

def get_jmacmap():
    jmacolors=get_jmacolors()
    jmacmap=LinearSegmentedColormap.from_list("jmacmap",colors=jmacolors)
    return jmacmap

def get_jwacmap():
    jwacolors=get_jwacolors()
    jwacmap=LinearSegmentedColormap.from_list("jwacmap",colors=jwacolors)
    return jwacmap

def make_cmap(colors,cname="custom"):
    cmap=LinearSegmentedColormap.from_list(cname,colors=colors)
    return cmap

def get_norm(levels,ncolors=256,extend="max"):
    return BoundaryNorm(levels,ncolors=ncolors,extend=extend)