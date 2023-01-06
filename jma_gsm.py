import pathlib
import sys
sys.path.append(pathlib.Path(__file__).resolve().parent)
import custom_colormap
from gsm_global import GSMGlobal
from weathermap import WheatherMap
jmacmap = custom_colormap.get_jmacmap2()
