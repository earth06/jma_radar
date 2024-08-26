#! /home/oonishi/miniconda3/envs/anal_main/bin/python
import gpvutil
import argparse
import pandas as pd
import os
import yaml

description = """get GPV files from Web\n \

実行霊
ex1) ./get_gpv_from_web.py MSM 202008160900 202008160900
ex2) ./get_gpv_from_web.py GSM 202008160900 202008200900
ex3) ./get_gpv_from_web.py MSM 202008160900 202008160900 -s RISH   #RISHから取得
ex4) ./get_gpv_from_web.py SFCT 202008160900 202008200900
現時点では時刻は関係ないがいったん渡しといてください
解析雨量・短時間予報についてはテストできていないので注意
"""

urldict = {
    "RISH": "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"  # RISH
}
parser = argparse.ArgumentParser(
    description=description, formatter_class=argparse.RawTextHelpFormatter
)

# select Products
parser.add_argument("product", choices=["MSM", "GSM", "SFCT", "ANAL", "CPS3","hGSM"])
# select time range
parser.add_argument(
    "begin",
    help="begin timestamp of data you get, YYYYMMDDhhmm format, TZ=UTC if you want to consider JST please add '-z JST' option",
)
parser.add_argument(
    "end", help="end timestamp of data you get, YYYYMMDDhhmm format, TZ=UTC"
)

# timezone setting
parser.add_argument("-z", "--timezone", choices=["UTC", "JST"], default="UTC")

# distributor setting
parser.add_argument("-s", "--source", choices=["xxxx", "RISH"], default="RISH")

# SFCT initial time settign
parser.add_argument("--sfcttime", default="230000", help="sfct initial time hhmmss")

# debug mode
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()
product = args.product
format = "%Y%m%d%H%M"
begin = pd.to_datetime(args.begin, format=format)
end = pd.to_datetime(args.end, format=format)
# decide URL
URL = urldict[args.source]

with open("./gpvpath.yaml") as f:
    CONFIG = yaml.safe_load(f)
# ROOTDIR
DATADIR = CONFIG["DATADIR"]

# debugオプションで./TESTに受信したデータを書き込む

if args.debug:
    OUTPUT = DATADIR + "/TEST"
else:
    OUTPUT = DATADIR + "/" + product
# JST => UTC
if args.timezone == "JST":
    begin = begin - pd.offsets.Hour(9)
    end = end - pd.offsets.Hour(9)

##MSM
if product == "MSM":
    timerange = pd.date_range(begin, end, freq="D")
    for d in timerange:
        msmfiles = gpvutil.generate_msm_path(d, init_time_utc="180000")
        # RISHからとるときはYYYY/MM/DDをurlに追加しないといけない
        if args.source == "RISH":
            url = URL + f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        else:
            url = URL
        for f in msmfiles:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(filepath, output=OUTPUT)

# hGSM
elif product == "hGSM":
    timerange = pd.date_range(begin, end, freq="D")
    for d in timerange:
        hgsmfiles = gpvutil.generate_hgsm_path(d)
        # RISHからとるときはYYYY/MM/DDをurlに追加しないといけない
        if args.source == "RISH":
            url = URL + f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        else:
            url = URL
        for f in hgsmfiles:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(filepath, output=OUTPUT)
    
# GSM
elif product == "GSM":
    timerange = pd.date_range(begin, end, freq="D")
    for d in timerange:
        gsmfiles = gpvutil.generate_gsm_path(d)
        # RISHからとるときはYYYY/MM/DDをurlに追加しないといけない
        if args.source == "RISH":
            url = URL + f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        else:
            url = URL
        for f in gsmfiles:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(
                filepath,
                output=OUTPUT
                + "/jp/"
                + d.strftime("%H%MUTC")
                + f"/{d.year:04d}/{d.month:02d}",
            )
# SFCT
elif product == "SFCT":
    timerange = pd.date_range(begin, end, freq="D")
    for d in timerange:
        sfctfiles = gpvutil.generate_sfct_path(d, init_time_utc=args.sfcttime)
        # RISHからとるときはYYYY/MM/DDをurlに追加しないといけない
        if args.source == "RISH":
            url = URL + f"/{d.year:04d}/{d.month:02d}/{d.day:02d}"
        else:
            url = URL
        for f in sfctfiles:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(filepath, output=OUTPUT)
# ANAL
elif product == "ANAL":
    timerange = pd.date_range(begin, end, freq="H")
    url = URL
    for h in timerange:
        print(h)
        analfiles = gpvutil.generate_anal_path(h)
        for f in analfiles:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(filepath, output=OUTPUT)
# CPS3
elif product == "CPS3":
    timerange = pd.date_range(begin, end, freq="D")
    url = URL
    for d in timerange:
        cps3files = gpvutil.generate_cps3_path(d)
        for f in cps3files:
            filepath = url + "/" + f
            gpvutil.downloads_gpv(filepath, output=OUTPUT)
