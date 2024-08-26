import requests
import os
import pandas as pd
from tqdm import tqdm
from time import sleep
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import yaml

urllib3.disable_warnings(InsecureRequestWarning)

proxies = None

with open("./gpvpath.yaml") as f:
    CONFIG = yaml.safe_load(f)


def downloads_gpv(url, output="MSM", debug=False):
    os.makedirs(f"{output}/", exist_ok=True)
    basename = os.path.basename(url)
    print(f"getting {url}")
    ##

    try:
        datasize = int(
            requests.head(url, proxies=proxies, verify=False).headers["content-length"]
        )
        data = requests.get(url, stream=True, proxies=proxies, verify=False)
        status = data.status_code
        count = 0
        if status == 403:
            print(f"[Error] {status} code error")
            # 2回tryしてダメなら取得をやめる
            for i in range(5):
                print("[Info] Preparing for Retry")
                sleep(120)
                data = requests.get(url, stream=True, proxies=proxies, verify=False)
                status = data.status_code
                if status >= 400 and count < 2:
                    count += 1
                    continue
                elif status >= 400 and count >= 3:
                    print(f"[Error] Retry also failed by status code {status}")
                    return 1
                else:
                    break

        elif status >= 400:
            print(f"[Error] {status} code error")
            raise Exception
        pbar = tqdm(total=datasize, unit="B", unit_scale=True)
        with open(f"{output}/{basename}", mode="wb") as f:
            for chunk in data.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))
        print(f"save to {output}/{basename}")
        pbar.close()
        sleep(1)
        return 0
    except Exception as e:
        with open(CONFIG["LOGDIR"] + "/failed_files.txt", "ta") as flog:
            flog.write(url + "\n")
        return 1


# MSM
def generate_msm_path(date, init_time_utc="180000"):
    yyyymmddhhmmss = date.strftime("%Y%m%d") + init_time_utc
    files = []
    for h1_h2 in ["00-15", "16-33", "34-39"]:
        f = f"Z__C_RJTD_{yyyymmddhhmmss}_MSM_GPV_Rjp_Lsurf_FH{h1_h2}_grib2.bin"
        files.append(f)
    return files


# GSM
def generate_gsm_path(date):
    """
    Parameters:
    ------------
    date :pd.Datatime
        "初期時刻"(YYYYMMDD0000 or YYYYMMDD12000)
    """
    yyyymmddhhmmss = date.strftime("%Y%m%d%H%M") + "00"
    files = []
    init_time = date.strftime("%H%M")
    labels = {
        "0000": ["0000-0312", "0315-0512", "0515-1100"],
        "1200": ["0000-0312", "0315-0800", "0803-1100"],
    }
    for h1_h2 in labels[init_time]:
        f = f"Z__C_RJTD_{yyyymmddhhmmss}_GSM_GPV_Rjp_Lsurf_FD{h1_h2}_grib2.bin"
        files.append(f)
    return files


def generate_hgsm_path(date):
    """_summary_

    Args:
        date (_type_): _description_

    Returns:
        _type_: _description_
    """
    yyyymmddhhmmss = date.strftime("%Y%m%d%H%M") + "00"
    files = []
    init_time = date.strftime("%H%M")
    timelabels = [
        "0000-0100",
        "0101-0200",
        "0201-0300",
        "0301-0400",
        "0401-0500",
        "0501-0512",
        "0515-0700",
        "0703-0900",
        "0903-1100",
    ]
    labels = {
        "0000": timelabels,
        "1200": timelabels,
    }
    for h1_h2 in labels[init_time]:
        f = f"Z__C_RJTD_{yyyymmddhhmmss}_GSM_GPV_Rjp_Gll0p1deg_Lsurf_FD{h1_h2}_grib2.bin"
        files.append(f)
    return files


# ANAL
def generate_anal_path(date):
    yyyymmddhhmmss = date.strftime("%Y%m%d%H%M") + "00"
    file = f"Z__C_RJTD_{yyyymmddhhmmss}_SRF_GPV_Ggis1km_Prr60lv_ANAL_grib2.bin"
    return [file]


# SFCT
def generate_sfct_path(date, init_time_utc="230000"):
    # def generate_sfct_path(date):
    if init_time_utc is not None:
        yyyymmddhhmmss = date.strftime("%Y%m%d") + init_time_utc
    else:
        yyyymmddhhmmss = date.strftime("%Y%m%d%H%M%S")
    files = []
    f01_06H = f"Z__C_RJTD_{yyyymmddhhmmss}_SRF_GPV_Ggis1km_Prr60lv_FH01-06_grib2.bin"
    f07_15H = f"Z__C_RJTD_{yyyymmddhhmmss}_SRF_GPV_Gll5km_Prr60lv_FH07-15_grib2.bin"
    return [f01_06H, f07_15H]


# CPS3
def generate_cps3_path(date):
    yyyyMMddhhmmss = date.strftime("%Y%m%d") + "000000"
    # 気温
    f1 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lh2_Ptt_Emb_grib2.bin"
    # 降水量
    f2 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lsurf_Prr_Emb_grib2.bin"
    # 海面気圧
    f3 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lsurf_Ppp_Emb_grib2.bin"
    # 海面水温
    f4 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lsurf_Pss_Emb_grib2.bin"
    # RH @850hPa
    f5 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lp850_Prh_Emb_grib2.bin"
    # wind @850hPa
    f6 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lp850_Pwu_Emb_grib2.bin"
    f7 = f"Z__C_RJTD_{yyyyMMddhhmmss}_EPSC_MGPV_Rgl_Gll1p25deg_Lp850_Pwv_Emb_grib2.bin"
    return [f1, f2, f3, f4, f5, f6, f7]
