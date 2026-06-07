"""
Shared helpers used by both pipelines: fetching data, computing the AQI, and
building features. I keep them in one file so the features written each hour
are exactly the same as the ones the model was trained on. API keys come from
environment variables, which are set as GitHub Secrets.
"""
import os, time, datetime as dt
import numpy as np, pandas as pd
import requests, hopsworks

# settings
LAT, LON, CITY = 24.86, 67.01, "Karachi"
PROJECT = "Internship_project"
FG_NAME, FG_VERSION = "aqi_features_daily", 1
POLL = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
OWM_KEY = os.environ["OPENWEATHER_API_KEY"]
HOPSWORKS_KEY = os.environ["HOPSWORKS_API_KEY"]

# connect to my Hopsworks project
def get_feature_store():
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY, project=PROJECT)
    return project, project.get_feature_store()

# pull raw hourly pollution data from OpenWeather, a chunk of dates at a time
def fetch_pollution(start_utc, end_utc):
    """Get hourly pollution data from OpenWeather between two UTC epoch seconds."""
    rows, window, s = [], 30 * 24 * 3600, start_utc
    while s < end_utc:
        e = min(s + window, end_utc)
        url = ("http://api.openweathermap.org/data/2.5/air_pollution/history"
               f"?lat={LAT}&lon={LON}&start={s}&end={e}&appid={OWM_KEY}")
        r = requests.get(url, timeout=30); r.raise_for_status()
        for it in r.json().get("list", []):
            rec = {"dt": it["dt"]}; rec.update(it["components"])
            rows.append(rec)
        s = e; time.sleep(1)
    df = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
    df["datetime"] = pd.to_datetime(df["dt"], unit="s", utc=True)
    return df.set_index("datetime")

# US EPA AQI from pollutant concentrations (2024 breakpoint tables).
# In Karachi the AQI is driven by particulates, so PM2.5 and PM10 set it.
PM25 = [(0,9.0,0,50),(9.1,35.4,51,100),(35.5,55.4,101,150),(55.5,125.4,151,200),
        (125.5,225.4,201,300),(225.5,325.4,301,400),(325.5,500.4,401,500)]
PM10 = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),
        (355,424,201,300),(425,504,301,400),(505,604,401,500)]
def _sub_index(C, bp):
    if pd.isna(C): return np.nan
    C = np.floor(C * 10) / 10
    for clo, chi, ilo, ihi in bp:
        if C <= chi:
            C = max(C, clo)
            return round((ihi - ilo) / (chi - clo) * (C - clo) + ilo)
    return 500

# Build the daily features. This has to line up exactly with what the model
# was trained on, otherwise predictions go sideways.
def build_features(raw_hourly):
    df = raw_hourly.copy()
    df[POLL] = df[POLL].replace(-9999.0, np.nan)
    df[POLL] = df[POLL].mask(df[POLL] < 0)
    df[POLL] = df[POLL].interpolate(limit=6).ffill().bfill()
    daily = df[POLL].resample("D").mean().dropna(how="all")
    daily["aqi"] = [max(_sub_index(r.pm2_5, PM25), _sub_index(r.pm10, PM10))
                    for r in daily.itertuples()]
    d = daily.copy()
    d["month"] = d.index.month.astype("int64"); d["doy"] = d.index.dayofyear.astype("int64"); d["dow"] = d.index.dayofweek.astype("int64")
    d["is_weekend"] = (d.dow >= 5).astype("int64")
    d["month_sin"] = np.sin(2*np.pi*d.month/12); d["month_cos"] = np.cos(2*np.pi*d.month/12)
    d["doy_sin"] = np.sin(2*np.pi*d.doy/365);   d["doy_cos"] = np.cos(2*np.pi*d.doy/365)
    for col in ["aqi", "pm2_5", "pm10", "o3", "no2", "co", "so2"]:
        for lag in [1, 2, 3, 7]:
            d[f"{col}_lag{lag}"] = d[col].shift(lag)
        d[f"{col}_rmean7"] = d[col].rolling(7).mean()
        d[f"{col}_rstd7"] = d[col].rolling(7).std()
    d["aqi_change"] = d.aqi.diff()
    d = d.reset_index().rename(columns={"datetime": "datetime"})
    d["date"] = d["datetime"].dt.strftime("%Y-%m-%d")
    d["city"] = CITY
    return d
