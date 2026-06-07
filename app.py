"""
Karachi AQI dashboard.

Pulls the last few weeks of pollution data from OpenWeather, rebuilds the same
features the model was trained on, and shows a 3-day forecast. The model is
loaded from the repo, so the dashboard runs on its own and does not need
Hopsworks to be up. Deployed on Streamlit Community Cloud, with the
OPENWEATHER_API_KEY set as a secret.
"""
import datetime as dt, time, os
import numpy as np, pandas as pd, requests, joblib
import streamlit as st
import plotly.graph_objects as go

LAT, LON, CITY = 24.86, 67.01, "Karachi"
POLL = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]

st.set_page_config(page_title=f"{CITY} AQI Forecast", page_icon="🌫️", layout="wide")

# OpenWeather key: try Streamlit secrets first, then an env var, then ask for it
OWM_KEY = ""
try:
    OWM_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
except Exception:
    pass
OWM_KEY = OWM_KEY or os.environ.get("OPENWEATHER_API_KEY", "")
if not OWM_KEY:
    OWM_KEY = st.text_input("Enter your OpenWeather API key to load data", type="password")
    if not OWM_KEY:
        st.info("Add OPENWEATHER_API_KEY in the app's Secrets, or paste it above.")
        st.stop()

# Turn pollutant concentrations into the US EPA AQI (2024 breakpoint tables).
# PM2.5 and PM10 are what drive the AQI in Karachi, so I use those.
PM25 = [(0,9.0,0,50),(9.1,35.4,51,100),(35.5,55.4,101,150),(55.5,125.4,151,200),
        (125.5,225.4,201,300),(225.5,325.4,301,400),(325.5,500.4,401,500)]
PM10 = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),
        (355,424,201,300),(425,504,301,400),(505,604,401,500)]
def _sub(C, bp):
    if pd.isna(C): return np.nan
    C = np.floor(C*10)/10
    for clo, chi, ilo, ihi in bp:
        if C <= chi:
            return round((ihi-ilo)/(chi-clo)*(max(C,clo)-clo)+ilo)
    return 500

# AQI bands: the label and colour each forecast card gets
CATEGORIES = [(50,"Good","#34a853"),(100,"Moderate","#f9ab00"),
              (150,"Unhealthy (Sensitive)","#ff6d00"),(200,"Unhealthy","#ea4335"),
              (300,"Very Unhealthy","#8e24aa"),(500,"Hazardous","#6d4c41")]
def category(aqi):
    for hi, label, color in CATEGORIES:
        if aqi <= hi: return label, color
    return "Hazardous", "#6d4c41"

# Grab about 25 days of hourly data, enough history for the 7-day lags. Cached
# for an hour so we are not hammering the API on every click.
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pollution(key):
    end = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start = end - 25*24*3600
    rows, s = [], start
    while s < end:
        e = min(s + 30*24*3600, end)
        url = ("http://api.openweathermap.org/data/2.5/air_pollution/history"
               f"?lat={LAT}&lon={LON}&start={s}&end={e}&appid={key}")
        r = requests.get(url, timeout=30); r.raise_for_status()
        for it in r.json().get("list", []):
            rec = {"dt": it["dt"]}; rec.update(it["components"]); rows.append(rec)
        s = e; time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
    df["datetime"] = pd.to_datetime(df["dt"], unit="s", utc=True)
    return df.set_index("datetime")

# Same feature engineering the training pipeline uses, so the model sees what it expects
def build_features(raw):
    df = raw.copy()
    df[POLL] = df[POLL].replace(-9999.0, np.nan).mask(df[POLL] < 0)
    df[POLL] = df[POLL].interpolate(limit=6).ffill().bfill()
    daily = df[POLL].resample("D").mean().dropna(how="all")
    daily["aqi"] = [max(_sub(r.pm2_5, PM25), _sub(r.pm10, PM10)) for r in daily.itertuples()]
    d = daily.copy()
    d["month"]=d.index.month.astype("int64"); d["doy"]=d.index.dayofyear.astype("int64")
    d["dow"]=d.index.dayofweek.astype("int64"); d["is_weekend"]=(d.dow>=5).astype("int64")
    d["month_sin"]=np.sin(2*np.pi*d.month/12); d["month_cos"]=np.cos(2*np.pi*d.month/12)
    d["doy_sin"]=np.sin(2*np.pi*d.doy/365);    d["doy_cos"]=np.cos(2*np.pi*d.doy/365)
    for col in ["aqi","pm2_5","pm10","o3","no2","co","so2"]:
        for lag in [1,2,3,7]: d[f"{col}_lag{lag}"]=d[col].shift(lag)
        d[f"{col}_rmean7"]=d[col].rolling(7).mean(); d[f"{col}_rstd7"]=d[col].rolling(7).std()
    d["aqi_change"]=d.aqi.diff()
    return daily, d

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")   # bundle: the model plus its feature list

# Fetch, build features, and predict the next three days
st.title(f"🌫️ {CITY} Air Quality, 3-Day Forecast")
st.caption("Live data from OpenWeather · US EPA AQI · forecasts the next 3 days")

try:
    raw = fetch_pollution(OWM_KEY)
    daily, feat = build_features(raw)
    bundle = load_model()
    X = feat[bundle["features"]].dropna().iloc[[-1]]
    preds = [round(float(p)) for p in bundle["model"].predict(X)[0]]
    last_date = feat.index[-1]
    today_aqi = int(daily["aqi"].iloc[-1])
except Exception as e:
    st.error(f"Could not load data or model: {e}")
    st.stop()

# today's AQI
label, color = category(today_aqi)
st.markdown(f"### Current AQI ({last_date.date()})")
st.markdown(
    f"<div style='background:{color};color:white;padding:18px;border-radius:12px;"
    f"font-size:28px;font-weight:700;width:260px'>{today_aqi} &nbsp;·&nbsp; {label}</div>",
    unsafe_allow_html=True)

# warn if the forecast looks bad
worst = max(preds)
if worst > 300:
    st.error(f"⚠️ HAZARDOUS air forecast (AQI up to {worst}). Avoid outdoor exposure.")
elif worst > 150:
    st.warning(f"⚠️ Unhealthy air forecast (AQI up to {worst}). Sensitive groups take care.")

# the three forecast cards
st.markdown("### 3-Day Forecast")
cols = st.columns(3)
for i, (c, p) in enumerate(zip(cols, preds)):
    d = (last_date + pd.Timedelta(days=i+1)).date()
    lab, col = category(p)
    c.markdown(
        f"<div style='background:{col};color:white;padding:16px;border-radius:12px;text-align:center'>"
        f"<div style='font-size:14px'>{d}</div>"
        f"<div style='font-size:34px;font-weight:700'>{p}</div>"
        f"<div style='font-size:13px'>{lab}</div></div>", unsafe_allow_html=True)

# recent history and the forecast on one chart
hist = daily["aqi"].tail(30)
fc_dates = [last_date + pd.Timedelta(days=i+1) for i in range(3)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="actual (last 30 days)",
                         line=dict(color="#1a237e")))
fig.add_trace(go.Scatter(x=[last_date]+fc_dates, y=[today_aqi]+preds, name="forecast",
                         line=dict(color="#ff6d00", dash="dash"), mode="lines+markers"))
fig.update_layout(height=380, yaxis_title="AQI", margin=dict(t=20),
                  legend=dict(orientation="h", y=1.1))
st.markdown("### Recent trend & forecast")
st.plotly_chart(fig, use_container_width=True)

# which features the model leans on most
try:
    est = bundle["model"].estimators_[0]
    imp = pd.Series(est.feature_importances_, index=bundle["features"]).sort_values(ascending=False).head(10)
    st.markdown("### What drives the forecast (top features)")
    st.bar_chart(imp)
except Exception:
    pass

st.caption(f"Model: {bundle['name']} · forecasts AQI over the next 3 days. Data updates hourly from OpenWeather.")
