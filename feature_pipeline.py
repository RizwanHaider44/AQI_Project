"""
Runs every hour on GitHub Actions. It grabs the last few weeks of pollution
data, rebuilds the daily features, and pushes the recent days into the Hopsworks
feature group so the store always has the latest AQI. I fetch about 20 days so
the 7-day lag and rolling features on the newest day have enough history.
"""
import datetime as dt
import aqi_common as ac

def run():
    end = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start = end - 20 * 24 * 3600
    raw = ac.fetch_pollution(start, end)
    feats = ac.build_features(raw).dropna()      # the first few days have incomplete lags, so drop them
    print(f"built {len(feats)} recent daily rows")

    _, fs = ac.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=ac.FG_NAME, version=ac.FG_VERSION,
        description="Daily AQI + engineered pollution features",
        primary_key=["date"], event_time="datetime", online_enabled=True,
    )
    # 'date' is the key, so a day that already exists is overwritten and new days are added
    fg.insert(feats, write_options={"wait_for_job": False})
    print("feature group updated:", ac.FG_NAME)

if __name__ == "__main__":
    run()
