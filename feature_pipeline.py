"""
feature_pipeline.py — runs hourly via GitHub Actions.
Fetches the last ~20 days of pollution data, rebuilds the daily features,
and upserts the recent days into the Hopsworks feature group so the store
always holds the latest AQI. (20-day window gives enough history for the
7-day lag/rolling features on the most recent day.)
"""
import datetime as dt
import aqi_common as ac

def run():
    end = int(dt.datetime.now(dt.timezone.utc).timestamp())
    start = end - 20 * 24 * 3600
    raw = ac.fetch_pollution(start, end)
    feats = ac.build_features(raw).dropna()      # drop early days with incomplete lags
    print(f"built {len(feats)} recent daily rows")

    _, fs = ac.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=ac.FG_NAME, version=ac.FG_VERSION,
        description="Daily AQI + engineered pollution features",
        primary_key=["date"], event_time="datetime", online_enabled=True,
    )
    # upsert: rows with an existing 'date' are overwritten, new days appended
    fg.insert(feats, write_options={"wait_for_job": False})
    print("feature group updated:", ac.FG_NAME)

if __name__ == "__main__":
    run()
