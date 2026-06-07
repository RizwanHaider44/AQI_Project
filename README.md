# 🔗 Live dashboard: https://aqiproject-9cwbnbcacgjdmndnegj7gf.streamlit.app/

# Karachi AQI Predictor

I built this for my Data Science internship at 10Pearls. It forecasts Karachi's Air Quality Index for the next three days, and it runs without any servers of my own: GitHub Actions runs the pipelines on a schedule, Hopsworks stores the features and the model, and the dashboard is hosted free on Streamlit.

The quickest way to see it working is the live dashboard linked above. The full write-up of how I built it, the results, and what I would improve next is in `AQI_Project_Report.pdf`.

## What's in here

| File | What it does |
|------|------|
| `app.py` | The Streamlit dashboard: pulls live data, runs the model, shows the 3-day forecast |
| `model.joblib` | The trained model the dashboard loads |
| `aqi_common.py` | Shared code for fetching data, computing the EPA AQI, and building features |
| `feature_pipeline.py` | Runs hourly to fetch new data and store it |
| `training_pipeline.py` | Runs daily to retrain and register the best model |
| `.github/workflows/` | The two schedules that run the pipelines |
| `requirements.txt` | The dashboard's Python packages |
| `AQI_Project_Report.pdf` | The full project report |

## A few honest notes

- The dashboard is self-contained, so it keeps working on its own even if the rest of the system is down.
- The hourly feature job runs fine. The daily training job is held up because Hopsworks was migrating its servers and one of its background jobs got stuck. I explain this in the report.
- GitHub's scheduled times run in UTC and can be a few minutes late, and they pause after about 60 days with no activity. Any new push wakes them.

## Running it yourself (optional)

To run the pipelines, add your own OPENWEATHER_API_KEY and HOPSWORKS_API_KEY as repository secrets under Settings, then Secrets and variables, then Actions, and trigger the workflows from the Actions tab.
