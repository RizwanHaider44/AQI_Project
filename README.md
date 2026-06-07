# AQI Predictor — Serverless ML Pipeline

Forecasts the Air Quality Index for Karachi 1–3 days ahead. Fully serverless:
GitHub Actions runs the pipelines, Hopsworks stores the features and models.

## What's here
| File | Role |
|------|------|
| `aqi_common.py` | Shared config, data fetch, EPA-AQI calc, feature engineering |
| `feature_pipeline.py` | Hourly: fetch recent data → features → Hopsworks Feature Store |
| `training_pipeline.py` | Daily: read features → train/compare models → Model Registry |
| `requirements.txt` | Dependencies (pinned to Hopsworks 4.7) |
| `.github/workflows/feature_pipeline.yml` | Hourly schedule |
| `.github/workflows/training_pipeline.yml` | Daily schedule |

## One-time setup
1. Create a repo and upload every file here (keep the `.github/workflows/` folder structure).
2. Add two repository secrets under **Settings → Secrets and variables → Actions**:
   - `OPENWEATHER_API_KEY`
   - `HOPSWORKS_API_KEY`
3. Open the **Actions** tab, enable workflows, and click **Run workflow** on each to test.

## Notes
- GitHub cron times are UTC and can run a few minutes late.
- Scheduled workflows pause after 60 days of repo inactivity — any push re-enables them.
- The feature pipeline uses a 20-day fetch window so the 7-day lag/rolling features
  on the latest day always have enough history.
