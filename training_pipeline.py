"""
training_pipeline.py — runs daily via GitHub Actions.
Reads all features from the Hopsworks feature group, rebuilds the 3-day-ahead
targets, trains and compares models, and registers the best one as a new
version in the Model Registry.
"""
import os, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import aqi_common as ac

NON_FEATURES = ["date", "city", "datetime", "y1", "y2", "y3"]

def run():
    project, fs = ac.get_feature_store()

    # 1) read features
    fg = fs.get_feature_group(name=ac.FG_NAME, version=ac.FG_VERSION)
    df = fg.read().sort_values("datetime").set_index("datetime")
    print(f"read {len(df)} rows from feature store")

    # 2) targets: AQI 1/2/3 days ahead
    df["y1"] = df.aqi.shift(-1); df["y2"] = df.aqi.shift(-2); df["y3"] = df.aqi.shift(-3)
    df = df.dropna()
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    X, Y = df[feat_cols], df[["y1", "y2", "y3"]]

    # 3) time-based split
    n_test = min(120, max(20, len(df)//5))
    X_tr, X_te = X.iloc[:-n_test], X.iloc[-n_test:]
    Y_tr, Y_te = Y.iloc[:-n_test], Y.iloc[-n_test:]

    def mean_rmse(pred):
        return np.mean([mean_squared_error(Y_te.iloc[:, i], pred[:, i])**0.5 for i in range(3)])

    models = {
        "Ridge": make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0))),
        "RandomForest": MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
        "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(n_estimators=400, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0)),
    }
    scored = {}
    for name, m in models.items():
        m.fit(X_tr, Y_tr); scored[name] = (mean_rmse(m.predict(X_te)), m)
    best_name = min(scored, key=lambda n: scored[n][0])
    best_rmse, best_model = scored[best_name]
    print(f"best: {best_name}  mean RMSE {best_rmse:.2f}")

    # per-horizon metrics for the registry
    pred = best_model.predict(X_te)
    metrics = {f"rmse_plus{i+1}d": float(mean_squared_error(Y_te.iloc[:, i], pred[:, i])**0.5) for i in range(3)}
    metrics.update({f"r2_plus{i+1}d": float(r2_score(Y_te.iloc[:, i], pred[:, i])) for i in range(3)})

    # 4) register new model version
    bundle = {"model": best_model, "features": feat_cols, "name": best_name}
    os.makedirs("aqi_model", exist_ok=True)
    joblib.dump(bundle, "aqi_model/model.joblib")
    mr = project.get_model_registry()
    model = mr.python.create_model(name="aqi_forecaster", metrics=metrics,
            description=f"{best_name}: daily-retrained AQI +1/+2/+3 day forecaster for {ac.CITY}")
    model.save("aqi_model")
    print("registered aqi_forecaster v" + str(model.version))

if __name__ == "__main__":
    run()
