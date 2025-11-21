# tasks.py
import pandas as pd
import numpy as np
import json
import os
import requests
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
from mlflow import MlflowClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping
from datetime import datetime

# -----------------------------
# 1. Load data
# -----------------------------
def load_data_from_api(output_path="/opt/airflow/data/data_clean.csv"):
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=13.6513&longitude=100.4964&"
        "start_date=2014-01-01&end_date=2025-01-01&"
        "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        "apparent_temperature,rain,vapour_pressure_deficit,et0_fao_evapotranspiration,"
        "cloud_cover_high,cloud_cover_mid,cloud_cover_low,surface_pressure,"
        "pressure_msl,weather_code,wind_gusts_10m,wind_direction_10m,wind_direction_100m,"
        "wind_speed_100m,wind_speed_10m,soil_moisture_100_to_255cm,soil_moisture_28_to_100cm,"
        "soil_moisture_7_to_28cm,soil_moisture_0_to_7cm,"
        "soil_temperature_100_to_255cm,soil_temperature_28_to_100cm,"
        "soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,is_day"
    )

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"API request failed: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= "2014-01-01") & (df["time"] < "2025-01-01")]
    df.to_csv(output_path, index=False)
    print(f"[tasks] API data saved to {output_path}")

# -----------------------------
# 2. Preprocess
# -----------------------------
def preprocess_data(input_path="/opt/airflow/data/data_clean.csv", output_path="data_preprocessed.csv"):
    df = pd.read_csv(input_path)
    df["target"] = (df["rain (mm)"] > 0.1).astype(int)
    df = df.drop(["time", "rain (mm)", "weather_code (wmo code)", "is_day ()"], axis=1)
    df.to_csv(output_path, index=False)
    print(f"[tasks] Preprocessed data saved to {output_path}")

# -----------------------------
# 3. Feature selection
# -----------------------------
def feature_selection(input_path="data_preprocessed.csv",
                      features_path="selected_features.json",
                      importance_path="feature_importance.csv",
                      output_path="data_selected.csv",
                      corr_threshold=0.7,
                      importance_cutoff=0.90):

    print("[tasks] FEATURE SELECTION START")
    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=1.0,
                          random_state=42, eval_metric="logloss")
    model.fit(X, y)

    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    importance_df.to_csv(importance_path, index=False)

    corr = X.corr().abs()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_threshold)]
    X_uncorr = X.drop(columns=to_drop_corr)

    importance_df_uncorr = importance_df[importance_df["Feature"].isin(X_uncorr.columns)]
    importance_df_uncorr["Cumulative"] = importance_df_uncorr["Importance"].cumsum() / importance_df_uncorr["Importance"].sum()
    cutoff_index = np.argmax(importance_df_uncorr["Cumulative"] >= importance_cutoff) + 1
    final_selected_features = importance_df_uncorr["Feature"].iloc[:cutoff_index].tolist()

    with open(features_path, "w") as f:
        json.dump(final_selected_features, f)

    df_selected = df[final_selected_features + ["target"]]
    df_selected.to_csv(output_path, index=False)
    print("[tasks] FEATURE SELECTION COMPLETE")

# -----------------------------
# 4. Train models
# -----------------------------
def train_models(input_path="data_selected.csv",
                 tracking_uri="http://127.0.0.1:5000",
                 experiment_name="rain_model_comparison",
                 model_dir="models"):

    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    scale_pos_weight = (y_train==0).sum()/(y_train==1).sum()

    models = {
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=1.0, scale_pos_weight=scale_pos_weight,
                                 random_state=42, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                       subsample=0.8, colsample_bytree=1.0, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                               class_weight="balanced", random_state=42)
    }

    results = []
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", test_acc)

            artifact_path = f"model_{name}"
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path)
            else:
                mlflow.sklearn.log_model(model, artifact_path)

            results.append({"model_name": name, "acc": float(test_acc), "artifact_path": artifact_path,
                            "run_id": mlflow.active_run().info.run_id})

            print(f"[tasks] {name} trained & logged.")

    return results
