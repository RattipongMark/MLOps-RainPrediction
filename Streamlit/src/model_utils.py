import dagshub
import mlflow
import mlflow.xgboost
import pandas as pd
from mlflow import MlflowClient
from src.utils import fetch_open_meteo_hourly
import json
import os
import streamlit as st


def forecast_rain_from_model(df_forecast=None, repo_owner='RattipongMark', repo_name='MLOps-RainPrediction',
                             registered_model_name='rain_prediction_model', model_alias='Production'):
    """Input: hourly df or None (load from API), output: df with predictions"""

    token = os.environ.get("DAGSHUB_TOKEN")
    dagshub.auth.add_app_token(token)

    # Load model
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name=registered_model_name, alias=model_alias)
    run_id = mv.run_id

    logged_artifacts = [f.path for f in client.list_artifacts(run_id)]
    if any("xgb" in f.lower() for f in logged_artifacts):
        model = mlflow.xgboost.load_model(f"models:/{registered_model_name}/{mv.version}")
        is_xgb = True
    else:
        model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{mv.version}")
        is_xgb = False

    # Load API if df_forecast not given
    if df_forecast is None:
        df_forecast = fetch_open_meteo_hourly()

    # Feature mapping (เหมือนเดิม)
    api_to_train_map = {
        "soil_moisture_0_to_1cm": "soil_moisture_0_to_7cm",
        "soil_moisture_1_to_3cm": "soil_moisture_0_to_7cm",
        "soil_moisture_3_to_9cm": "soil_moisture_0_to_7cm",
        "soil_moisture_9_to_27cm": "soil_moisture_7_to_28cm",
        "soil_moisture_27_to_81cm": "soil_moisture_28_to_100cm",
        "soil_temperature_0cm": "soil_temperature_0_to_7cm",
        "soil_temperature_6cm": "soil_temperature_0_to_7cm",
        "soil_temperature_18cm": "soil_temperature_7_to_28cm",
        "soil_temperature_54cm": "soil_temperature_28_to_100cm",
        "wind_direction_120m": "wind_direction_100m",
        "wind_speed_120m": "wind_speed_100m"
    }

    df_mapped = {}
    grouped = {}
    for col in df_forecast.columns:
        if col in api_to_train_map:
            new_col = api_to_train_map[col]
            grouped.setdefault(new_col, []).append(df_forecast[col])
        else:
            grouped.setdefault(col, []).append(df_forecast[col])

    for new_col, series_list in grouped.items():
        if len(series_list) == 1:
            df_mapped[new_col] = series_list[0]
        else:
            df_mapped[new_col] = pd.concat(series_list, axis=1).mean(axis=1)

    df_model_input = pd.DataFrame(df_mapped)
    df_model_input = df_model_input.drop(columns=["time", "rain", "weather_code"], errors='ignore')


    local_path = os.path.join(os.path.dirname(__file__), "../../data/feature.json")
    with open(local_path, "r") as f:
        feature_columns = json.load(f)
    df_model_input = df_model_input[feature_columns]


    # Predict
    predictions = model.predict(df_model_input)
    pred_probs = model.predict_proba(df_model_input)[:, 1] if is_xgb else None
    df_forecast["predicted_rain"] = predictions
    df_forecast["predicted_rain_prob"] = pred_probs

    return df_forecast