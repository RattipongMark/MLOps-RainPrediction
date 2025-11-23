import dagshub
import mlflow
import mlflow.xgboost
import pandas as pd
import requests
from mlflow import MlflowClient

def forecast_rain_from_model(
    repo_owner='RattipongMark',
    repo_name='MLOps-RainPrediction',
    registered_model_name='rain_prediction_model',
    model_alias='Production',
    latitude=13.6513,
    longitude=100.4964
):
    # -----------------------------
    # 1. Init Dagshub + MLflow
    # -----------------------------
    repo = dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    client = MlflowClient()
    
    mv = client.get_model_version_by_alias(name=registered_model_name, alias=model_alias)
    run_id = mv.run_id
    print(f"Loaded model version: {mv.version}, stage: {mv.current_stage}")

    # -----------------------------
    # 2. Load model
    # -----------------------------
    logged_artifacts = [f.path for f in client.list_artifacts(run_id)]
    if any("xgb" in f.lower() for f in logged_artifacts):
        model = mlflow.xgboost.load_model(f"models:/{registered_model_name}/{mv.version}")
        is_xgb = True
    else:
        model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/{mv.version}")
        is_xgb = False

    # -----------------------------
    # 3. Load forecast data from API
    # -----------------------------
    OPEN_METEO_FORECAST_API = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly="
        "temperature_2m,relative_humidity_2m,dew_point_2m,"
        "apparent_temperature,rain,vapour_pressure_deficit,et0_fao_evapotranspiration,"
        "cloud_cover_high,cloud_cover_mid,cloud_cover_low,surface_pressure,"
        "pressure_msl,weather_code,wind_gusts_10m,wind_direction_10m,wind_direction_120m,"
        "wind_speed_120m,wind_speed_10m,soil_moisture_27_to_81cm,"
        "soil_moisture_9_to_27cm,"
        "soil_moisture_3_to_9cm,soil_moisture_1_to_3cm,soil_moisture_0_to_1cm,"
        "soil_temperature_18cm,soil_temperature_0cm,soil_temperature_6cm,"
        "soil_temperature_54cm"
    )

    response = requests.get(OPEN_METEO_FORECAST_API)
    response.raise_for_status()
    data = response.json()
    df_forecast = pd.DataFrame(data["hourly"])
    df_forecast["time"] = pd.to_datetime(df_forecast["time"])

    # -----------------------------
    # 4. Map features
    # -----------------------------
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

    # -----------------------------
    # 5. Select features from training
    # -----------------------------
    local_path = client.download_artifacts(run_id, "data/reference_data.csv")
    df_train = pd.read_csv(local_path)
    feature_columns = [c for c in df_train.columns if c != "target"]
    df_model_input = df_model_input[feature_columns]

    # -----------------------------
    # 6. Predict
    # -----------------------------
    predictions = model.predict(df_model_input)
    pred_probs = model.predict_proba(df_model_input)[:, 1] if is_xgb else None

    df_forecast["predicted_rain"] = predictions
    df_forecast["predicted_rain_prob"] = pred_probs

    return df_forecast

# -----------------------------
# Example usage
# -----------------------------
df_result = forecast_rain_from_model()
print(df_result[["time", "predicted_rain", "predicted_rain_prob"]].head(50))
