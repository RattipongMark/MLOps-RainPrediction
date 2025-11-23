import requests
import pandas as pd
from datetime import datetime

OPEN_METEO_API = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=13.6513&longitude=100.4964&hourly="
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,precipitation_probability,precipitation,"
    "rain,vapour_pressure_deficit,et0_fao_evapotranspiration,visibility,"
    "evapotranspiration,cloud_cover_mid,cloud_cover_high,cloud_cover_low,"
    "cloud_cover,surface_pressure,pressure_msl,weather_code,"
    "wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,"
    "wind_direction_10m,wind_direction_80m,wind_direction_120m,"
    "wind_direction_180m,wind_gusts_10m,temperature_80m,temperature_120m,"
    "temperature_180m,soil_moisture_27_to_81cm,soil_moisture_9_to_27cm,"
    "soil_moisture_3_to_9cm,soil_moisture_1_to_3cm,soil_moisture_0_to_1cm,"
    "soil_temperature_18cm,soil_temperature_0cm,soil_temperature_6cm,"
    "soil_temperature_54cm"
    "&timezone=Asia/Bangkok"
)

# -----------------------------------------
# DAILY (ใช้ใน Dashboard)
# -----------------------------------------
def fetch_open_meteo():
    """Return daily summary 7 days"""
    res = requests.get(OPEN_METEO_API).json()

    hourly = res["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    # แปลงเป็น daily
    df["date"] = df["time"].dt.date

    daily = df.groupby("date").agg({
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "dew_point_2m": "mean",
        "cloud_cover_low": "mean",
        "precipitation_probability": "mean",
        "rain": "sum",
        "wind_speed_10m": "mean"
    }).reset_index()

    return daily.head(7)  # 7 days


# -----------------------------------------
# HOURLY (ใช้ในหน้า Prediction)
# -----------------------------------------
def fetch_open_meteo_hourly():
    """Return hourly data (168 rows = 7 days × 24 hours)."""
    res = requests.get(OPEN_METEO_API).json()

    hourly = res["hourly"]
    df = pd.DataFrame(hourly)

    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour

    return df
