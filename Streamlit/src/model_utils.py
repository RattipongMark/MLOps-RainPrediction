import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load model (local pickle)
# -----------------------------
def load_model():
    with open("src/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ฟีเจอร์ที่โมเดลต้องการ
FEATURES = [
    "relative_humidity_2m",
    "cloud_cover_low",
    "dew_point_2m",
    "temperature_2m",
    "wind_speed_10m"
]

LABEL_MAP = {
    0: "Not Rain",
    1: "Rain"
}


# --------------------------------------------------------
# DAILY PREDICTION (ใช้สำหรับ Dashboard 7 วัน)
# --------------------------------------------------------
def classify_rain_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict rain for daily aggregated df (7 rows)
    """
    X = df[FEATURES].values

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # probability of chosen class
    proba_pred = [y_proba[i, c] for i, c in enumerate(y_pred)]

    result_df = pd.DataFrame({
        "date": df["date"],
        "rain_class": y_pred,
        "rain_label": [LABEL_MAP[c] for c in y_pred],
        "rain_proba": proba_pred
    })

    return result_df


# --------------------------------------------------------
# HOURLY PREDICTION (ใช้ในหน้า Prediction)
# --------------------------------------------------------
def classify_rain_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict rain per hour for the next 7 days (168 rows)
    """
    X = df[FEATURES].values

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # prob of chosen class
    proba_pred = [y_proba[i, c] for i, c in enumerate(y_pred)]

    result_df = pd.DataFrame({
        "time": df["time"],
        "date": df["date"],
        "hour": df["hour"],
        "rain_class": y_pred,
        "rain_label": [LABEL_MAP[c] for c in y_pred],
        "rain_proba": proba_pred,
    })

    return result_df
