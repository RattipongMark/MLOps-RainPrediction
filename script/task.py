import os
import pandas as pd
import numpy as np
import json
import requests
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dagshub
import mlflow
from mlflow import MlflowClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping
from datetime import datetime
from datetime import date

# -----------------------------
# Global: Data directory
# -----------------------------
def get_data_dir():
    """Return absolute path to the data folder and create it if missing"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

DATA_DIR = get_data_dir()  # ใช้ทุกฟังก์ชัน

OPEN_METEO_FORECAST_API = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=13.6513&longitude=100.4964&hourly="
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

OPEN_METEO_INTEVAL_API = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=13.6513&longitude=100.4964&"
    "start_date=2014-01-01&end_date=2025-01-01&"
    "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,rain,vapour_pressure_deficit,et0_fao_evapotranspiration,"
    "cloud_cover_high,cloud_cover_mid,cloud_cover_low,surface_pressure,"
    "pressure_msl,weather_code,wind_gusts_10m,wind_direction_10m,wind_direction_100m,"
    "wind_speed_100m,wind_speed_10m,soil_moisture_28_to_100cm,"
    "soil_moisture_7_to_28cm,soil_moisture_0_to_7cm,"
    "soil_temperature_100_to_255cm,soil_temperature_28_to_100cm,"
    "soil_temperature_0_to_7cm,soil_temperature_7_to_28cm"
)

TODAY = date.today().strftime("%Y%m%d")
TODAY_FILENAME = f"newdata_{TODAY}.csv"

# -----------------------------
# 1. Load data
# -----------------------------
def load_data_from_api(output_filename=TODAY_FILENAME, ref_filename="yesterday.csv", window_size=10000):
    output_path = os.path.join(DATA_DIR, output_filename)
    ref_path = os.path.join(DATA_DIR, ref_filename)
    reference_path = os.path.join(DATA_DIR, "raw_reference.csv")

    # ---------- NEW: Handle new day rollover ----------
    if os.path.exists(output_path):
        modified_date = datetime.fromtimestamp(os.path.getmtime(output_path)).date()
        print(f"[tasks] Existing {output_filename} modified date: {modified_date}")
        today = datetime.now().date()

        if modified_date < today:
            # yesterday.csv -> backup old today.csv
            os.rename(output_path, ref_filename)
            print("[tasks] Detected new day → moved old today.csv to yesterday.csv")
        else:
            print("[tasks] Today’s data already loaded. Skipping API call.")
            return

    # ---- backup yesterday if missing ----
    if not os.path.exists(ref_path) and os.path.exists(reference_path):
        df_ref = pd.read_csv(reference_path)
        df_ref.to_csv(ref_path, index=False)
        print(df_ref.head())
        print(f"[tasks] yesterday.csv missing, copied reference.csv -> yesterday.csv")

    # ---- load new data from API ----
    response = requests.get(OPEN_METEO_FORECAST_API)
    if response.status_code != 200:
        raise ValueError(f"API request failed: {response.status_code}")

    data = response.json()
    df_new = pd.DataFrame(data["hourly"])
    df_new["time"] = pd.to_datetime(df_new["time"])
    for col in df_new.columns:
        if col != "time":
            df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    df_new = df_new.fillna(0)

    print(f"[tasks] Loaded new data from API, rows: {len(df_new)}")
    print(df_new.head())

    # ---- save raw current data temporarily ----
    df_new.to_csv(output_path, index=False)
    print(f"[tasks] Raw new data saved to {output_path}")


# -----------------------------
# 2. Preprocess and combine
# -----------------------------
def preprocess_data(
    window_size=10000, 
    input_filename=TODAY_FILENAME, 
    ref_filename="yesterday.csv", 
    output_filename="data_preprocessed.csv"):

    # ---------------- Paths ----------------
    today_path = os.path.join(DATA_DIR, input_filename)
    ref_path = os.path.join(DATA_DIR, ref_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    # ---------------- Mapping API → Train ----------------
    api_to_train_map = {
        # Soil Moisture groups
        "soil_moisture_0_to_1cm": "soil_moisture_0_to_7cm",
        "soil_moisture_1_to_3cm": "soil_moisture_0_to_7cm",
        "soil_moisture_3_to_9cm": "soil_moisture_0_to_7cm",

        "soil_moisture_9_to_27cm": "soil_moisture_7_to_28cm",
        "soil_moisture_27_to_81cm": "soil_moisture_28_to_100cm",

        # Soil Temperature groups
        "soil_temperature_0cm":    "soil_temperature_0_to_7cm",
        "soil_temperature_6cm":    "soil_temperature_0_to_7cm",

        "soil_temperature_18cm":   "soil_temperature_7_to_28cm",
        "soil_temperature_54cm":   "soil_temperature_28_to_100cm",

        "wind_direction_120m": "wind_direction_100m",
        "wind_speed_120m": "wind_speed_100m"
    }

    # ---------------- Load today's data ----------------
    df_today = pd.read_csv(today_path)

    # Target
    df_today["target"] = (df_today["rain"] > 0.1).astype(int)
    df_today = df_today.drop(["rain", "weather_code"], axis=1)  # keep time for concat

    # ---------------- Apply Mapping / Column Merge to today's data ----------------
    df_today_mapped = {}
    grouped = {}
    for col in df_today.columns:
        if col in api_to_train_map:
            new_col = api_to_train_map[col]
            grouped.setdefault(new_col, []).append(df_today[col])
        else:
            grouped.setdefault(col, []).append(df_today[col])

    for new_col, series_list in grouped.items():
        if len(series_list) == 1:
            df_today_mapped[new_col] = series_list[0]
        else:
            df_today_mapped[new_col] = pd.concat(series_list, axis=1).mean(axis=1)

    df_today_final = pd.DataFrame(df_today_mapped)

    # drop time if exists
    if "time" in df_today_final.columns:
        df_today_final = df_today_final.drop(columns=["time"])

    # ---------------- Load reference ----------------
    if os.path.exists(ref_path):
        df_ref = pd.read_csv(ref_path)

        # concat & keep only last N rows
        df_combined = pd.concat([df_ref, df_today_final], ignore_index=True).tail(window_size)

        print(f"[tasks] Combined data from {ref_filename} and {input_filename}, total rows: {len(df_combined)}")
    else:
        df_combined = df_today_final

    df_combined = df_combined.drop(columns=["Unnamed: 0"], errors='ignore')
    print(f"[tasks] Preprocessing complete, final columns: {df_combined.columns.tolist()}")
    print(df_combined.head())
    print(df_combined.info())

    # ---------------- Save ----------------
    df_combined.to_csv(output_path, index=False)
    print(f"[tasks] Preprocessed & combined data saved to {output_path}")

    return df_combined



# -----------------------------
# 3. Feature selection
# -----------------------------
def feature_selection(input_filename="data_preprocessed.csv",
                      features_filename="selected_features.json",
                      importance_filename="feature_importance.csv",
                      output_filename="current.csv",
                      raw_reference_filename="raw_reference.csv",
                      reference_output_filename="reference.csv",
                      corr_threshold=0.7,
                      importance_cutoff=0.90):

    input_path = os.path.join(DATA_DIR, input_filename)
    features_path = os.path.join(DATA_DIR, features_filename)
    importance_path = os.path.join(DATA_DIR, importance_filename)
    output_path = os.path.join(DATA_DIR, output_filename)
    raw_reference_path = os.path.join(DATA_DIR, raw_reference_filename)
    reference_path = os.path.join(DATA_DIR, reference_output_filename)

    print("[tasks] FEATURE SELECTION START")
    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # -------------------------
    # Sanitize X
    # -------------------------
    X = X.apply(pd.to_numeric, errors='coerce')  # convert all to numeric
    X = X.fillna(0)  # fill missing values
    if X.isna().any().any():
        raise ValueError("X still contains NaN after fillna")

    # -------------------------
    # Sanitize y
    # -------------------------
    y = y.astype(int)  # ensure 0/1
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(f"y has less than 2 classes: {unique_classes}, cannot train XGBClassifier")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X, y)

    # Save raw feature importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    # Importance-guided correlation pruning
    selected_features = []
    for feature in importance_df["Feature"]:
        if all(abs(X[feature].corr(X[sel])) <= corr_threshold for sel in selected_features):
            selected_features.append(feature)

    # Keep only features that cover the cumulative importance threshold
    importance_selected = importance_df[importance_df["Feature"].isin(selected_features)].copy()
    importance_selected["Cumulative"] = importance_selected["Importance"].cumsum() / importance_selected["Importance"].sum()
    cutoff_index = np.argmax(importance_selected["Cumulative"] >= importance_cutoff) + 1
    final_selected_features = importance_selected["Feature"].iloc[:cutoff_index].tolist()

    # Save final selected features
    with open(features_path, "w") as f:
        json.dump(final_selected_features, f)

    # Save reduced dataset
    df_selected = df[final_selected_features + ["target"]]
    df_selected.to_csv(output_path, index=False)
    print(f"[tasks] Selected features is {final_selected_features}")
    print(df_selected.head(10))

    df_raw_ref = pd.read_csv(raw_reference_path)
    df_raw_ref_selected = df_raw_ref[final_selected_features + ["target"]]
    df_raw_ref_selected.to_csv(reference_path, index=False)
    print(df_raw_ref_selected.head(10))

    def validate_features(current_df, reference_df):
        """
        Validate that current and reference datasets have the same columns
        and no missing values.
        
        Args:
            current_df (pd.DataFrame): DataFrame for current dataset
            reference_df (pd.DataFrame): DataFrame for reference dataset
        
        Raises:
            ValueError: If columns mismatch or there are NaN values
        """
        # 1. Check columns
        if list(current_df.columns) != list(reference_df.columns):
            raise ValueError(
                f"Column mismatch:\nCurrent: {current_df.columns.tolist()}\nReference: {reference_df.columns.tolist()}"
            )
        
        # 2. Check for NaN
        if current_df.isna().any().any():
            raise ValueError("Current dataset contains NaN values")
        if reference_df.isna().any().any():
            raise ValueError("Reference dataset contains NaN values")
        
        print("[validate_features] Columns match and no missing values found ✅")
    validate_features(df_selected, df_raw_ref_selected)

    print(df_selected.info())
    print(df_raw_ref_selected.info())

    print("[tasks] FEATURE SELECTION COMPLETE")


# -----------------------------
# 4. Train models
# -----------------------------
def train_models(input_filename="reference.csv",
                 experiment_name="rain_model_comparison",
                 model_dir=os.path.join(DATA_DIR, "models")):

    input_path = os.path.join(DATA_DIR, input_filename)
    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(model_dir, exist_ok=True)

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

# -----------------------------
# 5. save best model
# -----------------------------
def save_best_model(**context):
    ti = context["ti"]
    results = ti.xcom_pull(task_ids="train_models")

    best_model = max(results, key=lambda x: x["acc"])
    print(f"Best model: {best_model['model_name']} with accuracy {best_model['acc']:.4f}")

    best_model_uri = f"runs:/{best_model['run_id']}/{best_model['artifact_path']}"
    registered_model_name = "rain_prediction_model"

    model_version = mlflow.register_model(
        model_uri=best_model_uri,
        name=registered_model_name
    )

    print(f"Registered model version: name={registered_model_name}, version={model_version.version}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model '{registered_model_name}' version {model_version.version} is now in stage 'Production'.")

# -----------------------------
# 6. Generate Evidently report
# -----------------------------
def generate_evidently(current_filename="current.csv",
                       reference_filename="reference.csv",
                       registered_model_name="rain_prediction_model",
                       output_filename="evidently.html"):

    current_path = os.path.join(DATA_DIR, current_filename)
    reference_path = os.path.join(DATA_DIR, reference_filename)
    output_path = os.path.join(DATA_DIR, output_filename)

    df_cur = pd.read_csv(current_path)

    # ---- check first run ----
    if not os.path.exists(reference_path):
        print("[INFO] No reference dataset yet. First run.")
        df_cur.to_csv(reference_path, index=False)
        return {"first_run": True}

    df_ref = pd.read_csv(reference_path)

    # ---- load production model ----
    try:
        model_uri = f"models:/{registered_model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print("[INFO] No Production model yet. First run.", str(e))
        df_cur.to_csv(reference_path, index=False)
        return {"first_run": True}

    # ---- add predictions ----
    features_path = os.path.join(DATA_DIR, "selected_features.json")
    with open(features_path) as f:
        selected_features = json.load(f)

    # เลือกเฉพาะ features ที่โมเดลถูก train
    df_ref_model = df_ref[selected_features]
    df_cur_model = df_cur[selected_features]

    df_ref["prediction"] = model.predict(df_ref_model)
    df_cur["prediction"] = model.predict(df_cur_model)
    # df_ref["prediction"] = model.predict(df_ref.drop("target", axis=1))
    # df_cur["prediction"] = model.predict(df_cur.drop("target", axis=1))

    # ---- run Evidently ----
    column_mapping = ColumnMapping(target="target", prediction="prediction")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), ClassificationPreset()])
    report.run(reference_data=df_ref, current_data=df_cur, column_mapping=column_mapping)
    report.save_html(output_path)
    print(f"Evidently report saved to {output_path}")

    # ---- extract metrics ----
    result = report.as_dict()
    metrics = result["metrics"]

    def get_metric_result(metrics, metric_name):
        for m in metrics:
            if m.get("metric") == metric_name:
                return m.get("result")
        return None

    data_drift_result   = get_metric_result(metrics, "DatasetDriftMetric")
    target_drift_result = get_metric_result(metrics, "ColumnDriftMetric")
    cls_result          = get_metric_result(metrics, "ClassificationQualityMetric")

    dataset_drift   = data_drift_result["dataset_drift"]
    share_drifted   = data_drift_result["share_of_drifted_columns"]
    target_drift    = target_drift_result["drift_detected"]
    ref_acc         = cls_result["reference"]["accuracy"]
    cur_acc         = cls_result["current"]["accuracy"]
    accuracy_drop   = ref_acc - cur_acc

    print(f"Dataset drift: {dataset_drift}, share drifted: {share_drifted:.2%}")
    print(f"Target drift: {target_drift}")
    print(f"Reference acc: {ref_acc:.4f}, Current acc: {cur_acc:.4f}, Drop: {accuracy_drop:.4f}")

    # ---- log metrics to MLflow ----
    with mlflow.start_run(run_name=f"evidently_{datetime.now().date()}"):
        mlflow.log_artifact(output_path, artifact_path="evidently_reports")
        mlflow.log_metric("share_drifted_columns", share_drifted)
        mlflow.log_metric("accuracy_drop", accuracy_drop)
        mlflow.log_metric("dataset_drift_flag", int(dataset_drift))
        mlflow.log_metric("target_drift_flag", int(target_drift))

    return {
        "dataset_drift": dataset_drift,
        "share_drifted": share_drifted,
        "target_drift": target_drift,
        "ref_acc": ref_acc,
        "cur_acc": cur_acc,
        "accuracy_drop": accuracy_drop
    }


# -----------------------------
# 7. Decide retrain
# -----------------------------
def decide_retrain(current_filename="current.csv",
                   reference_filename="reference.csv",
                   **context):
    ti = context["ti"]
    ev_result = ti.xcom_pull(task_ids="generate_evidently")

    if ev_result.get("first_run", False):
        print("First run detected, proceeding to train model.")

    dataset_drift = ev_result["dataset_drift"]
    share_drifted = ev_result["share_drifted"]
    target_drift  = ev_result["target_drift"]
    ref_acc       = ev_result["ref_acc"]
    cur_acc       = ev_result["cur_acc"]
    accuracy_drop = ev_result["accuracy_drop"]

    acc_drop_threshold = 0.05       # 5% absolute accuracy drop
    drift_share_threshold = 0.3     # >= 30% of features drifted

    should_retrain = (
        (dataset_drift and share_drifted >= drift_share_threshold)
        or target_drift
        or (accuracy_drop >= acc_drop_threshold)
    )

    dag_run = context.get("dag_run")
    dag_conf = dag_run.conf if dag_run else {}
    force_retrain = bool(dag_conf.get("force_retrain", False))

    execution_date = context.get("logical_date") or context.get("execution_date")
    if execution_date is not None:
        is_monday = execution_date.weekday() == 0  # Monday = 0
    else:
        is_monday = datetime.utcnow().weekday() == 0

    if is_monday or force_retrain:
        should_retrain = True

    print(f"should_retrain_from_metrics={should_retrain}")

    if should_retrain:
        current_path = os.path.join(DATA_DIR, current_filename)
        reference_path = os.path.join(DATA_DIR, reference_filename)
        if os.path.exists(current_path):
            os.replace(current_path, reference_path)
            print(f"[tasks] Retrain triggered: {current_filename} -> {reference_filename}")

    return should_retrain
