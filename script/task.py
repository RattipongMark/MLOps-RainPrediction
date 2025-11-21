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
def load_data_from_api(output_path="../data/data_clean.csv"):
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
def preprocess_data(input_path="../data/data_clean.csv", output_path="../data/data_preprocessed.csv"):
    df = pd.read_csv(input_path)
    df["target"] = (df["rain (mm)"] > 0.1).astype(int)
    df = df.drop(["time", "rain (mm)", "weather_code (wmo code)", "is_day ()"], axis=1)
    df.to_csv(output_path, index=False)
    print(f"[tasks] Preprocessed data saved to {output_path}")

# -----------------------------
# 3. Feature selection
# -----------------------------
def feature_selection(input_path="../data/data_preprocessed.csv",
                      features_path="../data/selected_features.json",
                      importance_path="../data/feature_importance.csv",
                      output_path="../data/data_selected.csv",
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
def train_models(input_path="../data/data_selected.csv",
                #  tracking_uri="http://127.0.0.1:5000",
                 experiment_name="rain_model_comparison",
                 model_dir="models"):

    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(model_dir, exist_ok=True)

    # mlflow.set_tracking_uri(tracking_uri)
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

    # 2) Promote this version to Production (and optionally archive old ones)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,   # this archives any previous Production versions
    )

    print(f"Model '{registered_model_name}' version {model_version.version} is now in stage 'Production'.")


def generate_evidently(input_path="../data/data_selected.csv",
                       registered_model_name="rain_prediction_model",
                       model_dir="models",
                       output_path="../data/evidently.html"):
    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]


    # Load MLflow model
    try:
        model_uri = f"models:/{registered_model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        # Case 1: model name not found
        if "Model not found" in str(e):
            print("[INFO] No model registered yet. First run.")
            return {"first_run": True}

        # Case 2: model exists but no version in Production
        if "No versions of model" in str(e) or "No version is in the specified stage" in str(e):
            print("[INFO] Model exists but no Production version yet. First run.")
            return {"first_run": True}

        # Other unexpected MLflow exceptions: re-raise them
        raise e

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ref_df = X_train.copy()
    cur_df = X_test.copy()
    ref_df["target"] = y_train.values
    cur_df["target"] = y_test.values
    ref_df["prediction"] = model.predict(X_train)
    cur_df["prediction"] = model.predict(X_test)

    column_mapping = ColumnMapping(target="target", prediction="prediction")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), ClassificationPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)
    report.save_html(output_path)
    print(f"Evidently report saved to {output_path}")
    # Extract dict
    result = report.as_dict()
    metrics = result["metrics"]

    data_drift_result = next(m for m in metrics if m["metric"] == "DataDriftPreset")["result"]
    target_drift_result = next(m for m in metrics if m["metric"] == "TargetDriftPreset")["result"]
    cls_result         = next(m for m in metrics if m["metric"] == "ClassificationPreset")["result"]

    dataset_drift   = data_drift_result["dataset_drift"]
    share_drifted   = data_drift_result["share_of_drifted_columns"]
    target_drift    = target_drift_result["drift_detected"]
    ref_acc         = cls_result["reference"]["accuracy"]
    cur_acc         = cls_result["current"]["accuracy"]
    accuracy_drop   = ref_acc - cur_acc

    print(f"[monitoring] dataset_drift={dataset_drift}, "
          f"share_drifted={share_drifted:.3f}, "
          f"target_drift={target_drift}, "
          f"ref_acc={ref_acc:.3f}, cur_acc={cur_acc:.3f}, "
          f"accuracy_drop={accuracy_drop:.3f}, ")
          #f"should_retrain={should_retrain}")

    # Log to MLflow (new run for monitoring step)
    with mlflow.start_run(run_name=f"evidently_{datetime.now().date()}"):
        mlflow.log_artifact(output_path, artifact_path="../data/evidently_reports")
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

def decide_retrain(**context):
    ti = context["ti"]
    ev_result = ti.xcom_pull(task_ids="generate_evidently")

    if ev_result.get("first_run", False):
        print("First run detected, proceeding to train model.")
        return True

    dataset_drift = ev_result["dataset_drift"]
    share_drifted = ev_result["share_drifted"]
    target_drift  = ev_result["target_drift"]
    ref_acc       = ev_result["ref_acc"]
    cur_acc       = ev_result["cur_acc"]
    accuracy_drop = ev_result["accuracy_drop"]

    # thresholds you commented out before
    acc_drop_threshold = 0.05       # 5% absolute accuracy drop
    drift_share_threshold = 0.3     # >= 30% of features drifted

    should_retrain = (
        (dataset_drift and share_drifted >= drift_share_threshold)
        or target_drift
        or (accuracy_drop >= acc_drop_threshold)
    )

    print(
        f"should_retrain_from_metrics={should_retrain}"
    )

    # When you trigger the DAG manually, you can pass {"force_retrain": true}
    dag_run = context.get("dag_run")
    dag_conf = dag_run.conf if dag_run else {}
    force_retrain = bool(dag_conf.get("force_retrain", False))

    
    execution_date = context.get("logical_date") or context.get("execution_date")
    if execution_date is not None:
        is_monday = execution_date.weekday() == 0  # Monday = 0
    else:
        # Fallback to "now" if for some reason it's missing
        is_monday = datetime.utcnow().weekday() == 0

    if is_monday or force_retrain:
        should_retrain = True


    return should_retrain   # ShortCircuitOperator expects True/False
