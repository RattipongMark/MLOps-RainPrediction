import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
# from my_pipeline_module import (
#     load_data,
#     preprocess_data,
#     feature_selection,
#     train_models,
#     generate_evidently
# )

import joblib
import os

# -----------------------------
# 1. Load data
# -----------------------------
def load_data(input_path="/Users/chanakan/Documents/CPE393_MLOPS/test-mlflow-evidently/open-meteo-13.67N100.49E5m.csv",
              output_path="data_clean.csv"):
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df[(df["time"] >= "2014-01-01") & (df["time"] < "2025-01-01")]
    df.to_csv(output_path, index=False)
    print(f"Data loaded and saved to {output_path}")


# -----------------------------
# 2. Preprocess data
# -----------------------------
def preprocess_data(input_path="data_clean.csv", output_path="data_preprocessed.csv"):
    df = pd.read_csv(input_path)
    df["target"] = (df["rain (mm)"] > 0.1).astype(int)
    df = df.drop(["time", "rain (mm)", "weather_code (wmo code)", "is_day ()"], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


# -----------------------------
# 3. Feature selection (updated)
# -----------------------------
def feature_selection(input_path="data_preprocessed.csv",
                      features_path="selected_features.json",
                      importance_path="feature_importance.csv",
                      output_path="data_selected.csv",
                      corr_threshold=0.7,
                      importance_cutoff=0.90):

    print("===== FEATURE SELECTION START =====")

    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # -------------------------------------------------------------
    # STEP 1 — Train an XGBoost model to get feature importances
    # -------------------------------------------------------------
    print("Training XGBoost for feature importance...")
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

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Save raw importance for inspection
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved raw feature importance → {importance_path}")

    # -------------------------------------------------------------
    # STEP 2 — Remove correlated features
    # -------------------------------------------------------------
    print("Dropping correlated features...")
    corr = X.corr().abs()
    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop_corr = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > corr_threshold)
    ]

    print(f"Dropping {len(to_drop_corr)} correlated features.")

    X_uncorr = X.drop(columns=to_drop_corr)

    # -------------------------------------------------------------
    # STEP 3 — Apply cumulative importance cutoff
    # -------------------------------------------------------------
    print("Applying cumulative importance selection...")

    # Only keep importance values for remaining (uncorrelated) features
    importance_df_uncorr = importance_df[
        importance_df["Feature"].isin(X_uncorr.columns)
    ].reset_index(drop=True)

    importance_df_uncorr["Cumulative"] = (
        importance_df_uncorr["Importance"].cumsum() /
        importance_df_uncorr["Importance"].sum()
    )

    cutoff_index = np.argmax(
        importance_df_uncorr["Cumulative"] >= importance_cutoff
    ) + 1

    final_selected_features = \
        importance_df_uncorr["Feature"].iloc[:cutoff_index].tolist()

    print(f"Selected {len(final_selected_features)} features after cumulative cutoff.")

    # -------------------------------------------------------------
    # STEP 4 — Save final selected features
    # -------------------------------------------------------------
    with open(features_path, "w") as f:
        json.dump(final_selected_features, f)

    print(f"Final selected features saved → {features_path}")

    # Save final dataset (selected features + target)
    df_selected = df[final_selected_features + ["target"]]
    df_selected.to_csv(output_path, index=False)

    print(f"Selected dataframe saved → {output_path}")
    print("===== FEATURE SELECTION COMPLETE =====")

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

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric="logloss"
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=1.0,
            class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10,
            class_weight="balanced", random_state=42
        )
    }

    # Train + log MLflow
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else y_train_pred
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Metrics
            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("train_auc", roc_auc_score(y_train, y_train_proba))
            mlflow.log_metric("test_auc", roc_auc_score(y_test, y_test_proba))

            # Log model
            model_path = os.path.join(model_dir, model_name)
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path=model_path)
            else:
                mlflow.sklearn.log_model(model, artifact_path=model_path)

            print(f"{model_name} trained and logged.")


# -----------------------------
# 5. Generate Evidently
# -----------------------------
def generate_evidently(input_path="data_selected.csv",
                       model_name="XGBoost",
                       model_dir="models",
                       output_path="evidently.html"):
    df = pd.read_csv(input_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Load MLflow model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}/model"
    model = mlflow.sklearn.load_model(model_uri) if model_name != "XGBoost" else mlflow.xgboost.load_model(model_uri)

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

with DAG(
    dag_id="rain_model_multi_task",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mark"],
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    feature_selection_task = PythonOperator(
        task_id="feature_selection",
        python_callable=feature_selection
    )

    train_models_task = PythonOperator(
        task_id="train_models",
        python_callable=train_models
    )

    generate_evidently_task = PythonOperator(
        task_id="generate_evidently",
        python_callable=generate_evidently
    )

    # DAG dependencies
    load_data_task >> preprocess_task >> feature_selection_task >> train_models_task >> generate_evidently_task
