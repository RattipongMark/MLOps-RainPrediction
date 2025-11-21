def run_training():
    import dagshub
    # dagshub_token = '1cff28c7e5d4b684113fd31db220311391db8688'
    # dagshub.init(
    # repo_owner='RattipongMark',
    # repo_name='chanakan_code',
    # mlflow=True,
    # token=dagshub_token
    # )
    dagshub.init(repo_owner='RattipongMark', repo_name='MLOps-RainPrediction', mlflow=True)
    

    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    from xgboost import XGBClassifier
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
        classification_report
    )
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import pandas as pd

    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset

    # =========================
    # Load data
    # =========================
    df = pd.read_csv("training_data.csv")  # เปลี่ยน path ให้ตรงกับไฟล์จริง
    features = df.drop("target", axis=1)
    target = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    scale_pos_weight = (target == 0).sum() / (target == 1).sum()

    # =========================
    # Model dictionary
    # =========================
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=1.0,
            class_weight="balanced",
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        )
    }

    # =========================
    # MLflow setup
    # =========================
    EXPERIMENT_NAME = "rain_model_comparison"
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = y_train_pred
                y_test_proba = y_test_pred

            # =========================
            # Metrics
            # =========================
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_roc_auc = roc_auc_score(y_train, y_train_proba)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("train_roc_auc", train_roc_auc)
            mlflow.log_metric("test_roc_auc", test_roc_auc)

            print(f"{model_name} -> "
                f"Train AUC: {train_roc_auc:.4f}, Test AUC: {test_roc_auc:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

            # =========================
            # ROC Curve - TRAIN
            # =========================
            plt.figure(figsize=(8, 6))
            RocCurveDisplay.from_predictions(y_train, y_train_proba, name="Train", color="blue")
            plt.title(f"{model_name} ROC Curve (Train)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("roc_curve_train.png")
            plt.close()
            mlflow.log_artifact("roc_curve_train.png", artifact_path="plots")

            # =========================
            # ROC Curve - TEST
            # =========================
            plt.figure(figsize=(8, 6))
            RocCurveDisplay.from_predictions(y_test, y_test_proba, name="Test", color="red")
            plt.title(f"{model_name} ROC Curve (Test)")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("roc_curve_test.png")
            plt.close()
            mlflow.log_artifact("roc_curve_test.png", artifact_path="plots")

            # =========================
            # Classification report
            # =========================
            report = classification_report(y_test, y_test_pred, target_names=["No Rain", "Rain"])
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt", artifact_path="reports")

            # =========================
            # Confusion Matrix Plot
            # =========================
            cm = confusion_matrix(y_test, y_test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

            # =========================
            # Log model
            # =========================
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            # =========================
            # Evidently Report
            # =========================
            ref_df = X_train.copy()
            cur_df = X_test.copy()

            ref_df["target"] = y_train.values
            cur_df["target"] = y_test.values

            ref_df["prediction"] = model.predict(X_train)
            cur_df["prediction"] = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                ref_df["proba_1"] = model.predict_proba(X_train)[:, 1]
                ref_df["proba_0"] = 1 - ref_df["proba_1"]
                cur_df["proba_1"] = model.predict_proba(X_test)[:, 1]
                cur_df["proba_0"] = 1 - cur_df["proba_1"]

            column_mapping = ColumnMapping(target="target", prediction="prediction")

            report = Report(metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                ClassificationPreset()
            ])
            report.run(reference_data=ref_df, current_data=cur_df, column_mapping=column_mapping)

            report.save_html("evidently_report.html")
            report.save_json("evidently_report.json")
            mlflow.log_artifact("evidently_report.html")
            mlflow.log_artifact("evidently_report.json")

            print("Evidently report generated and logged to MLflow artifacts.")

    # =========================
    # End of script
    # =========================
if __name__ == "__main__":
    run_training()
