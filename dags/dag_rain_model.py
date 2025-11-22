from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import ShortCircuitOperator

# from airflow.operators.shortcircuit import ShortCircuitOperator
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../script')))
from task import load_data_from_api, preprocess_data, feature_selection, train_models, save_best_model, generate_evidently, decide_retrain, upload_dataset_to_dagshub
import dagshub
import mlflow

dagshub.init(repo_owner='RattipongMark', repo_name='MLOps-RainPrediction', mlflow=True)

with DAG(
    dag_id="rain_model_multi_task_refactored",
    start_date=datetime(2025,1,1),
    schedule="@daily",
    catchup=False,
    tags=["rain"]
) as dag:

    load_data_task = PythonOperator(task_id="load_data", python_callable=load_data_from_api)

    preprocess_task = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)

    feature_selection_task = PythonOperator(task_id="feature_selection", python_callable=feature_selection)

    generate_evidently_task = PythonOperator(task_id="generate_evidently", python_callable=generate_evidently)

    decide_retrain_task = ShortCircuitOperator(task_id="decide_retrain_task", python_callable=decide_retrain)

    train_models_task = PythonOperator(task_id="train_models", python_callable=train_models)

    upload_dataset_to_dagshub_task = PythonOperator(task_id="upload_dataset_to_dagshub", python_callable=upload_dataset_to_dagshub)

    save_best_model_task = PythonOperator(task_id="save_best_model", python_callable=save_best_model)

    # -----------------------------

    load_data_task >> preprocess_task >> feature_selection_task >> generate_evidently_task >> decide_retrain_task >> train_models_task >> save_best_model_task >> upload_dataset_to_dagshub_task
