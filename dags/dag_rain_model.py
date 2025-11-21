from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
# from airflow.operators.shortcircuit import ShortCircuitOperator
from datetime import datetime
from script.task import load_data_from_api, preprocess_data, feature_selection, train_models, save_best_model, generate_evidently, decide_retrain

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

    save_best_model_task = PythonOperator(task_id="save_best_model", python_callable=save_best_model)

    # -----------------------------

    load_data_task >> preprocess_task >> feature_selection_task >> generate_evidently_task \
    >> decide_retrain_task >> train_models_task >> save_best_model_task
