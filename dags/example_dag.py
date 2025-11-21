from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="example_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    task1 = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Hello Airflow! mm'"
    )
