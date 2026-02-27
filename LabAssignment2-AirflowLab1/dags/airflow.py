from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.iris_lab import load_data

from src.iris_lab import (
    load_data,
    preprocess_iris,
    train_save_models,
    load_models_and_report,
)

default_args = {
    "owner": "your_name",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="Airflow_Iris_Clustering_Compare",
    default_args=default_args,
    description="Offline Iris clustering: KMeans vs Agglomerative",
    catchup=False,
) as dag:

    load_task = PythonOperator(
        task_id="load_iris",
        python_callable=load_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_iris",
        python_callable=preprocess_iris,
        op_args=[load_task.output],
    )

    train_task = PythonOperator(
        task_id="train_save_models",
        python_callable=train_save_models,
        op_args=[preprocess_task.output],
        op_kwargs={"k_min": 2, "k_max": 10},
    )

    report_task = PythonOperator(
        task_id="load_models_and_report",
        python_callable=load_models_and_report,
        op_args=[train_task.output],
    )

    load_task >> preprocess_task >> train_task >> report_task


if __name__ == "__main__":
    dag.test()
