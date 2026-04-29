from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='bronze_olist_ingestion',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['phase-1', 'bronze'],
) as dag:

    run_pyspark_ingestion = BashOperator(
        task_id='run_pyspark_ingestion',
        bash_command='cd /opt/airflow && python src/ingestion/ingest.py',
    )
