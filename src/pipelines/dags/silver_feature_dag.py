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
    dag_id='silver_feature_engineering',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['phase-2', 'silver', 'features'],
) as dag:

    run_silver_transformation = BashOperator(
        task_id='run_silver_transformation',
        bash_command='cd /opt/airflow && python src/features/transformation.py',
    )

    run_feature_engineering = BashOperator(
        task_id='run_feature_engineering',
        bash_command='cd /opt/airflow && python src/features/definitions.py',
    )

    apply_feast = BashOperator(
        task_id='apply_feast',
        bash_command='cd /opt/airflow/src/features/feature_repo && feast apply',
    )

    run_silver_transformation >> run_feature_engineering >> apply_feast
