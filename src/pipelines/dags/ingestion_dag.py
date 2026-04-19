from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'amdox-ds',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_neuralretail_ingestion',
    default_args=default_args,
    description='Daily ingestion of POS and ERP data to Delta Bronze layer',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=['neuralretail', 'ingestion', 'bronze'],
) as dag:

    # Path to the spark script relative to AIRFLOW_HOME or absolute
    SPARK_SCRIPT_PATH = os.path.join(os.environ.get('AIRFLOW_HOME', '/opt/airflow'), 'dags/src/ingestion/spark_ingest.py')

    ingest_task = SparkSubmitOperator(
        task_id='ingest_pos_and_erp',
        application=SPARK_SCRIPT_PATH,
        name='neuralretail_ingest_job',
        conn_id='spark_default',
        packages='io.delta:delta-spark_2.12:3.0.0,io.openlineage:openlineage-spark_2.12:1.11.3',
        conf={
            'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension,openlineage.spark.SparkOpenLineageExtension',
            'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog',
            # --- MOVED FROM PYTHON TO AIRFLOW ---
            'spark.openlineage.transport.type': 'http',
            'spark.openlineage.transport.url': 'http://marquez:5000', 
            'spark.openlineage.namespace': 'neuralretail'
        },
        verbose=True
    )

    ingest_task
