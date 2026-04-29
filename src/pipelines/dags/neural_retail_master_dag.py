from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess

# Default arguments for the DAG
default_args = {
    'owner': 'neuralretail',
    'depends_on_past': False,
    'start_date': datetime(2026, 4, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_script(script_path):
    """Utility to run project scripts via subprocess."""
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Script {script_path} failed: {result.stderr}")
    print(result.stdout)

with DAG(
    'neural_retail_master_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for NeuralRetail platform',
    schedule_interval=timedelta(days=7),
    catchup=False,
    tags=['production', 'ml_lifecycle'],
) as dag:

    # 1. Foundation: Ingestion
    start_pipeline = EmptyOperator(task_id='start_pipeline')
    
    ingest_bronze = EmptyOperator(task_id='ingest_bronze_olist')

    # 2. Silver: Feature Engineering
    build_features = PythonOperator(
        task_id='build_sales_features',
        python_callable=run_script,
        op_args=['src/features/build_features.py'],
    )
    
    generate_churn_features = PythonOperator(
        task_id='generate_churn_features',
        python_callable=run_script,
        op_args=['src/features/generate_churn_features.py'],
    )

    # 3. Gold: Model Training
    train_xgboost = PythonOperator(
        task_id='train_xgboost_churn',
        python_callable=run_script,
        op_args=['src/models/train_xgboost.py'],
    )
    
    train_stacked_churn = PythonOperator(
        task_id='train_stacked_churn_ensemble',
        python_callable=run_script,
        op_args=['src/models/train_churn_stacked.py'],
    )
    
    train_segmentation = EmptyOperator(task_id='train_kmeans_segmentation')

    # 4. MLOps: Monitoring & Validation
    drift_check = EmptyOperator(task_id='evidently_drift_check')
    
    validate_model_performance = EmptyOperator(task_id='validate_metrics_against_thresholds')

    end_pipeline = EmptyOperator(task_id='end_pipeline')

    # Dependency Graph
    start_pipeline >> ingest_bronze
    ingest_bronze >> [build_features, generate_churn_features]
    generate_churn_features >> [train_xgboost, train_stacked_churn]
    build_features >> train_segmentation
    [train_xgboost, train_stacked_churn, train_segmentation] >> drift_check
    drift_check >> validate_model_performance >> end_pipeline
