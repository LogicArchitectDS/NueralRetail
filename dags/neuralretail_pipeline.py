# Neural Retail Pipeline DAG
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None
    PythonOperator = None
    BranchPythonOperator = None
    EmptyOperator = None
    print("Airflow not available — DAG definition skipped in this environment")
