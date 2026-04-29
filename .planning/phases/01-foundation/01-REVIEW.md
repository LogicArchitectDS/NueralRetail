---
phase: 01-foundation
reviewed: 2024-05-15T14:30:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/ingestion/spark_ingest.py
  - src/ingestion/config.py
  - src/pipelines/dags/ingestion_dag.py
  - src/quality/pos_expectations.py
  - tests/unit/test_ingestion.py
findings:
  critical: 0
  warning: 5
  info: 3
  total: 8
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2024-05-15
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

The implementation of Phase 1 provides a solid foundation for data ingestion using Spark, Delta Lake, and Great Expectations. The modularity of settings and the integration with Airflow and OpenLineage are well-structured. However, there are several portability and reliability issues, particularly regarding local filesystem assumptions and incomplete data quality checks.

## Warnings

### WR-01: Brittle `sys.path` Manipulation
**File:** `src/ingestion/spark_ingest.py:9-16`
**Issue:** The script attempts to fix import issues by dynamically modifying `sys.path`. This is an anti-pattern that indicates the environment is not correctly configured for the package structure. It can lead to unpredictable behavior when the script is executed from different working directories or as part of a larger pipeline.
**Fix:**
Ensure the project root is in `PYTHONPATH` during execution (e.g., in the Airflow environment or via `spark-submit --py-files`). Remove the `sys.path` hack:
```python
from src.ingestion.config import settings
from src.quality.pos_expectations import validate_pos_data
```

### WR-02: Local Filesystem Dependency
**File:** `src/ingestion/spark_ingest.py:42-45`
**Issue:** The usage of `os.path.exists`, `os.path.isdir`, and `os.listdir` assumes that the landing zone is a local or network-mounted filesystem. This will fail if the landing zone is transitioned to cloud storage (e.g., S3, ADLS, GCS).
**Fix:**
Use Spark's built-in path handling or a library like `fsspec` that supports multiple protocols. For checking if data exists, you can attempt to read with Spark and handle the error or use `dbutils.fs` if in Databricks.
```python
# Simplified check using Spark (might need tuning for performance)
try:
    df = spark.read.format("csv")...load(source_path)
    if df.isEmpty():
        return
except Exception as e:
    # Handle missing path
    return
```

### WR-03: Incomplete Data Quality Validation
**File:** `src/ingestion/spark_ingest.py:51`
**Issue:** `df.limit(1000).toPandas()` only validates the first 1000 rows of the dataset. This "sample-based" validation can easily miss data quality issues in larger datasets and provides a false sense of security.
**Fix:**
Integrate Great Expectations directly with the Spark DataFrame to validate the entire dataset without converting to Pandas.
```python
# In pos_expectations.py, use SparkExecutionEngine instead of PandasExecutionEngine
# In spark_ingest.py:
is_valid = validate_pos_data(df) # Pass the full Spark DF
```

### WR-04: Redundant DQ Suite Recreation
**File:** `src/quality/pos_expectations.py:58`
**Issue:** `create_pos_expectation_suite` is called inside `validate_pos_data`, meaning the expectation suite is recreated and saved to the context on every validation run. This is inefficient and can cause metadata bloat or concurrency issues.
**Fix:**
Separate the suite creation from the validation. Create the suite once as part of a setup script or check for its existence before creating it.
```python
def validate_pos_data(df: pd.DataFrame):
    context = ge.get_context()
    suite_name = "pos_sales_suite"
    if suite_name not in context.list_expectation_suite_names():
        create_pos_expectation_suite(context)
    ...
```

### WR-05: Silent PII Column Omission
**File:** `src/ingestion/spark_ingest.py:32`
**Issue:** If a column configured for PII hashing is missing from the source DataFrame, the script silently skips it. This could lead to accidental PII leakage if a column name changes or is misspelled in the configuration.
**Fix:**
Log a warning or raise an exception if a mandatory PII column is missing.
```python
for col in columns:
    if col in df.columns:
        # hash...
    else:
        print(f"WARNING: PII column '{col}' not found in source schema.")
```

## Info

### IN-01: Inconsistent Ingestion Timestamps
**File:** `src/ingestion/spark_ingest.py:66`
**Issue:** `_ingestion_date` uses Python's `datetime.now()` (local time), while `_ingestion_timestamp` uses Spark's `current_timestamp()` (cluster time). These might differ in distributed environments.
**Fix:**
Use Spark functions for both to ensure consistency across the cluster.
```python
df = df.withColumn("_ingestion_timestamp", F.current_timestamp()) \
       .withColumn("_ingestion_date", F.to_date(F.col("_ingestion_timestamp")))
```

### IN-02: Missing Logger Usage
**File:** `src/ingestion/spark_ingest.py`
**Issue:** The script uses `print()` for logging. In an Airflow/Spark environment, using the standard `logging` library is preferred for better log management and level filtering.
**Fix:**
```python
import logging
logger = logging.getLogger(__name__)
# Replace print() with logger.info() or logger.error()
```

### IN-03: Inconsistent Spark/PySpark Versions
**File:** `docker/airflow/Dockerfile`
**Issue:** `SPARK_VERSION` is set to `3.5.0` but `pyspark` is installed as `3.5.1`. While usually compatible, it is best practice to keep these versions identical to avoid unexpected behavior.
**Fix:**
Set both to the same version (e.g., `3.5.1`).

---

_Reviewed: 2024-05-15_
_Reviewer: gsd-code-reviewer_
_Depth: standard_
