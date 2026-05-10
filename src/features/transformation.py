"""
Silver layer transformations for NeuralRetail.
Promotes canonical Bronze transactions to Silver with PII anonymization.
"""
import os
from pathlib import Path
import shutil
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def _resolve_data_root() -> Path:
    """Dynamically resolve the project data directory."""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find data directory at: {data_dir}")
    return data_dir

def create_silver_transactions():
    spark = SparkSession.builder.appName("NeuralRetail-Silver").master("local[*]").getOrCreate()
    data_root = _resolve_data_root()
    
    # Updated to use the new canonical Bronze source
    bronze_path = data_root / "bronze" / "transactions.parquet"
    silver_path = data_root / "silver" / "transactions"
    
    print(f"[SILVER] Reading canonical Bronze data from: {bronze_path}")
    df = spark.read.parquet(str(bronze_path))
    
    print("[SILVER] Applying Amdox Data Privacy rules (Anonymizing Customer ID)...")
    # Hash the customer_id to anonymize PII
    # Map canonical fields to downstream Silver schema
    silver_df = df.withColumn("customer_id_anonymized", F.sha2(F.col("customer_id").cast("string"), 256)) \
                  .drop("customer_id") \
                  .withColumnRenamed("customer_id_anonymized", "customer_id") \
                  .withColumnRenamed("invoice_date", "transaction_timestamp") \
                  .withColumnRenamed("invoice_no", "transaction_id") \
                  .withColumnRenamed("revenue", "price")
    
    # Cast timestamp correctly if needed
    silver_df = silver_df.withColumn("transaction_timestamp", F.to_timestamp("transaction_timestamp"))
    
    print(f"[SILVER] Writing to Silver: {silver_path}")
    if silver_path.exists():
        shutil.rmtree(silver_path)
        
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    # coalesce(1) ensures it writes as a single file for Feast compatibility
    silver_df.coalesce(1).write.mode("overwrite").parquet(str(silver_path))
    
    print("[SILVER] Success: Canonical Transactions promoted to Silver.\n")
    spark.stop()

if __name__ == "__main__":
    create_silver_transactions()
