"""
Silver layer transformations for NeuralRetail.
Reads POS Bronze data, anonymizes PII, and promotes to Silver.
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
    
    bronze_path = data_root / "bronze" / "bronze_pos_sales"
    silver_path = data_root / "silver" / "transactions"
    
    print(f"[SILVER] Reading Bronze POS data from: {bronze_path}")
    df = spark.read.parquet(str(bronze_path))
    
    print("[SILVER] Applying Amdox Data Privacy rules (Hashing PII)...")
    # Amdox Security Requirement: Hash the email to anonymize PII
    silver_df = df.withColumn("customer_id_hashed", F.sha2(F.col("email"), 256)) \
                  .drop("email", "_source_file") \
                  .withColumnRenamed("_ingestion_timestamp", "transaction_timestamp") \
                  .withColumnRenamed("amount", "price") # Rename to price for downstream uniformity
    
    print(f"[SILVER] Writing to Silver: {silver_path}")
    if silver_path.exists():
        shutil.rmtree(silver_path)
        
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    # coalesce(1) ensures it writes as a single file for Feast compatibility later
    silver_df.coalesce(1).write.mode("overwrite").parquet(str(silver_path))
    
    print("[SILVER] Success: POS Transactions promoted to Silver.\n")
    spark.stop()

if __name__ == "__main__":
    create_silver_transactions()