"""
Feature definitions for NeuralRetail.
Calculates RFM and Lag features using PySpark for the Feature Store.
"""
import os
from pathlib import Path
import shutil
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def _resolve_data_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"

def calculate_features():
    spark = SparkSession.builder.appName("NeuralRetail-Features").master("local[*]").getOrCreate()
    data_root = _resolve_data_root()
    
    silver_path = data_root / "silver" / "transactions"
    rfm_path = data_root / "features" / "rfm"
    demand_path = data_root / "features" / "demand"
    
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver transactions not found at {silver_path}. Run transformation.py first.")
        
    print(f"[FEATURES] Calculating RFM from: {silver_path}")
    df = spark.read.parquet(str(silver_path))
    
    # 1. RFM Calculation
    max_date = df.select(F.max("transaction_timestamp")).collect()[0][0]
    
    rfm = df.groupBy("customer_id").agg(
        F.datediff(F.lit(max_date), F.max("transaction_timestamp")).alias("recency"),
        F.countDistinct("transaction_id").alias("frequency"),
        F.sum("price").alias("monetary")
    )
    rfm = rfm.withColumn("event_timestamp", F.lit(max_date).cast("timestamp"))
    
    print(f"[FEATURES] Writing RFM to: {rfm_path}")
    if rfm_path.exists(): shutil.rmtree(rfm_path)
    rfm.coalesce(1).write.mode("overwrite").parquet(str(rfm_path))
    
    # 2. Demand Lag Calculation
    print(f"[FEATURES] Calculating Demand Lags...")
    daily_sales = df.withColumn("date", F.to_date("transaction_timestamp")) \
                    .groupBy("date") \
                    .agg(F.sum("price").alias("daily_revenue"), F.count("transaction_id").alias("order_count"))
    
    window_spec = Window.orderBy("date")
    demand = daily_sales.withColumn("rev_lag_1d", F.lag("daily_revenue", 1).over(window_spec)) \
                        .withColumn("rev_lag_7d", F.lag("daily_revenue", 7).over(window_spec)) \
                        .fillna(0)
    demand = demand.withColumn("event_timestamp", F.col("date").cast("timestamp")).withColumn("store_id", F.lit("ST_001"))
    
    print(f"[FEATURES] Writing Demand to: {demand_path}")
    if demand_path.exists(): shutil.rmtree(demand_path)
    demand.coalesce(1).write.mode("overwrite").parquet(str(demand_path))
    
    print("[FEATURES] Success: Features materialized for Feast.")
    spark.stop()

if __name__ == "__main__":
    calculate_features()