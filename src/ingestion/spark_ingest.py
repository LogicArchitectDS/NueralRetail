import sys
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from openlineage.spark import SparkOpenLineageExtension

try:
    from src.ingestion.config import settings
    from src.quality.pos_expectations import validate_pos_data
except ImportError:
    sys.path.append(os.getcwd())
    from src.ingestion.config import settings
    from src.quality.pos_expectations import validate_pos_data

def create_spark_session():
    """Create Spark session with OpenLineage and Delta extensions."""
    return (SparkSession.builder
            .appName("NeuralRetail-Ingestion")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension,openlineage.spark.SparkOpenLineageExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.openlineage.transport.type", "http")
            .config("spark.openlineage.transport.url", os.getenv("OPENLINEAGE_URL", "http://marquez:5000"))
            .config("spark.openlineage.namespace", "neuralretail")
            .getOrCreate())

def hash_pii_columns(df, columns):
    """Applies SHA-256 hashing to PII columns."""
    for col in columns:
        if col in df.columns:
            salt = os.getenv("PII_SALT", "SIMULATED_SALT_2026") 
            df = df.withColumn(col, F.sha2(F.concat(F.col(col), F.lit(salt)), 256))
    return df

def ingest_source(spark, source_name):
    """Ingests a single source from landing to bronze with DQ and Lineage."""
    source_path = os.path.join(settings.LANDING_ZONE, source_name)
    target_table = settings.TABLES.get(source_name)
    pii_cols = settings.PII_COLUMNS.get(source_name, [])

    if not os.path.exists(source_path):
        return

    # 1. Read (OpenLineage will track this input)
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(source_path)

    # 2. Data Quality Gate (Great Expectations)
    if source_name == "pos":
        # Convert a sample to pandas for GE validation (standard GE-Spark integration is more complex)
        pdf_sample = df.limit(1000).toPandas() 
        is_valid = validate_pos_data(pdf_sample)
        if not is_valid:
            print(f"DQ FAILURE: {source_name} failed Great Expectations validation. Aborting.")
            return

    # 3. Transformations (PII Hashing)
    df = hash_pii_columns(df, pii_cols)

    # 4. Metadata
    df = df.withColumn("_ingestion_timestamp", F.current_timestamp()) \
           .withColumn("_source_file", F.input_file_name()) \
           .withColumn("_ingestion_date", F.lit(datetime.now().strftime("%Y-%m-%d")))

    # 5. Write (OpenLineage will track this output)
    target_path = os.path.join(settings.BRONZE_ZONE, target_table)
    (df.write.format("delta")
     .mode("append")
     .partitionBy("_ingestion_date")
     .save(target_path))
    
    print(f"Successfully ingested {source_name} to {target_table}")

if __name__ == "__main__":
    spark_session = create_spark_session()
    for source in settings.TABLES.keys():
        ingest_source(spark_session, source)
    spark_session.stop()
