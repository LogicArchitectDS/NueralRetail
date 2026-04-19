import logging
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.ingestion.config import settings
from src.quality.pos_expectations import validate_pos_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create Spark session with Delta extensions."""
    return (SparkSession.builder
            .appName("NeuralRetail-Ingestion")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())

def hash_pii_columns(df, columns):
    """Applies SHA-256 hashing to PII columns using the configured salt."""
    for col in columns:
        if col in df.columns:
            df = df.withColumn(col, F.sha2(F.concat(F.col(col), F.lit(settings.PII_SALT)), 256))
        else:
            logger.warning(f"PII column '{col}' not found in source schema.")
    return df

def get_dir_size(path):
    """Calculates total size of a directory or file in bytes."""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def ingest_source(spark, source_name):
    """Ingests a single source from landing to bronze with DQ and Lineage."""
    source_path = os.path.join(settings.LANDING_ZONE, source_name)
    target_table = settings.TABLES.get(source_name)
    pii_cols = settings.PII_COLUMNS.get(source_name, [])

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path {source_path} does not exist for source {source_name}")
    
    # Check for empty files or directories
    if get_dir_size(source_path) == 0:
        logger.warning(f"Skipping {source_name}: Source is empty (0 bytes).")
        return

    logger.info(f"Reading source {source_name} from {source_path}")
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(source_path)

    # 2. Data Quality Gate (Great Expectations)
    if source_name == "pos":
        # Convert a sample to pandas for GE validation
        # Note: For production, consider Spark-native GE validation or Pandera
        pdf_sample = df.limit(1000).toPandas() 
        is_valid = validate_pos_data(pdf_sample)
        if not is_valid:
            error_msg = f"DQ FAILURE: {source_name} failed Great Expectations validation. Aborting."
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 3. Transformations (PII Hashing)
    df = hash_pii_columns(df, pii_cols)

    # 4. Metadata
    df = df.withColumn("_ingestion_timestamp", F.current_timestamp()) \
           .withColumn("_source_file", F.input_file_name()) \
           .withColumn("_ingestion_date", F.to_date(F.col("_ingestion_timestamp")))

    # 5. Write
    target_path = os.path.join(settings.BRONZE_ZONE, target_table)
    logger.info(f"Writing {source_name} to {target_path}")
    (df.write.format("delta")
     .mode("append")
     .partitionBy("_ingestion_date")
     .save(target_path))
    
    logger.info(f"Successfully ingested {source_name} to {target_table}")

if __name__ == "__main__":
    spark_session = create_spark_session()
    try:
        for source in settings.TABLES.keys():
            ingest_source(spark_session, source)
    finally:
        spark_session.stop()
