import pytest
import os
from unittest.mock import patch
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-spark") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

def test_hash_pii_columns(spark):
    # Mock settings.PII_SALT for the test
    with patch("src.ingestion.spark_ingest.settings") as mock_settings:
        mock_settings.PII_SALT = "test_salt"
        from src.ingestion.spark_ingest import hash_pii_columns
        
        data = [("123", "test@example.com"), ("456", "user@domain.com")]
        df = spark.createDataFrame(data, ["customer_id", "email"])
        
        hashed_df = hash_pii_columns(df, ["customer_id", "email"])
        
        results = hashed_df.collect()
        
        # Check that the values are hashed (not equal to original)
        assert results[0]["customer_id"] != "123"
        assert results[0]["email"] != "test@example.com"
        # Check that they are hex strings (SHA-256)
        assert len(results[0]["customer_id"]) == 64
