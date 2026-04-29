"""
Bronze data ingestion for the Olist landing zone.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import uuid

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


TEMP_WRITE_ROOT = Path("/tmp/neuralretail_bronze")


def _candidate_data_roots() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates: list[Path] = []

    env_value = os.environ.get("NEURALRETAIL_DATA_DIR")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.extend(
        [
            Path("/opt/airflow/data"),
            Path("/home/seshu/NueralRetail_Solo/data"),
            project_root / "data",
            Path.cwd() / "data",
        ]
    )
    return candidates


def _resolve_data_root() -> Path:
    """Return the mounted data directory for local runs and Airflow container runs."""
    candidates = _candidate_data_roots()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find a data directory. Checked: {searched}")


def _list_olist_tables(landing_dir: Path) -> list[str]:
    """Discover all CSV files in the Olist landing zone."""
    if not landing_dir.exists():
        raise FileNotFoundError(f"Olist landing directory does not exist: {landing_dir}")

    tables = sorted(path.stem for path in landing_dir.glob("*.csv"))
    if not tables:
        raise FileNotFoundError(f"No CSV files found in Olist landing directory: {landing_dir}")
    return tables


def _get_spark() -> SparkSession:
    """Return the project-wide SparkSession."""
    return (
        SparkSession.builder
        .appName("NeuralRetail-Bronze-Ingestion")
        .master("local[*]")
        .getOrCreate()
    )


def ingest_to_bronze(table_name: str) -> DataFrame:
    """Ingest a single Olist CSV table into the Bronze Parquet layer."""
    spark = _get_spark()
    data_root = _resolve_data_root()
    landing_path = data_root / "landing" / "olist" / f"{table_name}.csv"
    bronze_path = data_root / "bronze" / "olist" / table_name
    temp_bronze_path = TEMP_WRITE_ROOT / f"{table_name}_{uuid.uuid4().hex}"

    print(f"[INGEST] Reading  : {landing_path}")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(landing_path))
    )

    df = df.withColumn("_ingest_timestamp", F.current_timestamp())
    temp_bronze_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INGEST] Writing  : {bronze_path}  ({df.count()} rows)")
    df.coalesce(1).write.mode("overwrite").parquet(str(temp_bronze_path))

    shutil.rmtree(bronze_path, ignore_errors=True)
    shutil.move(str(temp_bronze_path), str(bronze_path))

    print(f"[INGEST] Success: {table_name} promoted to Bronze.\n")
    return df


if __name__ == "__main__":
    spark = _get_spark()
    try:
        landing_dir = _resolve_data_root() / "landing" / "olist"
        for table in _list_olist_tables(landing_dir):
            ingest_to_bronze(table)
    finally:
        spark.stop()
        print("[INGEST] All tables ingested. SparkSession stopped.")
