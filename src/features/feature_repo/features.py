from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int32, Int64, Float32, Float64, String
import os
from pathlib import Path

# 1. Dynamically locate the static Parquet files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RFM_PATH = os.path.join(PROJECT_ROOT, "data", "features", "rfm_features.parquet")
DEMAND_PATH = os.path.join(PROJECT_ROOT, "data", "features", "demand_features.parquet")

# 2. Define the Data Sources
rfm_source = FileSource(
    path=RFM_PATH,
    timestamp_field="event_timestamp",
)

demand_source = FileSource(
    path=DEMAND_PATH,
    timestamp_field="event_timestamp",
)

# 3. Define the Entities (The Primary Keys)
customer = Entity(name="customer", join_keys=["customer_id"])
store = Entity(name="store", join_keys=["store_id"])

# 4. Define the Feature Views (What the ML Models will query)
rfm_fv = FeatureView(
    name="customer_rfm",
    entities=[customer],
    ttl=timedelta(days=3650), # How far back to look for data
    schema=[
        Field(name="recency", dtype=Int32),
        Field(name="frequency", dtype=Int64),
        Field(name="monetary", dtype=Float64),
    ],
    source=rfm_source,
)

demand_fv = FeatureView(
    name="store_demand_lags",
    entities=[store],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="daily_revenue", dtype=Float64),
        Field(name="order_count", dtype=Int64),
        Field(name="rev_lag_1d", dtype=Float64),
        Field(name="rev_lag_7d", dtype=Float64),
    ],
    source=demand_source,
)