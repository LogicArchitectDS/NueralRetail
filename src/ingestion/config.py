from pydantic_settings import BaseSettings
from typing import List, Dict

class IngestionSettings(BaseSettings):
    LANDING_ZONE: str = "data/landing"
    BRONZE_ZONE: str = "data/bronze"
    
    # Define PII columns per source
    PII_COLUMNS: Dict[str, List[str]] = {
        "pos": ["customer_id", "email", "customer_name", "phone"],
        "erp": ["supplier_contact", "employee_id"]
    }
    
    # Delta table names
    TABLES: Dict[str, str] = {
        "pos": "bronze_pos_sales",
        "erp": "bronze_erp_inventory"
    }

    class Config:
        env_prefix = "INGEST_"

settings = IngestionSettings()
