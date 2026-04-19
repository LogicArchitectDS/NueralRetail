from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Core application settings loading environment variables.
    These are populated from the .env file or system environment variables.
    """

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "neuralretail"
    postgres_user: str = "nr_user"
    postgres_password: str

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5002"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str
    aws_secret_access_key: str
    mlflow_artifact_bucket: str = "nr-mlflow-artifacts"

    # Feast Feature Store
    feast_redis_host: str = "localhost"
    feast_redis_port: int = 6379
    feast_s3_bucket: str = "nr-feature-store"

    # JWT Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    api_key_hash: str

    # HashiCorp Vault
    vault_addr: str = "http://localhost:8200"
    vault_token: str
    vault_mount_point: str = "neuralretail"

    # External APIs
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"
    timegpt_api_key: str

    # Airflow
    airflow_fernet_key: str
    airflow_secret_key: str

    # Alerting
    slack_webhook_url: str
    pagerduty_routing_key: str

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    allowed_origins: str = "http://localhost:8501,http://localhost:8000"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Singleton settings instance
settings = Settings()
