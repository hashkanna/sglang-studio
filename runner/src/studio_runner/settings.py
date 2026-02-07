from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_dsn: str = "postgresql+psycopg://studio:studio@postgres:5432/studio"
    poll_interval_seconds: float = 1.0

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_bucket: str = "studio-artifacts"
    minio_secure: bool = False

    model_config = SettingsConfigDict(env_prefix="STUDIO_", extra="ignore")


settings = Settings()
