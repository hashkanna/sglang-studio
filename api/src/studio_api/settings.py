from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_dsn: str = "postgresql+psycopg://studio:studio@postgres:5432/studio"
    api_name: str = "SGLang Studio API"

    model_config = SettingsConfigDict(env_prefix="STUDIO_", extra="ignore")


settings = Settings()
