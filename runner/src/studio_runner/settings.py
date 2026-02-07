from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    db_dsn: str = "postgresql+psycopg://studio:studio@postgres:5432/studio"
    poll_interval_seconds: float = 1.0

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_bucket: str = "studio-artifacts"
    minio_secure: bool = False

    local_artifacts_root: str = "/tmp/studio-run-artifacts"

    sglang_jax_adapter_mode: str = "auto"
    sglang_jax_root: str = "/workspaces/sglang-jax"
    sglang_jax_bench_entrypoint: str | None = None
    sglang_jax_bench_unittest_selector: str | None = None
    sglang_jax_bench_command: str | None = None
    sglang_jax_python_executable: str = "python3"
    sglang_jax_bench_timeout_seconds: int = 900

    sglang_pytorch_adapter_mode: str = "auto"
    sglang_pytorch_root: str = "/workspaces/sglang"
    sglang_pytorch_bench_entrypoint: str | None = None
    sglang_pytorch_bench_command: str | None = None
    sglang_pytorch_python_executable: str = "python3"
    sglang_pytorch_bench_timeout_seconds: int = 600

    model_config = SettingsConfigDict(env_prefix="STUDIO_", extra="ignore")


settings = Settings()
