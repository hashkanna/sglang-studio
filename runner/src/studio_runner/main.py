from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone

from minio import Minio
from minio.error import S3Error
from sqlalchemy import text
from sqlalchemy.orm import Session

from studio_runner.adapters import run_backend_inference
from studio_runner.db import SessionLocal, engine
from studio_runner.models import Run
from studio_runner.settings import settings


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _minio_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def _ensure_bucket(client: Minio) -> None:
    try:
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
    except S3Error:
        # Runner should still function even if artifact upload has transient issues.
        pass


def _upload_result_artifact(client: Minio, run_id: str, result: dict) -> str | None:
    key = f"runs/{run_id}/result.json"
    payload = json.dumps(result, sort_keys=True).encode("utf-8")
    try:
        client.put_object(
            settings.minio_bucket,
            key,
            io.BytesIO(payload),
            len(payload),
            content_type="application/json",
        )
        return key
    except S3Error:
        return None


def _claim_pending_run(session: Session) -> dict | None:
    query = text(
        """
        SELECT id
        FROM runs
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        """
    )

    with session.begin():
        row = session.execute(query).first()
        if row is None:
            return None

        run = session.get(Run, row.id)
        if run is None:
            return None

        run.status = "running"
        run.updated_at = _utcnow()

        return {
            "id": run.id,
            "backend": run.backend,
            "mode": run.mode,
            "prompt": run.prompt,
            "parameters": run.parameters,
            "score_input": run.score_input,
            "mask_config": run.mask_config,
            "tolerance": run.tolerance,
            "repro_metadata": run.repro_metadata,
        }


def _mark_succeeded(session: Session, run_id: str, result: dict, artifact_key: str | None) -> None:
    with session.begin():
        run = session.get(Run, run_id)
        if run is None:
            return
        run.status = "succeeded"
        run.result_json = result
        run.artifact_key = artifact_key
        run.completed_at = _utcnow()
        run.updated_at = _utcnow()
        run.error = None


def _mark_failed(session: Session, run_id: str, error: str) -> None:
    with session.begin():
        run = session.get(Run, run_id)
        if run is None:
            return
        run.status = "failed"
        run.error = error[:4000]
        run.completed_at = _utcnow()
        run.updated_at = _utcnow()


def _apply_online_schema_migrations() -> None:
    statements = [
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'benchmark'",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS score_input JSON",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS mask_config JSON",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS tolerance JSON",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS repro_metadata JSON",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS score_input_hash VARCHAR(64)",
        "ALTER TABLE runs ADD COLUMN IF NOT EXISTS mask_hash VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_runs_mode ON runs (mode)",
        "CREATE INDEX IF NOT EXISTS ix_runs_score_input_hash ON runs (score_input_hash)",
        "CREATE INDEX IF NOT EXISTS ix_runs_mask_hash ON runs (mask_hash)",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def main() -> None:
    client = _minio_client()
    _ensure_bucket(client)
    _apply_online_schema_migrations()

    while True:
        session = SessionLocal()
        claimed: dict | None = None
        try:
            claimed = _claim_pending_run(session)
            if claimed is None:
                time.sleep(settings.poll_interval_seconds)
                continue

            result = run_backend_inference(
                run_id=claimed["id"],
                backend=claimed["backend"],
                prompt=claimed["prompt"],
                parameters=claimed["parameters"] or {},
            )
            artifact_key = _upload_result_artifact(client, claimed["id"], result)
            _mark_succeeded(session, claimed["id"], result, artifact_key)
        except Exception as exc:  # pragma: no cover - process-level safety
            run_id = claimed["id"] if claimed else "unknown"
            if run_id != "unknown":
                _mark_failed(session, run_id, f"Runner failure: {exc}")
        finally:
            session.close()


if __name__ == "__main__":
    main()
