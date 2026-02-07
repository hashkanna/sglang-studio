from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from studio_api.db import Base, engine, get_session
from studio_api.metrics import compare_results
from studio_api.models import Run
from studio_api.schemas import CompareRequest, CompareResponse, RunCreate, RunView, ToleranceConfig
from studio_api.settings import settings


app = FastAPI(title=settings.api_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_online_schema_migrations()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


def _stable_json_hash(payload: dict | None) -> str | None:
    if payload is None:
        return None
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _to_run_view(run: Run) -> RunView:
    return RunView(
        id=run.id,
        backend=run.backend,
        mode=run.mode,
        prompt=run.prompt,
        parameters=run.parameters,
        score_input=run.score_input,
        mask_config=run.mask_config,
        tolerance=run.tolerance,
        repro_metadata=run.repro_metadata,
        score_input_hash=run.score_input_hash,
        mask_hash=run.mask_hash,
        status=run.status,
        result_json=run.result_json,
        artifact_key=run.artifact_key,
        error=run.error,
        created_at=run.created_at,
        updated_at=run.updated_at,
        completed_at=run.completed_at,
    )


@app.post("/api/v1/runs", response_model=RunView)
def create_run(payload: RunCreate, session: Session = Depends(get_session)) -> RunView:
    score_input = payload.score_input.model_dump() if payload.score_input else None
    mask_config = payload.mask_config.model_dump() if payload.mask_config else None
    tolerance = (
        payload.tolerance.model_dump()
        if payload.tolerance
        else (ToleranceConfig().model_dump() if payload.mode == "score" else None)
    )
    repro_metadata = payload.repro_metadata.model_dump() if payload.repro_metadata else None
    prompt = payload.prompt if payload.prompt is not None else payload.score_input.query

    run = Run(
        backend=payload.backend,
        mode=payload.mode,
        prompt=prompt,
        parameters=payload.parameters,
        score_input=score_input,
        mask_config=mask_config,
        tolerance=tolerance,
        repro_metadata=repro_metadata,
        score_input_hash=_stable_json_hash(score_input),
        mask_hash=_stable_json_hash(mask_config),
        status="pending",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return _to_run_view(run)


@app.get("/api/v1/runs", response_model=list[RunView])
def list_runs(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> list[RunView]:
    rows: Sequence[Run] = session.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit)).all()
    return [_to_run_view(run) for run in rows]


@app.get("/api/v1/runs/{run_id}", response_model=RunView)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunView:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _to_run_view(run)


@app.post("/api/v1/compares", response_model=CompareResponse)
def compare_runs(payload: CompareRequest, session: Session = Depends(get_session)) -> CompareResponse:
    left = session.get(Run, payload.left_run_id)
    right = session.get(Run, payload.right_run_id)

    if left is None or right is None:
        raise HTTPException(status_code=404, detail="One or both runs not found")
    if left.status != "succeeded" or right.status != "succeeded":
        raise HTTPException(status_code=409, detail="Both runs must be succeeded before comparison")

    left_result = left.result_json or {}
    right_result = right.result_json or {}
    diff = compare_results(left_result, right_result)

    return CompareResponse(
        left_run_id=left.id,
        right_run_id=right.id,
        score_abs_diff=diff["score_abs_diff"],
        latency_ms_diff=diff["latency_ms_diff"],
        latency_pct_diff=diff["latency_pct_diff"],
        throughput_items_per_s_diff=diff["throughput_items_per_s_diff"],
    )


@app.post("/api/v1/runs/{run_id}/cancel", response_model=RunView)
def cancel_run(run_id: str, session: Session = Depends(get_session)) -> RunView:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status in {"succeeded", "failed"}:
        return _to_run_view(run)

    run.status = "failed"
    run.error = "Canceled"
    run.completed_at = datetime.now(tz=timezone.utc)
    session.commit()
    session.refresh(run)
    return _to_run_view(run)
