from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from studio_api.db import Base, engine, get_session
from studio_api.metrics import compare_results
from studio_api.models import Run
from studio_api.schemas import CompareRequest, CompareResponse, RunCreate, RunView
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _to_run_view(run: Run) -> RunView:
    return RunView(
        id=run.id,
        backend=run.backend,
        prompt=run.prompt,
        parameters=run.parameters,
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
    run = Run(
        backend=payload.backend,
        prompt=payload.prompt,
        parameters=payload.parameters,
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
