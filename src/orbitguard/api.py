"""HTTP API: screening results from the database and final-risk forecasts for CDM events.

Run with ``orbitguard serve`` (or ``uvicorn orbitguard.api:app``). Interactive
documentation is served at ``/docs``.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orbitguard import __version__
from orbitguard.covariance import population
from orbitguard.store import (
    ConjunctionQuery, ConjunctionRow, Run, get_run_elements, latest_run, list_pipeline_runs,
    list_runs, make_engine, query_conjunctions,
)

log = logging.getLogger(__name__)

DEFAULT_RISK_MODEL_PATH = Path("data/models/cdm_risk_model.joblib")
WEB_DIR = Path(__file__).parent / "web"
MAX_PAGE = 500


# ---------------------------------------------------------------- response schemas

def _as_utc(t: datetime) -> datetime:
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    generated_at: datetime
    groups: list[str]
    objects_screened: int
    window_hours: float
    threshold_km: float
    hard_body_radius_m: float | None
    covariance_source: str | None
    n_conjunctions: int

    @field_validator("groups", mode="before")
    @classmethod
    def _split_groups(cls, v):
        return v.split(",") if isinstance(v, str) else v

    @field_validator("generated_at")
    @classmethod
    def _utc(cls, v):
        return _as_utc(v)


class ConjunctionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    norad_id_1: int
    name_1: str
    norad_id_2: int
    name_2: str
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    co_orbital: bool
    pc: float | None = Field(description="Probability of collision; null for co-orbital pairs")

    @field_validator("tca")
    @classmethod
    def _utc(cls, v):
        return _as_utc(v)


class ConjunctionPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ConjunctionOut]


class ElementSetOut(BaseModel):
    norad_id: int
    name: str
    population: str = Field(description="starlink, oneweb, debris or other")
    tle1: str
    tle2: str


class RunElementsOut(BaseModel):
    run_id: int
    generated_at: datetime
    count: int
    objects: list[ElementSetOut]


class PipelineRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    started_at: datetime
    finished_at: datetime
    status: str
    groups: list[str]
    run_id: int | None
    metrics: dict
    drift_alerts: list[dict]
    validation: list[dict]
    message: str

    @field_validator("groups", mode="before")
    @classmethod
    def _split_groups(cls, v):
        return v.split(",") if isinstance(v, str) else v

    @field_validator("metrics", "drift_alerts", "validation", mode="before")
    @classmethod
    def _parse_json(cls, v):
        return json.loads(v) if isinstance(v, str) else v

    @field_validator("started_at", "finished_at")
    @classmethod
    def _utc(cls, v):
        return _as_utc(v)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: Literal["ok", "unavailable"]
    latest_run_at: datetime | None
    risk_model_loaded: bool


class CDMEvent(BaseModel):
    """One close-approach event: its Conjunction Data Messages, any order.

    Each CDM is an object with at least ``time_to_tca`` (days) and ``risk`` (log10 Pc);
    any other field from the ESA CDM format (miss_distance, c_sigma_t, c_object_type,
    ...) is used when present. Only CDMs issued at least 2 days before TCA are used.
    """
    event_id: int
    cdms: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("cdms")
    @classmethod
    def _required_fields(cls, cdms):
        for i, cdm in enumerate(cdms):
            missing = {"time_to_tca", "risk"} - cdm.keys()
            if missing:
                raise ValueError(f"CDM {i} is missing {sorted(missing)}")
        return cdms


class RiskRequest(BaseModel):
    events: list[CDMEvent] = Field(min_length=1, max_length=1000)


class RiskForecast(BaseModel):
    event_id: int
    cdms_used: int
    latest_risk: float = Field(description="log10 Pc of the most recent usable CDM")
    predicted_final_risk: float = Field(description="Forecast log10 Pc of the final CDM")
    high_risk: bool = Field(description="Forecast at or above the 1e-6 notification threshold")
    flagged_by: Literal["latest_risk", "late_riser_model", "none"]
    late_riser_probability: float | None


class RiskResponse(BaseModel):
    model: str
    forecasts: list[RiskForecast]


# ---------------------------------------------------------------- application

def get_session(request: Request):
    with Session(request.app.state.engine) as session:
        yield session


# Module level on purpose: with postponed annotations FastAPI resolves type hints in
# module scope, so an alias defined inside create_app would be read as a query parameter.
SessionDep = Annotated[Session, Depends(get_session)]


def create_app(engine: Engine | None = None, risk_model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or make_engine()
        app.state.risk_model = _load_risk_model(
            risk_model_path or Path(os.environ.get("ORBITGUARD_RISK_MODEL", DEFAULT_RISK_MODEL_PATH)))
        yield
        app.state.engine.dispose()

    app = FastAPI(
        title="OrbitGuard API",
        version=__version__,
        description="Orbital conjunction screening results with probability of collision, "
                    "and final-risk forecasts for conjunction data messages.",
        lifespan=lifespan,
    )
    # Element sets for a full catalogue are ~3 MB of JSON; compressed they are ~1 MB.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.get("/health", response_model=Health, tags=["service"])
    def health(request: Request, session: SessionDep):
        try:
            session.execute(text("SELECT 1"))
            run = latest_run(session)
            db_ok = True
        except Exception:  # noqa: BLE001 - report, do not crash the health check
            log.exception("Database health check failed")
            run, db_ok = None, False
        model_loaded = request.app.state.risk_model is not None
        return Health(
            status="ok" if db_ok else "degraded", version=__version__,
            database="ok" if db_ok else "unavailable",
            latest_run_at=_as_utc(run.generated_at) if run else None,
            risk_model_loaded=model_loaded,
        )

    @app.get("/runs", response_model=list[RunOut], tags=["screening"])
    def runs(session: SessionDep, limit: Annotated[int, Query(ge=1, le=100)] = 20):
        return list_runs(session, limit)

    @app.get("/runs/latest", response_model=RunOut, tags=["screening"])
    def runs_latest(session: SessionDep):
        run = latest_run(session)
        if run is None:
            raise HTTPException(404, "No screening runs stored yet")
        return run

    @app.get("/runs/{run_id}", response_model=RunOut, tags=["screening"])
    def run_by_id(run_id: int, session: SessionDep):
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, f"Run {run_id} not found")
        return run

    @app.get("/runs/{run_id}/elements", response_model=RunElementsOut, tags=["screening"])
    def run_elements(run_id: int, session: SessionDep):
        """The element sets (two-line elements) the run screened, for drawing its objects.

        Kept only for recent runs; older runs return 404.
        """
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(404, f"Run {run_id} not found")
        objects = get_run_elements(session, run_id)
        if objects is None:
            raise HTTPException(404, f"Element sets are not kept for run {run_id}")
        return RunElementsOut(
            run_id=run_id, generated_at=_as_utc(run.generated_at), count=len(objects),
            objects=[ElementSetOut(population=population(o["name"]), **o) for o in objects])

    @app.get("/conjunctions", response_model=ConjunctionPage, tags=["screening"])
    def conjunctions(
        session: SessionDep,
        run_id: Annotated[int | None, Query(description="Default: latest run")] = None,
        min_pc: Annotated[float | None, Query(ge=0, le=1)] = None,
        max_miss_km: Annotated[float | None, Query(gt=0)] = None,
        norad_id: Annotated[int | None, Query(description="Either object of the pair")] = None,
        include_co_orbital: bool = False,
        tca_from: datetime | None = None,
        tca_to: datetime | None = None,
        order_by: Literal["pc", "miss", "tca"] = "pc",
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        if run_id is not None and session.get(Run, run_id) is None:
            raise HTTPException(404, f"Run {run_id} not found")
        q = ConjunctionQuery(run_id=run_id, min_pc=min_pc, max_miss_km=max_miss_km,
                             norad_id=norad_id, include_co_orbital=include_co_orbital,
                             tca_from=tca_from, tca_to=tca_to, order_by=order_by,
                             limit=limit, offset=offset)
        total, rows = query_conjunctions(session, q)
        return ConjunctionPage(total=total, limit=limit, offset=offset,
                               items=[ConjunctionOut.model_validate(r) for r in rows])

    @app.get("/conjunctions/{conjunction_id}", response_model=ConjunctionOut, tags=["screening"])
    def conjunction(conjunction_id: int, session: SessionDep):
        row = session.get(ConjunctionRow, conjunction_id)
        if row is None:
            raise HTTPException(404, f"Conjunction {conjunction_id} not found")
        return row

    @app.get("/objects/{norad_id}/conjunctions", response_model=ConjunctionPage, tags=["screening"])
    def object_conjunctions(
        norad_id: int, session: SessionDep,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        total, rows = query_conjunctions(session, ConjunctionQuery(
            norad_id=norad_id, include_co_orbital=True, limit=limit, offset=offset))
        return ConjunctionPage(total=total, limit=limit, offset=offset,
                               items=[ConjunctionOut.model_validate(r) for r in rows])

    @app.get("/pipeline/runs", response_model=list[PipelineRunOut], tags=["monitoring"])
    def pipeline_runs(
        session: SessionDep,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        status: Literal["success", "failed_validation", "error"] | None = None,
    ):
        """Recent pipeline executions with validation reports, metrics and drift alerts."""
        return list_pipeline_runs(session, limit=limit, status=status)

    @app.post("/risk/forecast", response_model=RiskResponse, tags=["risk model"])
    def risk_forecast(body: RiskRequest, request: Request):
        model = request.app.state.risk_model
        if model is None:
            raise HTTPException(503, "Risk model not loaded; train it with `orbitguard risk train`")
        return forecast_events(model, body.events)

    # The dashboard: static files at the root. Mounted last, so every API route above wins.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="dashboard")
    return app


def _load_risk_model(path: Path):
    if not path.exists():
        log.warning("No risk model at %s; /risk/forecast will return 503", path)
        return None
    try:
        from orbitguard.cdm.model import RiskModel
    except ImportError:
        log.warning("Risk model found at %s but the [ml] extra is not installed", path)
        return None
    try:
        return RiskModel.load(path)
    except Exception:  # noqa: BLE001 - a bad model file must not stop the screening API
        # Usually an environment mismatch: the pickle references a library missing here.
        log.exception("Could not load risk model %s; /risk/forecast will return 503", path)
        return None


def forecast_events(model, events: list[CDMEvent]) -> RiskResponse:
    """Build features from the submitted CDMs and apply the Phase 3 model."""
    import numpy as np
    import pandas as pd

    from orbitguard.cdm.data import HIGH_RISK, INPUT_CUTOFF_DAYS, sort_cdms
    from orbitguard.cdm.features import CATEGORICAL, TREND_COLUMNS, build_features

    rows = [{**cdm, "event_id": e.event_id} for e in events for cdm in e.cdms]
    cdms = pd.DataFrame(rows)
    for col in set(TREND_COLUMNS) | set(CATEGORICAL):
        if col not in cdms:
            cdms[col] = np.nan
    usable = cdms[cdms.time_to_tca >= INPUT_CUTOFF_DAYS]
    unusable = sorted(set(cdms.event_id) - set(usable.event_id))
    if unusable:
        raise HTTPException(422, f"Events {unusable} have no CDM issued at least "
                                 f"{INPUT_CUTOFF_DAYS:g} days before TCA")

    feats = build_features(sort_cdms(usable))
    # The model expects exactly its training columns; absent fields become missing values.
    feats = feats.reindex(columns=model.regressor.feature_name_)
    for col in CATEGORICAL:
        feats[col] = feats[col].astype("object")
    pred = model.predict(feats)
    last = feats["last_risk"].to_numpy(dtype=float)
    flagged = last >= HIGH_RISK
    prob = np.full(len(feats), np.nan)
    if (~flagged).any():
        from orbitguard.cdm.model import align_categories
        X = feats[~flagged].copy()
        align_categories(X, categories=model.categories)
        prob[~flagged] = model.late_riser_probability(X)
    counts = usable.groupby("event_id").size()

    forecasts = []
    for i, event_id in enumerate(feats.index):
        p = float(pred.iloc[i])
        high = p >= HIGH_RISK
        forecasts.append(RiskForecast(
            event_id=int(event_id), cdms_used=int(counts.loc[event_id]),
            latest_risk=float(last[i]), predicted_final_risk=p, high_risk=high,
            flagged_by="latest_risk" if flagged[i] else ("late_riser_model" if high else "none"),
            late_riser_probability=None if flagged[i] else float(prob[i]),
        ))
    return RiskResponse(
        model=f"latest risk + learned corrections (late-riser threshold "
              f"{model.late_riser_threshold:g})",
        forecasts=forecasts,
    )


app = create_app()
