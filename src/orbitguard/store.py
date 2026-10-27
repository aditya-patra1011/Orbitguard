"""Persist screening runs and their conjunctions in a SQL database.

SQLAlchemy keeps the code database-agnostic: SQLite (a single file) for local use,
PostgreSQL in Docker. The URL comes from ``ORBITGUARD_DB_URL`` or defaults to
``sqlite:///data/orbitguard.db``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, func, insert,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
from sqlalchemy.pool import StaticPool

DEFAULT_DB_URL = "sqlite:///data/orbitguard.db"


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    groups: Mapped[str] = mapped_column(String(500))
    objects_screened: Mapped[int] = mapped_column(Integer)
    window_hours: Mapped[float] = mapped_column(Float)
    threshold_km: Mapped[float] = mapped_column(Float)
    hard_body_radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    covariance_source: Mapped[str | None] = mapped_column(String(500), nullable=True)
    n_conjunctions: Mapped[int] = mapped_column(Integer)

    conjunctions: Mapped[list["ConjunctionRow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan")


class ConjunctionRow(Base):
    __tablename__ = "conjunctions"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    norad_id_1: Mapped[int] = mapped_column(Integer, index=True)
    name_1: Mapped[str] = mapped_column(String(100))
    norad_id_2: Mapped[int] = mapped_column(Integer, index=True)
    name_2: Mapped[str] = mapped_column(String(100))
    tca: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    miss_distance_km: Mapped[float] = mapped_column(Float)
    relative_speed_km_s: Mapped[float] = mapped_column(Float)
    co_orbital: Mapped[bool] = mapped_column(Boolean)
    pc: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)

    run: Mapped[Run] = relationship(back_populates="conjunctions")


class PipelineRun(Base):
    """One execution of `orbitguard pipeline run`, successful or not."""
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(30), index=True)   # success | failed_validation | error
    groups: Mapped[str] = mapped_column(String(500))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"),
                                               nullable=True)
    metrics: Mapped[str] = mapped_column(Text, default="{}")      # JSON
    validation: Mapped[str] = mapped_column(Text, default="[]")   # JSON, one report per group
    drift_alerts: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    message: Mapped[str] = mapped_column(Text, default="")


def save_pipeline_run(session: Session, **fields) -> PipelineRun:
    for key in ("metrics", "validation", "drift_alerts"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    run = PipelineRun(**fields)
    session.add(run)
    session.flush()
    return run


def list_pipeline_runs(session: Session, limit: int = 20,
                       status: str | None = None, groups: str | None = None) -> list[PipelineRun]:
    stmt = select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(PipelineRun.status == status)
    if groups:
        stmt = stmt.where(PipelineRun.groups == groups)
    return list(session.scalars(stmt))


def database_url() -> str:
    return os.environ.get("ORBITGUARD_DB_URL", DEFAULT_DB_URL)


def make_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    kwargs = {}
    if url in ("sqlite://", "sqlite:///:memory:"):
        # One shared in-memory database across threads (used by tests and the API test client).
        kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
    elif url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"connect_args": {"check_same_thread": False}}
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return engine


def _utc(t: datetime) -> datetime:
    """SQLite drops time zones; treat naive values as UTC."""
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _parse_time(value) -> datetime:
    return _utc(value if isinstance(value, datetime) else datetime.fromisoformat(value))


def save_result(session: Session, result: dict) -> Run:
    """Store one screening result in the JSON layout written by ``orbitguard screen``."""
    cov = result.get("covariance_model") or {}
    conjunctions = result.get("conjunctions", [])
    run = Run(
        generated_at=_parse_time(result["generated_at"]),
        groups=",".join(result.get("groups", [])),
        objects_screened=int(result["objects_screened"]),
        window_hours=float(result["window_hours"]),
        threshold_km=float(result["threshold_km"]),
        hard_body_radius_m=result.get("hard_body_radius_m"),
        covariance_source=cov.get("source"),
        n_conjunctions=len(conjunctions),
    )
    session.add(run)
    session.flush()  # assigns run.id
    if conjunctions:
        # Bulk insert: one ORM object per row is ~10x slower for 45k-row runs.
        session.execute(insert(ConjunctionRow), [
            {
                "run_id": run.id,
                "norad_id_1": c["norad_id_1"], "name_1": c["name_1"],
                "norad_id_2": c["norad_id_2"], "name_2": c["name_2"],
                "tca": _parse_time(c["tca"]),
                "miss_distance_km": c["miss_distance_km"],
                "relative_speed_km_s": c["relative_speed_km_s"],
                # Phase 1 results used "persistent" before the co-orbital flag existed.
                "co_orbital": bool(c.get("co_orbital", c.get("persistent", False))),
                "pc": c.get("pc"),
            }
            for c in conjunctions
        ])
    return run


def import_files(session: Session, paths: Iterable[Path]) -> list[Run]:
    """Import result JSON files, skipping runs already stored (same generated_at)."""
    stored = {_utc(t) for t in session.scalars(select(Run.generated_at))}
    runs = []
    for path in sorted(paths):
        result = json.loads(Path(path).read_text())
        if _parse_time(result["generated_at"]) in stored:
            continue
        runs.append(save_result(session, result))
    return runs


def list_runs(session: Session, limit: int = 20) -> list[Run]:
    return list(session.scalars(select(Run).order_by(Run.generated_at.desc()).limit(limit)))


def latest_run(session: Session) -> Run | None:
    return session.scalars(select(Run).order_by(Run.generated_at.desc()).limit(1)).first()


@dataclass
class ConjunctionQuery:
    run_id: int | None = None          # None: the latest run
    min_pc: float | None = None
    max_miss_km: float | None = None
    norad_id: int | None = None        # either object of the pair
    include_co_orbital: bool = False
    tca_from: datetime | None = None
    tca_to: datetime | None = None
    order_by: str = "pc"               # pc | miss | tca
    limit: int = 50
    offset: int = 0


ORDERINGS = {
    "pc": (ConjunctionRow.pc.desc().nulls_last(), ConjunctionRow.miss_distance_km),
    "miss": (ConjunctionRow.miss_distance_km,),
    "tca": (ConjunctionRow.tca,),
}


def query_conjunctions(session: Session, q: ConjunctionQuery) -> tuple[int, list[ConjunctionRow]]:
    """Return (total matching rows, one page of rows)."""
    run_id = q.run_id
    if run_id is None:
        latest = latest_run(session)
        if latest is None:
            return 0, []
        run_id = latest.id
    stmt = select(ConjunctionRow).where(ConjunctionRow.run_id == run_id)
    if q.min_pc is not None:
        stmt = stmt.where(ConjunctionRow.pc >= q.min_pc)
    if q.max_miss_km is not None:
        stmt = stmt.where(ConjunctionRow.miss_distance_km <= q.max_miss_km)
    if q.norad_id is not None:
        stmt = stmt.where((ConjunctionRow.norad_id_1 == q.norad_id) | (ConjunctionRow.norad_id_2 == q.norad_id))
    if not q.include_co_orbital:
        stmt = stmt.where(ConjunctionRow.co_orbital.is_(False))
    if q.tca_from is not None:
        stmt = stmt.where(ConjunctionRow.tca >= _utc(q.tca_from))
    if q.tca_to is not None:
        stmt = stmt.where(ConjunctionRow.tca <= _utc(q.tca_to))
    total = session.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = session.scalars(stmt.order_by(*ORDERINGS[q.order_by]).limit(q.limit).offset(q.offset))
    return int(total), list(rows)
