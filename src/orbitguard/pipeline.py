"""The end-to-end pipeline: fetch -> validate -> recalibrate -> screen -> store -> monitor.

``orbitguard pipeline run`` executes it once; a scheduler runs it repeatedly. Every
execution is recorded in the ``pipeline_runs`` table with its status, validation
reports, metrics and drift alerts, whether it succeeded or not.

Drift monitoring compares each run's metrics with recent successful runs of the same
groups using a robust z-score (median and MAD), so one odd run does not distort the
baseline. An alert does not stop the pipeline: it is a signal for a human to look.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Callable

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orbitguard.calibration import MAX_SNAPSHOTS, calibrate_from_cache
from orbitguard.covariance import DEFAULT_MODEL, load_covariance
from orbitguard.ingest import DEFAULT_CACHE_DIR, fetch_group, prune_snapshots, validate_records
from orbitguard.propagation import build_catalog, element_sets
from orbitguard.screening import screen
from orbitguard.store import (
    PipelineRun, compact, list_pipeline_runs, prune_conjunctions, save_pipeline_run, save_result,
    save_run_elements,
)
from orbitguard.validation import validate_latest

log = logging.getLogger(__name__)

HIGH_PC = 1e-4
# Metrics watched for drift. Ratios rather than raw counts where the catalog size matters.
DRIFT_METRICS = ("objects_screened", "conjunctions_per_1000_objects", "high_risk_events",
                 "median_element_age_days")


@dataclass
class PipelineConfig:
    groups: list[str]
    hours: float = 24.0
    step_s: float = 10.0
    threshold_km: float = 5.0
    hbr_m: float = 10.0
    cache_dir: Path = DEFAULT_CACHE_DIR
    covariance_path: Path = Path("data/models/covariance.json")
    results_dir: Path = Path("data/processed")
    calibrate: bool = True
    refresh: bool = False
    drift_window: int = 10        # recent successful runs used as the baseline
    drift_min_history: int = 3    # fewer runs than this: no drift verdict yet
    drift_z: float = 3.0          # robust z-score that raises an alert
    # Retention, so an unattended pipeline does not fill its disk or cache. None keeps all.
    keep_runs: int | None = 20                    # runs that keep their conjunctions (~10 MB each)
    keep_snapshots: int | None = MAX_SNAPSHOTS    # per group; all that calibration reads


@dataclass
class DriftAlert:
    metric: str
    value: float
    baseline_median: float
    robust_z: float

    def to_dict(self) -> dict:
        return {"metric": self.metric, "value": self.value,
                "baseline_median": self.baseline_median, "robust_z": round(self.robust_z, 2)}


# Metrics that count events. Counts of rare events vary by about sqrt(n) on their own
# (Poisson noise): 4 high-risk events after 7, 7 and 7 is ordinary, not a 9-sigma jump.
COUNT_METRICS = ("objects_screened", "high_risk_events")


def detect_drift(current: dict, history: list[dict], z_threshold: float = 3.0,
                 min_history: int = 3, metrics=DRIFT_METRICS,
                 count_metrics=COUNT_METRICS) -> list[DriftAlert]:
    """Robust z-score of each metric against its recent history (median / MAD)."""
    alerts = []
    for m in metrics:
        past = [h[m] for h in history if h.get(m) is not None]
        value = current.get(m)
        if value is None or len(past) < min_history:
            continue
        med = median(past)
        mad = median(abs(x - med) for x in past)
        # A perfectly stable history has MAD 0; floor the scale so tiny changes do not alert,
        # and never below Poisson noise for counts.
        noise = math.sqrt(max(med, 1.0)) if m in count_metrics else 0.0
        scale = max(1.4826 * mad, 0.05 * abs(med), noise, 1e-9)
        z = (value - med) / scale
        if abs(z) > z_threshold:
            alerts.append(DriftAlert(m, float(value), float(med), float(z)))
    return alerts


def _load_validated(cache_dir: Path, reports) -> list[dict]:
    """Records of the validated snapshots, de-duplicated by NORAD ID across groups."""
    by_id: dict[int, dict] = {}
    for report in reports:
        records = json.loads((cache_dir / report.snapshot).read_text())
        for rec in validate_records(records):
            by_id[int(rec["NORAD_CAT_ID"])] = rec
    return list(by_id.values())


@dataclass
class PipelineOutcome:
    status: str
    pipeline_run_id: int
    run_id: int | None = None
    metrics: dict = field(default_factory=dict)
    drift_alerts: list[dict] = field(default_factory=list)
    validation: list[dict] = field(default_factory=list)
    message: str = ""


def run_pipeline(cfg: PipelineConfig, engine: Engine, now: datetime | None = None,
                 fetch: Callable = fetch_group) -> PipelineOutcome:
    started = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    groups_key = ",".join(cfg.groups)
    reports: list[dict] = []

    def finish(status: str, **fields) -> PipelineOutcome:
        with Session(engine) as session:
            rec = save_pipeline_run(
                session, started_at=started, finished_at=datetime.now(timezone.utc),
                status=status, groups=groups_key, validation=reports, **fields)
            session.commit()
            return PipelineOutcome(status=status, pipeline_run_id=rec.id, run_id=rec.run_id,
                                   metrics=json.loads(rec.metrics),
                                   drift_alerts=json.loads(rec.drift_alerts),
                                   validation=reports, message=rec.message)

    try:
        # 1. Fetch (the 2-hour cache keeps us within CelesTrak's rate limits).
        for group in cfg.groups:
            fetch(group, cache_dir=cfg.cache_dir, refresh=cfg.refresh, now=started)

        # 2. Validate every group before anything is computed from it.
        validation = [validate_latest(g, cache_dir=cfg.cache_dir) for g in cfg.groups]
        reports.extend(r.to_dict() for r in validation)
        errors = [f"{r.group}: {c.name} ({c.detail})" for r in validation for c in r.errors]
        if errors:
            log.error("Validation failed: %s", "; ".join(errors))
            return finish("failed_validation", message="; ".join(errors))

        metrics: dict = {"validation_warnings": sum(len(r.warnings) for r in validation)}

        # 3. Recalibrate uncertainty from recent cached snapshot pairs (if too few, the
        #    previous model is kept).
        if cfg.calibrate:
            fitted = calibrate_from_cache(cfg.cache_dir, today=started)
            if fitted is not None:
                model, _, n_pairs = fitted
                model.save(cfg.covariance_path)
                metrics["calibration_pairs"] = n_pairs
                metrics["calibrated_sigma_t_2d_km"] = {
                    pop: round(float(m.sigma_rtn(2.0)[1]), 3) for pop, m in model.models.items()}

        # 4. Screen exactly the snapshots that passed validation (never re-download here:
        #    a fresh download could differ from what was checked).
        records = _load_validated(cfg.cache_dir, validation)
        catalog = build_catalog(records, at=started)
        cov = load_covariance(cfg.covariance_path) if cfg.covariance_path.exists() else DEFAULT_MODEL
        t0 = time.perf_counter()
        conjunctions = screen(catalog, started, hours=cfg.hours, step_s=cfg.step_s,
                              threshold_km=cfg.threshold_km, cov_model=cov,
                              hbr_km=cfg.hbr_m / 1000.0)
        screening_s = time.perf_counter() - t0

        # 5. Store.
        result = {
            "generated_at": started.isoformat(), "groups": cfg.groups,
            "objects_screened": len(catalog), "window_hours": cfg.hours,
            "threshold_km": cfg.threshold_km, "hard_body_radius_m": cfg.hbr_m,
            "covariance_model": {"source": cov.source},
            "conjunctions": [c.to_dict() for c in conjunctions],
        }
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        (cfg.results_dir / f"conjunctions_{started:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(result))
        with Session(engine) as session:
            run_id = save_result(session, result).id
            save_run_elements(session, run_id, element_sets(catalog))  # for the dashboard
            pruned = prune_conjunctions(session, cfg.keep_runs) if cfg.keep_runs else 0
            session.commit()
        if pruned:
            compact(engine)
        if cfg.keep_snapshots:
            for group in cfg.groups:
                prune_snapshots(cfg.cache_dir, group, cfg.keep_snapshots)

        # 6. Metrics and drift.
        events = [c for c in conjunctions if not c.co_orbital]
        ages = sorted((started - e).total_seconds() / 86400.0 for e in catalog.epochs)
        metrics.update({
            "objects_screened": len(catalog),
            "conjunctions": len(events),
            "co_orbital_pairs": len(conjunctions) - len(events),
            "conjunctions_per_1000_objects": round(1000 * len(events) / max(len(catalog), 1), 3),
            "high_risk_events": sum(c.pc is not None and c.pc >= HIGH_PC for c in events),
            "max_pc": max((c.pc for c in events if c.pc is not None), default=0.0),
            "median_element_age_days": round(median(ages), 3) if ages else None,
            "screening_seconds": round(screening_s, 1),
        })
        with Session(engine) as session:
            history = [json.loads(p.metrics) for p in list_pipeline_runs(
                session, limit=cfg.drift_window, status="success", groups=groups_key)]
        alerts = [a.to_dict() for a in detect_drift(metrics, history, cfg.drift_z,
                                                    cfg.drift_min_history)]
        for a in alerts:
            log.warning("Drift: %s = %s (baseline median %s, robust z %s)",
                        a["metric"], a["value"], a["baseline_median"], a["robust_z"])
        note = "" if len(history) >= cfg.drift_min_history else (
            f"drift check needs {cfg.drift_min_history} earlier successful runs; have {len(history)}")
        return finish("success", run_id=run_id, metrics=metrics, drift_alerts=alerts, message=note)

    except Exception as exc:  # noqa: BLE001 - record every failure, then report it
        log.exception("Pipeline failed")
        return finish("error", message=f"{type(exc).__name__}: {exc}")


MIN_INTERVAL_S = 2 * 3600  # CelesTrak publishes about every 2 hours; faster gains nothing


def run_on_schedule(run_once: Callable[[], PipelineOutcome], every_s: float,
                    stop: "threading.Event | None" = None, max_runs: int | None = None
                    ) -> list[PipelineOutcome]:
    """Run ``run_once`` every ``every_s`` seconds until ``stop`` is set.

    A failed run does not stop the schedule: it is recorded and the next run happens on
    time. The interval is measured start to start, so slow runs do not drift the clock.
    """
    import threading

    if every_s < MIN_INTERVAL_S:
        raise ValueError(f"Interval must be at least {MIN_INTERVAL_S / 3600:g} h "
                         "(CelesTrak updates about every 2 hours)")
    stop = stop or threading.Event()
    outcomes = []
    while not stop.is_set():
        started = time.monotonic()
        outcome = run_once()
        outcomes.append(outcome)
        log.info("Scheduled run %s: %s", outcome.pipeline_run_id, outcome.status)
        if max_runs is not None and len(outcomes) >= max_runs:
            break
        # Wait for the next slot; returns early as soon as a stop is requested.
        stop.wait(max(0.0, every_s - (time.monotonic() - started)))
    return outcomes
