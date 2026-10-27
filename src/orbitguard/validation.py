"""Data validation for downloaded GP snapshots.

Bad input data is the most common way an automated pipeline produces confidently
wrong output. Every snapshot is checked before it is screened:

- **errors** stop the pipeline (truncated or corrupt downloads, impossible data);
- **warnings** are recorded and reported but do not stop it.

Checks compare the snapshot with itself (schema, duplicates, epochs, physical ranges)
and with the previous snapshot of the same group (sudden changes in size).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Literal

from orbitguard.ingest import DEFAULT_CACHE_DIR, REQUIRED_FIELDS

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Thresholds:
    max_invalid_fraction: float = 0.01      # records failing schema / range checks
    max_count_change: float = 0.20          # vs. previous snapshot of the group
    max_median_age_days: float = 3.0        # typical element-set age
    max_stale_fraction: float = 0.20        # older than stale_days
    stale_days: float = 14.0
    future_tolerance_days: float = 1.0      # epochs in the future mean bad clocks or data
    mean_motion_range: tuple[float, float] = (0.05, 18.0)   # rev/day, GEO..lowest LEO


@dataclass
class Check:
    name: str
    passed: bool
    severity: Severity
    detail: str


@dataclass
class Report:
    group: str
    snapshot: str
    n_records: int
    checks: list[Check] = field(default_factory=list)

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _epoch(rec: dict) -> datetime | None:
    try:
        return datetime.strptime(rec["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None


def _record_problem(rec: dict, t: Thresholds) -> str | None:
    """Why a record is unusable, or None if it is fine."""
    missing = [f for f in REQUIRED_FIELDS if rec.get(f) in (None, "")]
    if missing:
        return "missing " + ",".join(missing)
    if _epoch(rec) is None:
        return "bad epoch"
    try:
        e, n = float(rec["ECCENTRICITY"]), float(rec["MEAN_MOTION"])
    except (TypeError, ValueError):
        return "non-numeric elements"
    if not 0.0 <= e < 1.0:
        return "eccentricity out of range"
    if not t.mean_motion_range[0] <= n <= t.mean_motion_range[1]:
        return "mean motion out of range"
    return None


def validate_snapshot(records: list[dict], group: str, now: datetime,
                      previous: list[dict] | None = None, snapshot: str = "",
                      thresholds: Thresholds = Thresholds()) -> Report:
    t = thresholds
    report = Report(group=group, snapshot=snapshot, n_records=len(records))
    add = lambda name, passed, severity, detail: report.checks.append(  # noqa: E731
        Check(name, bool(passed), severity, detail))

    add("not_empty", len(records) > 0, "error", f"{len(records)} records")
    if not records:
        return report

    problems = [p for p in (_record_problem(r, t) for r in records) if p]
    frac = len(problems) / len(records)
    add("invalid_records", frac <= t.max_invalid_fraction, "error",
        f"{len(problems)} invalid ({frac:.1%}; limit {t.max_invalid_fraction:.0%})"
        + (f"; e.g. {problems[0]}" if problems else ""))

    ids = [r.get("NORAD_CAT_ID") for r in records]
    dupes = len(ids) - len(set(ids))
    add("unique_ids", dupes == 0, "warning", f"{dupes} duplicate NORAD IDs")

    epochs = [e for e in map(_epoch, records) if e is not None]
    if epochs:
        ages = [(now - e).total_seconds() / 86400.0 for e in epochs]
        future = sum(a < -t.future_tolerance_days for a in ages)
        add("no_future_epochs", future == 0, "error",
            f"{future} epochs more than {t.future_tolerance_days:g} day ahead of now")
        med = median(ages)
        add("fresh_data", med <= t.max_median_age_days, "warning",
            f"median element-set age {med:.2f} days (limit {t.max_median_age_days:g})")
        stale = sum(a > t.stale_days for a in ages) / len(ages)
        add("few_stale_elements", stale <= t.max_stale_fraction, "warning",
            f"{stale:.1%} older than {t.stale_days:g} days (limit {t.max_stale_fraction:.0%})")

    if previous:
        change = (len(records) - len(previous)) / len(previous)
        add("stable_count", abs(change) <= t.max_count_change, "error",
            f"{len(previous)} -> {len(records)} records ({change:+.1%}; "
            f"limit +-{t.max_count_change:.0%})")
    return report


def snapshots(cache_dir: Path, group: str) -> list[Path]:
    return sorted(cache_dir.glob(f"gp_{group}_*.json"))


def validate_latest(group: str, cache_dir: Path = DEFAULT_CACHE_DIR, now: datetime | None = None,
                    thresholds: Thresholds = Thresholds()) -> Report:
    """Validate the newest cached snapshot of a group against the one before it.

    ``now`` defaults to the snapshot's own download time, so old snapshots are judged
    by the conditions when they were fetched.
    """
    paths = snapshots(cache_dir, group)
    if not paths:
        return validate_snapshot([], group, now or datetime.now(timezone.utc))
    latest = paths[-1]
    fetched = datetime.strptime(latest.stem.rsplit("_", 1)[1], "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc)
    previous = json.loads(paths[-2].read_text()) if len(paths) > 1 else None
    return validate_snapshot(json.loads(latest.read_text()), group, now or fetched,
                             previous=previous, snapshot=latest.name, thresholds=thresholds)
