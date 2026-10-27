"""Calibrate the covariance model from successive element sets of the same objects.

For every object present in an older and a newer snapshot, propagate the older
element set to the newer one's epoch and compare it with the newer element set's
own state there. The newer element set is fitted to fresher tracking data, so the
difference approximates the prediction error of the older one after ``age`` days.

Errors are expressed in the RTN frame of the newer state, binned by age, and a
robust spread (MAD) per bin is fitted with sigma = sigma0 + growth * age.
Manoeuvring satellites produce large outliers; the MAD keeps them from inflating
the fit, and they are counted separately.

Only pairs up to ``MAX_PAIR_AGE_DAYS`` apart are used. Screening propagates element
sets that are about a day old over a 24-hour window, so that is the range the model
has to be right in. Along-track error grows faster than linearly (drag, manoeuvres):
Starlink measured ~30 km at 2 days but ~400 km at 8 days, and a straight line fitted
through both overstated the 2-day error threefold.

The newer element set has its own error, so the measured spread is slightly
conservative (it includes both sets' errors).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec

from orbitguard.covariance import CovarianceModel, PopulationCovariance, population, rtn_basis
from orbitguard.ingest import validate_records
from orbitguard.propagation import propagate_one

log = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826
OUTLIER_SIGMAS = 5.0
MAX_PAIR_AGE_DAYS = 3.0
MAX_SNAPSHOTS = 8       # latest snapshots per group; older ones describe stale behaviour
MIN_RESIDUALS = 50


@dataclass
class Residuals:
    norad_ids: np.ndarray   # (n,)
    names: list[str]
    age_days: np.ndarray    # (n,)
    rtn_km: np.ndarray      # (n, 3) old-prediction minus new state, in RTN


def _epoch(rec: dict) -> datetime:
    return datetime.strptime(rec["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)


def _satrec(rec: dict) -> Satrec | None:
    sat = Satrec()
    try:
        omm.initialize(sat, rec)
    except (KeyError, ValueError):
        return None
    return None if sat.error else sat


def compute_residuals(old: list[dict], new: list[dict],
                      min_age_days: float = 0.25,
                      max_age_days: float = MAX_PAIR_AGE_DAYS) -> Residuals:
    """RTN prediction errors of ``old`` element sets checked against ``new`` ones."""
    old_by_id = {int(r["NORAD_CAT_ID"]): r for r in old}
    ids, names, ages, rows = [], [], [], []
    for rec in new:
        nid = int(rec["NORAD_CAT_ID"])
        prev = old_by_id.get(nid)
        if prev is None:
            continue
        t_new = _epoch(rec)
        age = (t_new - _epoch(prev)).total_seconds() / 86400.0
        if not (min_age_days <= age <= max_age_days):
            continue
        sat_old, sat_new = _satrec(prev), _satrec(rec)
        if sat_old is None or sat_new is None:
            continue
        try:
            r_pred, _ = propagate_one(sat_old, t_new)
            r_ref, v_ref = propagate_one(sat_new, t_new)
        except ValueError:
            continue  # decayed or otherwise failed propagation
        ids.append(nid)
        names.append(rec["OBJECT_NAME"])
        ages.append(age)
        rows.append(rtn_basis(r_ref, v_ref).T @ (r_pred - r_ref))
    return Residuals(np.array(ids), names, np.array(ages), np.array(rows).reshape(-1, 3))


def robust_sigma(x: np.ndarray) -> float:
    return float(MAD_TO_SIGMA * np.median(np.abs(x - np.median(x))))


def _fit_line_nonneg(age: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Least-squares sigma = a + b*age with a, b >= 0."""
    if len(age) >= 2 and np.ptp(age) > 0:
        b, a = np.polyfit(age, sigma, 1)
        if a >= 0 and b >= 0:
            return float(a), float(b)
        if b < 0:  # no measurable growth: constant spread
            return float(np.mean(sigma)), 0.0
    # Intercept forced to zero (also the fallback for a single age bin).
    return 0.0, float(np.sum(sigma * age) / np.sum(age**2))


def fit_model(res: Residuals, n_bins: int = 6, min_per_bin: int = 50,
              source: str = "calibrated") -> tuple[CovarianceModel, dict]:
    """Fit the linear-growth RTN model; return it with a summary of the fit."""
    if len(res.age_days) < min_per_bin:
        raise ValueError(f"Only {len(res.age_days)} residuals; need at least {min_per_bin}")
    n_bins = max(1, min(n_bins, len(res.age_days) // min_per_bin))
    edges = np.quantile(res.age_days, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, res.age_days, side="right") - 1, 0, n_bins - 1)

    bin_age, bin_sigma = [], []
    for b in range(n_bins):
        sel = which == b
        if sel.sum() < min_per_bin:
            continue
        bin_age.append(float(np.median(res.age_days[sel])))
        bin_sigma.append([robust_sigma(res.rtn_km[sel, k]) for k in range(3)])
    bin_age, bin_sigma = np.array(bin_age), np.array(bin_sigma)

    fits = [_fit_line_nonneg(bin_age, bin_sigma[:, k]) for k in range(3)]
    model = CovarianceModel(
        sigma0_km=tuple(round(a, 4) for a, _ in fits),
        growth_km_per_day=tuple(round(b, 4) for _, b in fits),
        source=source,
    )

    predicted = np.array([model.sigma_rtn(a) for a in res.age_days])
    outlier = np.any(np.abs(res.rtn_km - np.median(res.rtn_km, axis=0)) > OUTLIER_SIGMAS * predicted, axis=1)
    summary = {
        "residuals": int(len(res.age_days)),
        "age_days_range": [float(res.age_days.min()), float(res.age_days.max())],
        "bins": [{"age_days": a, "sigma_rtn_km": s.tolist()} for a, s in zip(bin_age, bin_sigma)],
        "outlier_fraction": float(outlier.mean()),
    }
    return model, summary


def subset(res: Residuals, mask: np.ndarray) -> Residuals:
    return Residuals(res.norad_ids[mask], [n for n, m in zip(res.names, mask) if m],
                     res.age_days[mask], res.rtn_km[mask])


def fit_populations(res: Residuals, source: str, min_samples: int = 300
                    ) -> tuple[PopulationCovariance, dict]:
    """Fit one model per population with enough samples, plus a pooled fallback."""
    default, pooled = fit_model(res, source=f"{source} (pooled)")
    pops = np.array([population(n) for n in res.names])
    models, summaries = {}, {"pooled": pooled}
    for pop in sorted(set(pops)):
        mask = pops == pop
        if mask.sum() < min_samples:
            continue
        models[pop], summaries[pop] = fit_model(subset(res, mask), source=f"{source} ({pop})")
    return PopulationCovariance(models, default, source), summaries


def _fetched_at(path: Path) -> datetime:
    stamp = path.stem.rsplit("_", 1)[1]
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def load_snapshot_pairs(cache_dir: Path, max_snapshots: int = MAX_SNAPSHOTS,
                        max_gap_days: float = MAX_PAIR_AGE_DAYS
                        ) -> list[tuple[list[dict], list[dict]]]:
    """(older, newer) record lists for pairs of each group's latest cached snapshots.

    Pairs fetched more than ``max_gap_days`` apart are skipped before loading: almost
    none of their element sets would pass the age limit in ``compute_residuals``.
    """
    groups: dict[str, list[Path]] = {}
    for path in sorted(cache_dir.glob("gp_*_*.json")):
        group = path.stem[len("gp_"):].rsplit("_", 1)[0]
        groups.setdefault(group, []).append(path)
    pairs = []
    for paths in groups.values():
        paths = paths[-max_snapshots:]  # file names sort by fetch time
        times = [_fetched_at(p) for p in paths]
        snaps: dict[int, list[dict]] = {}

        def load(i: int) -> list[dict]:
            if i not in snaps:
                snaps[i] = validate_records(json.loads(paths[i].read_text()))
            return snaps[i]

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if (times[j] - times[i]).total_seconds() / 86400.0 <= max_gap_days:
                    pairs.append((load(i), load(j)))
    return pairs


def calibrate_from_cache(cache_dir: Path, today: datetime | None = None
                         ) -> tuple[PopulationCovariance, dict, int] | None:
    """Fit per-population models from pairs of recent cached snapshots.

    Returns (model, summaries, number of snapshot pairs), or None when the cache does
    not yet hold enough snapshots close enough together to fit from.
    """
    pairs = load_snapshot_pairs(cache_dir)
    if not pairs:
        return None
    parts = [compute_residuals(old, new) for old, new in pairs]
    if sum(len(p.age_days) for p in parts) < MIN_RESIDUALS:
        return None
    res = Residuals(
        norad_ids=np.concatenate([p.norad_ids for p in parts]),
        names=[n for p in parts for n in p.names],
        age_days=np.concatenate([p.age_days for p in parts]),
        rtn_km=np.concatenate([p.rtn_km for p in parts]),
    )
    today = today or datetime.now(timezone.utc)
    model, summaries = fit_populations(
        res, source=f"calibrated {today:%Y-%m-%d} from {len(res.age_days)} element-set pairs")
    return model, summaries, len(pairs)
