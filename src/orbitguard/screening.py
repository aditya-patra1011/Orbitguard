"""All-vs-all conjunction screening.

Pipeline:
  1. Propagate the whole catalog on a fixed time grid (vectorised SGP4).
  2. At each step, find pairs closer than a screening radius with a KD-tree.
     The radius is padded by v_rel_max * step / 2 so a true close approach that
     happens between two grid points cannot be missed.
  3. Group hits into encounters (runs of consecutive steps per pair) and refine
     the time of closest approach (TCA) with a bounded 1-D minimisation.
  4. Score each encounter with a probability of collision (Pc) from the
     covariance model and the combined hard-body radius.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from orbitguard.covariance import DEFAULT_MODEL, CovarianceModel
from orbitguard.propagation import Catalog, _julian, propagate_one
from orbitguard.risk import collision_probability

log = logging.getLogger(__name__)

# Head-on collision of two objects in LEO: about 2 x 7.8 km/s.
MAX_REL_SPEED_KM_S = 15.8
# Slack for curvature error in the linear pre-filter; generous on purpose.
LINEAR_FILTER_MARGIN_KM = 2.0
# Below this relative speed two objects are flying together (same plane, formation,
# docked) rather than crossing; such pairs are reported separately.
CO_ORBITAL_SPEED_KM_S = 0.1
# Combined hard-body radius when object sizes are unknown (10 m).
DEFAULT_HBR_KM = 0.01


@dataclass
class Conjunction:
    norad_id_1: int
    name_1: str
    norad_id_2: int
    name_2: str
    tca: datetime
    miss_distance_km: float
    relative_speed_km_s: float
    co_orbital: bool
    # None for co-orbital pairs: the short-encounter Pc model does not apply to them.
    pc: float | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tca"] = self.tca.isoformat()
        return d


def _linear_min_distance(dr: np.ndarray, dv: np.ndarray, max_dt: float) -> np.ndarray:
    """Closest approach assuming straight-line relative motion within +/- max_dt seconds.

    Over a few seconds both objects feel almost the same gravity, so relative
    motion is very nearly linear; this is a cheap, vectorised pre-filter.
    """
    dv2 = np.einsum("ij,ij->i", dv, dv)
    t = -np.einsum("ij,ij->i", dr, dv) / np.where(dv2 > 0, dv2, 1.0)
    t = np.clip(t, -max_dt, max_dt)
    return np.linalg.norm(dr + dv * t[:, None], axis=1)


def _collect_hits(catalog: Catalog, start: datetime, offsets_s: np.ndarray, radius_km: float,
                  keep_km: float, step_s: float, chunk_steps: int) -> np.ndarray:
    """Return rows (step, i, j, est_miss_km) for pairs within radius whose linearly
    extrapolated miss distance is below ``keep_km``."""
    hits = []
    n_coarse = 0
    for c0 in range(0, len(offsets_s), chunk_steps):
        r, v, ok = catalog.propagate(start, offsets_s[c0:c0 + chunk_steps])
        for k in range(r.shape[1]):
            valid = np.flatnonzero(ok[:, k])
            pts = r[valid, k]
            pairs = cKDTree(pts).query_pairs(radius_km, output_type="ndarray")
            if len(pairs) == 0:
                continue
            n_coarse += len(pairs)
            gi = valid[pairs]
            dr = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dv = v[gi[:, 0], k] - v[gi[:, 1], k]
            est = _linear_min_distance(dr, dv, step_s)
            keep = est <= keep_km
            if not keep.any():
                continue
            step = np.full(keep.sum(), c0 + k)
            hits.append(np.column_stack([step, gi[keep, 0], gi[keep, 1], est[keep]]))
    log.info("%d coarse pairs, %d kept after linear filter", n_coarse, sum(map(len, hits)))
    return np.concatenate(hits) if hits else np.empty((0, 4))


def _encounters(hits: np.ndarray):
    """Yield (i, j, steps, distances) for each run of consecutive steps of one pair."""
    if len(hits) == 0:
        return
    order = np.lexsort((hits[:, 0], hits[:, 2], hits[:, 1]))
    hits = hits[order]
    i, j, step = hits[:, 1].astype(int), hits[:, 2].astype(int), hits[:, 0].astype(int)
    new_run = np.ones(len(hits), dtype=bool)
    new_run[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1]) | (step[1:] != step[:-1] + 1)
    bounds = np.append(np.flatnonzero(new_run), len(hits))
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield i[a], j[a], step[a:b], hits[a:b, 3]


def _refine_tca(sat1, sat2, jd: float, fr: float, lo_s: float, hi_s: float) -> tuple[float, float]:
    """Minimise separation over [lo_s, hi_s] seconds after (jd, fr). Returns (t_s, miss_km)."""

    def sep(t_s: float) -> float:
        f = fr + t_s / 86400.0
        _, r1, _ = sat1.sgp4(jd, f)
        _, r2, _ = sat2.sgp4(jd, f)
        return float(np.linalg.norm(np.subtract(r1, r2)))

    res = minimize_scalar(sep, bounds=(lo_s, hi_s), method="bounded", options={"xatol": 1e-3})
    return float(res.x), float(res.fun)


def screen(
    catalog: Catalog,
    start: datetime,
    hours: float = 24.0,
    step_s: float = 10.0,
    threshold_km: float = 5.0,
    chunk_steps: int = 60,
    cov_model: CovarianceModel = DEFAULT_MODEL,
    hbr_km: float = DEFAULT_HBR_KM,
) -> list[Conjunction]:
    """Find all close approaches below ``threshold_km`` in [start, start + hours],
    sorted by miss distance, each with a probability of collision."""
    offsets = np.arange(0.0, hours * 3600.0 + step_s, step_s)
    radius = threshold_km + MAX_REL_SPEED_KM_S * step_s / 2
    log.info("Screening %d objects over %d steps (radius %.1f km)", len(catalog), len(offsets), radius)

    hits = _collect_hits(catalog, start, offsets, radius, threshold_km + LINEAR_FILTER_MARGIN_KM,
                         step_s, chunk_steps)

    jd, fr = _julian(start)
    last_step = len(offsets) - 1
    results = []
    for i, j, steps, dists in _encounters(hits):
        k = steps[np.argmin(dists)]
        lo, hi = offsets[max(k - 1, 0)], offsets[min(k + 1, last_step)]
        t_s, miss = _refine_tca(catalog.satrecs[i], catalog.satrecs[j], jd, fr, lo, hi)
        if miss > threshold_km:
            continue
        tca = start + timedelta(seconds=t_s)
        r1, v1 = propagate_one(catalog.satrecs[i], tca)
        r2, v2 = propagate_one(catalog.satrecs[j], tca)
        rel_speed = float(np.linalg.norm(v1 - v2))
        co_orbital = rel_speed < CO_ORBITAL_SPEED_KM_S
        # A minimum on the window edge is an approach that started before/after the window.
        at_edge = t_s <= offsets[0] + 1e-2 or t_s >= offsets[-1] - 1e-2
        if at_edge and not co_orbital:
            continue
        pc = None
        if not co_orbital:
            age1 = (tca - catalog.epochs[i]).total_seconds() / 86400.0
            age2 = (tca - catalog.epochs[j]).total_seconds() / 86400.0
            cov = cov_model.covariance(r1, v1, age1) + cov_model.covariance(r2, v2, age2)
            pc = collision_probability(r1 - r2, v1 - v2, cov, hbr_km)
        results.append(Conjunction(
            norad_id_1=int(catalog.norad_ids[i]), name_1=catalog.names[i],
            norad_id_2=int(catalog.norad_ids[j]), name_2=catalog.names[j],
            tca=tca, miss_distance_km=miss,
            relative_speed_km_s=rel_speed, co_orbital=co_orbital, pc=pc,
        ))
    results.sort(key=lambda c: c.miss_distance_km)
    return results
