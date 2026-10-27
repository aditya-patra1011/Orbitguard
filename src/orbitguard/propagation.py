"""Build an SGP4 catalog from GP records and propagate it in bulk."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sgp4 import exporter, omm
from sgp4.api import Satrec, SatrecArray, jday

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6378.137
MU_EARTH = 398600.4418  # km^3 / s^2


@dataclass
class Catalog:
    norad_ids: np.ndarray
    names: list[str]
    satrecs: list[Satrec]
    epochs: list[datetime]

    def __post_init__(self):
        self._array = SatrecArray(self.satrecs)

    def __len__(self) -> int:
        return len(self.satrecs)

    def propagate(self, start: datetime, offsets_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return TEME positions (km) and velocities (km/s), both (n_sat, n_time, 3),
        and a validity mask (n_sat, n_time)."""
        jd, fr = _julian(start)
        fr_arr = fr + np.asarray(offsets_s, dtype=float) / 86400.0
        jd_arr = np.full_like(fr_arr, jd)
        err, r, v = self._array.sgp4(jd_arr, fr_arr)
        return r, v, err == 0


def _julian(t: datetime) -> tuple[float, float]:
    t = t.astimezone(timezone.utc)
    return jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)


def propagate_one(sat: Satrec, t: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Position (km) and velocity (km/s) of a single satellite in TEME."""
    err, r, v = sat.sgp4(*_julian(t))
    if err:
        raise ValueError(f"SGP4 error {err} for satellite {sat.satnum}")
    return np.array(r), np.array(v)


def element_sets(catalog: Catalog) -> list[dict]:
    """The catalog's satellites as two-line elements, for storing with a run.

    Exported from the initialised satellites themselves, so a client propagating these
    lines (e.g. satellite.js) sees what screening saw: within ~10 m over a day.
    """
    out = []
    for nid, name, sat in zip(catalog.norad_ids, catalog.names, catalog.satrecs):
        line1, line2 = exporter.export_tle(sat)
        out.append({"norad_id": int(nid), "name": name, "tle1": line1, "tle2": line2})
    return out


def perigee_apogee_km(rec: dict) -> tuple[float, float]:
    """Perigee and apogee altitudes from mean elements (two-body approximation)."""
    n = float(rec["MEAN_MOTION"]) * 2 * np.pi / 86400.0  # rad/s
    a = (MU_EARTH / n**2) ** (1 / 3)
    e = float(rec["ECCENTRICITY"])
    return a * (1 - e) - EARTH_RADIUS_KM, a * (1 + e) - EARTH_RADIUS_KM


def build_catalog(
    records: list[dict],
    at: datetime,
    max_epoch_age: timedelta = timedelta(days=14),
    max_perigee_km: float | None = None,
) -> Catalog:
    """Initialise SGP4 satellites, skipping stale element sets and those that fail to init.

    Element sets older than ``max_epoch_age`` are dropped: SGP4 error grows by roughly
    1-3 km per day, so stale data produces meaningless miss distances.
    """
    ids, names, sats, epochs = [], [], [], []
    stale = failed = 0
    for rec in records:
        epoch = datetime.strptime(rec["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
        if abs(at - epoch) > max_epoch_age:
            stale += 1
            continue
        if max_perigee_km is not None and perigee_apogee_km(rec)[0] > max_perigee_km:
            continue
        sat = Satrec()
        try:
            omm.initialize(sat, rec)
        except (KeyError, ValueError):
            failed += 1
            continue
        if sat.error:
            failed += 1
            continue
        ids.append(int(rec["NORAD_CAT_ID"]))
        names.append(rec["OBJECT_NAME"])
        sats.append(sat)
        epochs.append(epoch)
    log.info("Catalog: %d objects (%d stale, %d failed init)", len(sats), stale, failed)
    return Catalog(np.array(ids), names, sats, epochs)
