from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbitguard.ingest import validate_records
from orbitguard.propagation import EARTH_RADIUS_KM, build_catalog, propagate_one
from orbitguard.screening import screen

EPOCH = datetime(2026, 10, 19, 10, 0, 0, tzinfo=timezone.utc)


def make_record(norad_id, inclination, mean_anomaly, raan=0.0, mean_motion=15.2, name=None):
    return {
        "OBJECT_NAME": name or f"SAT-{norad_id}", "OBJECT_ID": "2026-001A",
        "EPOCH": EPOCH.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "MEAN_MOTION": mean_motion, "ECCENTRICITY": 0.0001, "INCLINATION": inclination,
        "RA_OF_ASC_NODE": raan, "ARG_OF_PERICENTER": 0.0, "MEAN_ANOMALY": mean_anomaly,
        "EPHEMERIS_TYPE": 0, "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": norad_id,
        "ELEMENT_SET_NO": 999, "REV_AT_EPOCH": 1, "BSTAR": 0.0, "MEAN_MOTION_DOT": 0.0,
        "MEAN_MOTION_DDOT": 0.0,
    }


def test_validate_drops_bad_records():
    good = make_record(1, 51.6, 0.0)
    missing = {k: v for k, v in good.items() if k != "BSTAR"}
    hyperbolic = {**good, "ECCENTRICITY": 1.2}
    bad_epoch = {**good, "EPOCH": "yesterday"}
    assert validate_records([good, missing, hyperbolic, bad_epoch]) == [good]


def test_propagated_altitude_is_leo():
    cat = build_catalog([make_record(1, 51.6, 0.0)], at=EPOCH)
    r, _ = propagate_one(cat.satrecs[0], EPOCH + timedelta(hours=3))
    altitude = np.linalg.norm(r) - EARTH_RADIUS_KM
    assert 350 < altitude < 550


def test_stale_elements_are_dropped():
    cat = build_catalog([make_record(1, 51.6, 0.0)], at=EPOCH + timedelta(days=30))
    assert len(cat) == 0


def _crossing_pair():
    # Same altitude and node, different inclinations, both reaching the ascending
    # node (mean anomaly 360 deg) at the same moment. SGP4 short-period terms differ
    # with inclination, so the true miss is ~9 km rather than zero.
    return [make_record(1, 53.0, 300.0), make_record(2, 97.5, 300.0)]


def test_finds_crossing_conjunction():
    cat = build_catalog(_crossing_pair(), at=EPOCH)
    assert screen(cat, EPOCH, hours=0.5, step_s=10.0, threshold_km=5.0) == []
    found = screen(cat, EPOCH, hours=0.5, step_s=10.0, threshold_km=20.0)
    assert len(found) == 1
    c = found[0]
    assert 5.0 < c.miss_distance_km < 15.0
    assert c.relative_speed_km_s > 5.0
    assert not c.co_orbital
    assert c.pc is not None and 0.0 <= c.pc <= 1.0
    # 60 deg of mean anomaly at 15.2 rev/day is ~947 s.
    expected = EPOCH + timedelta(seconds=60 / 360 * 86400 / 15.2)
    assert abs((c.tca - expected).total_seconds()) < 30


def test_refined_tca_matches_brute_force():
    cat = build_catalog(_crossing_pair(), at=EPOCH)
    c = screen(cat, EPOCH, hours=0.5, step_s=10.0, threshold_km=20.0)[0]
    dists = []
    for dt in np.arange(-20, 20, 0.01):
        t = c.tca + timedelta(seconds=float(dt))
        r1, _ = propagate_one(cat.satrecs[0], t)
        r2, _ = propagate_one(cat.satrecs[1], t)
        dists.append(np.linalg.norm(r1 - r2))
    assert c.miss_distance_km == pytest.approx(min(dists), abs=0.05)


def test_coarse_step_does_not_miss_fast_encounter():
    # A 60 s grid moves ~900 km between samples at this relative speed;
    # the padded screening radius must still catch it.
    cat = build_catalog(_crossing_pair(), at=EPOCH)
    assert len(screen(cat, EPOCH, hours=0.5, step_s=60.0, threshold_km=20.0)) == 1


def test_formation_flyers_are_flagged_co_orbital():
    cat = build_catalog([make_record(1, 53.0, 0.0), make_record(2, 53.0, 0.01)], at=EPOCH)
    found = screen(cat, EPOCH, hours=0.5, step_s=10.0, threshold_km=5.0)
    assert found and all(c.co_orbital for c in found)
    assert all(c.pc is None for c in found)


def test_pc_rises_when_uncertainty_matches_miss_distance():
    # The crossing misses by ~9 km. With 10 m uncertainty a hit is impossible;
    # with ~10 km along-track uncertainty a hit becomes plausible.
    from orbitguard.covariance import CovarianceModel
    cat = build_catalog(_crossing_pair(), at=EPOCH)
    tight = CovarianceModel((0.01, 0.01, 0.01), (0.0, 0.0, 0.0))
    loose = CovarianceModel((5.0, 10.0, 5.0), (0.0, 0.0, 0.0))
    pc_tight = screen(cat, EPOCH, hours=0.5, threshold_km=20.0, cov_model=tight)[0].pc
    pc_loose = screen(cat, EPOCH, hours=0.5, threshold_km=20.0, cov_model=loose)[0].pc
    assert pc_tight < 1e-12 < pc_loose


def test_distant_objects_produce_nothing():
    recs = [make_record(1, 53.0, 0.0), make_record(2, 53.0, 0.0, mean_motion=13.0)]
    cat = build_catalog(recs, at=EPOCH)
    assert screen(cat, EPOCH, hours=0.5, threshold_km=5.0) == []
