import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from orbitguard.calibration import (Residuals, calibrate_from_cache, compute_residuals,
                                    fit_model, fit_populations, load_snapshot_pairs)
from orbitguard.covariance import population
from test_orbitguard import make_record


def test_along_track_shift_appears_on_t_axis():
    # Same epoch, mean anomaly ahead by 0.01 deg: the "old" prediction leads the
    # reference along-track by a * 0.01 deg ~ 1.2 km, with ~no radial/normal error.
    new = make_record(1, 53.0, 10.0)
    old = make_record(1, 53.0, 10.01)
    res = compute_residuals([old], [new], min_age_days=0.0)
    r, t, n = res.rtn_km[0]
    assert t == pytest.approx(6780 * np.radians(0.01), rel=0.05)
    assert abs(r) < 0.05 and abs(n) < 0.05


def test_objects_without_a_newer_element_set_are_skipped():
    rec = make_record(1, 53.0, 10.0)
    assert len(compute_residuals([rec], [rec]).age_days) == 0  # age 0 < min_age


def test_element_sets_older_than_three_days_are_not_used():
    new = make_record(1, 53.0, 10.0)
    epoch = datetime.strptime(new["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")
    recent = {**new, "EPOCH": (epoch - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.%f")}
    stale = {**new, "EPOCH": (epoch - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S.%f")}
    assert len(compute_residuals([recent], [new]).age_days) == 1
    assert len(compute_residuals([stale], [new]).age_days) == 0


def _write_snapshot(cache_dir, group, fetched, set_no):
    rec = {**make_record(1, 53.0, 10.0), "ELEMENT_SET_NO": set_no}
    (cache_dir / f"gp_{group}_{fetched:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps([rec]))


def test_snapshot_pairs_are_recent_and_close_together(tmp_path):
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for day in (0, 1, 2, 6):  # the day-6 snapshot is more than 3 days after all others
        _write_snapshot(tmp_path, "starlink", start + timedelta(days=day), set_no=day)
    pairs = load_snapshot_pairs(tmp_path)
    assert sorted((a[0]["ELEMENT_SET_NO"], b[0]["ELEMENT_SET_NO"]) for a, b in pairs) == [
        (0, 1), (0, 2), (1, 2)]

    # Only the latest snapshots of a group are paired.
    assert [(a[0]["ELEMENT_SET_NO"], b[0]["ELEMENT_SET_NO"])
            for a, b in load_snapshot_pairs(tmp_path, max_snapshots=2)] == []
    assert [(a[0]["ELEMENT_SET_NO"], b[0]["ELEMENT_SET_NO"])
            for a, b in load_snapshot_pairs(tmp_path, max_snapshots=3)] == [(1, 2)]


def test_calibration_needs_enough_recent_residuals(tmp_path):
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _write_snapshot(tmp_path, "starlink", start, set_no=1)
    _write_snapshot(tmp_path, "starlink", start + timedelta(days=1), set_no=2)
    assert calibrate_from_cache(tmp_path) is None  # one pair, far fewer than 50 residuals


def test_fit_recovers_known_growth_despite_manoeuvre_outliers():
    rng = np.random.default_rng(1)
    n = 6000
    age = rng.uniform(0.5, 5.0, n)
    sigma0, growth = np.array([0.1, 0.4, 0.1]), np.array([0.05, 1.2, 0.03])
    rtn = rng.normal(size=(n, 3)) * (sigma0 + growth * age[:, None])
    manoeuvre = rng.random(n) < 0.03
    rtn[manoeuvre, 1] += rng.choice([-1, 1], manoeuvre.sum()) * 200.0  # km-scale jumps
    model, summary = fit_model(Residuals(np.arange(n), [""] * n, age, rtn))

    assert model.growth_km_per_day[1] == pytest.approx(1.2, rel=0.1)
    assert model.sigma_rtn(3.0) == pytest.approx(sigma0 + growth * 3.0, rel=0.15)
    assert summary["outlier_fraction"] == pytest.approx(0.03, abs=0.01)


def test_fit_rejects_too_few_residuals():
    with pytest.raises(ValueError):
        fit_model(Residuals(np.arange(3), [""] * 3, np.ones(3), np.zeros((3, 3))))


@pytest.mark.parametrize("name, pop", [
    ("STARLINK-1234", "starlink"), ("ONEWEB-0012", "oneweb"),
    ("FENGYUN 1C DEB", "debris"), ("ISS (ZARYA)", "other"),
])
def test_population(name, pop):
    assert population(name) == pop


def test_population_models_are_fitted_separately():
    rng = np.random.default_rng(2)
    n = 2000
    age = rng.uniform(0.5, 4.0, 2 * n)
    growth = np.where(np.arange(2 * n) < n, 10.0, 0.2)  # manoeuvring vs passive
    rtn = rng.normal(size=(2 * n, 3)) * (0.05 + growth[:, None] * age[:, None])
    names = ["STARLINK-1"] * n + ["COSMOS 2251 DEB"] * n
    pc, _ = fit_populations(Residuals(np.arange(2 * n), names, age, rtn), source="test")
    assert pc.for_object("STARLINK-9").growth_km_per_day[1] == pytest.approx(10.0, rel=0.15)
    assert pc.for_object("FENGYUN 1C DEB").growth_km_per_day[1] == pytest.approx(0.2, rel=0.15)
    assert pc.for_object("ISS (ZARYA)") is pc.default
