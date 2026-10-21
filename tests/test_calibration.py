import numpy as np
import pytest

from orbitguard.calibration import Residuals, compute_residuals, fit_model, fit_populations
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
