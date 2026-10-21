import numpy as np
import pytest

from orbitguard.covariance import (
    DEFAULT_MODEL, CovarianceModel, PopulationCovariance, load_covariance, rtn_basis,
)

R = np.array([6800.0, 0.0, 0.0])
V = np.array([0.0, 6.0, 4.0])


def test_rtn_basis_is_right_handed_orthonormal():
    b = rtn_basis(R, V)
    assert b.T @ b == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(b) == pytest.approx(1.0)
    assert b[:, 0] == pytest.approx(R / np.linalg.norm(R))
    # For a circular orbit the along-track axis is the velocity direction.
    assert b[:, 1] == pytest.approx(V / np.linalg.norm(V))


def test_covariance_variances_land_on_rtn_axes():
    model = CovarianceModel((0.1, 0.5, 0.2), (0.0, 0.0, 0.0))
    cov = model.covariance(R, V, age_days=0.0)
    b = rtn_basis(R, V)
    assert b.T @ cov @ b == pytest.approx(np.diag([0.01, 0.25, 0.04]), abs=1e-12)


def test_uncertainty_grows_with_age_in_either_direction():
    s0 = DEFAULT_MODEL.sigma_rtn(0.0)
    assert np.all(DEFAULT_MODEL.sigma_rtn(2.0) > s0)
    assert DEFAULT_MODEL.sigma_rtn(-2.0) == pytest.approx(DEFAULT_MODEL.sigma_rtn(2.0))
    # Along-track dominates.
    assert DEFAULT_MODEL.sigma_rtn(1.0)[1] > DEFAULT_MODEL.sigma_rtn(1.0)[0]


def test_save_load_round_trip(tmp_path):
    model = CovarianceModel((0.1, 0.2, 0.3), (0.4, 0.5, 0.6), source="calibrated")
    path = tmp_path / "cov.json"
    model.save(path)
    assert CovarianceModel.load(path) == model


def test_population_covariance_round_trip(tmp_path):
    starlink = CovarianceModel((0.1, 1.0, 0.1), (0.2, 10.0, 0.1), source="starlink")
    default = CovarianceModel((0.1, 0.5, 0.1), (0.05, 1.0, 0.05), source="pooled")
    model = PopulationCovariance({"starlink": starlink}, default, source="test")
    path = tmp_path / "cov.json"
    model.save(path)
    loaded = load_covariance(path)
    assert loaded == model
    assert loaded.for_object("STARLINK-1007") == starlink
    assert loaded.for_object("ISS (ZARYA)") == default


def test_single_model_file_still_loads(tmp_path):
    path = tmp_path / "cov.json"
    DEFAULT_MODEL.save(path)
    assert load_covariance(path) == DEFAULT_MODEL


def test_zero_intercept_model_stays_positive_definite_at_epoch():
    model = CovarianceModel((0.0, 0.0, 0.0), (0.04, 0.4, 0.05))
    assert np.all(np.linalg.eigvalsh(model.covariance(R, V, age_days=0.0)) > 0)
