import numpy as np
import pytest

from orbitguard.risk import collision_probability, encounter_plane


def monte_carlo_pc(dr, dv, cov, hbr, n=2_000_000, seed=0):
    """Reference Pc: sample relative positions, count those inside the hard-body disc
    in the encounter plane."""
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(dr, cov, size=n)
    plane = encounter_plane(np.asarray(dr, float), np.asarray(dv, float))
    radial = np.linalg.norm(samples @ plane.T, axis=1)
    return np.mean(radial < hbr)


def test_encounter_plane_is_orthonormal_and_perpendicular_to_dv():
    dr, dv = np.array([0.3, -0.2, 0.1]), np.array([1.0, 7.0, -3.0])
    p = encounter_plane(dr, dv)
    assert p @ p.T == pytest.approx(np.eye(2), abs=1e-12)
    assert p @ dv == pytest.approx([0, 0], abs=1e-12)


def test_zero_miss_isotropic_matches_closed_form():
    # Disc centred on an isotropic Gaussian: Pc = 1 - exp(-R^2 / 2 sigma^2).
    sigma, hbr = 0.2, 0.05
    pc = collision_probability(np.zeros(3), np.array([0, 10.0, 0]), sigma**2 * np.eye(3), hbr)
    assert pc == pytest.approx(1 - np.exp(-hbr**2 / (2 * sigma**2)), rel=1e-6)


@pytest.mark.parametrize("dr, cov", [
    ([0.15, 0.0, 0.0], np.diag([0.04, 0.5, 0.01])),
    ([0.05, 0.0, 0.08], np.array([[0.02, 0.0, 0.006], [0.0, 1.0, 0.0], [0.006, 0.0, 0.01]])),
])
def test_matches_monte_carlo(dr, cov):
    dr, dv, hbr = np.array(dr), np.array([0.0, 12.0, 0.5]), 0.03
    pc = collision_probability(dr, dv, cov, hbr)
    ref = monte_carlo_pc(dr, dv, cov, hbr)
    stderr = np.sqrt(ref * (1 - ref) / 2_000_000)
    assert abs(pc - ref) < 4 * stderr


def test_uncertainty_along_relative_velocity_is_irrelevant():
    dr, dv, hbr = np.array([0.1, 0.0, 0.0]), np.array([0.0, 14.0, 0.0]), 0.02
    base = np.diag([0.01, 0.01, 0.01])
    stretched = base + np.diag([0.0, 100.0, 0.0])  # only along dv
    assert collision_probability(dr, dv, stretched, hbr) == pytest.approx(
        collision_probability(dr, dv, base, hbr), rel=1e-9)


def test_far_miss_is_negligible_and_bounded():
    pc = collision_probability(np.array([50.0, 0, 0]), np.array([0, 7.0, 0]), 0.01 * np.eye(3), 0.01)
    assert 0.0 <= pc < 1e-12


def test_dilution_region_larger_uncertainty_can_lower_pc():
    # With a near-zero miss, inflating the covariance spreads probability away from
    # the disc: Pc falls. This "dilution" is why huge TLE errors yield tiny Pc.
    dr, dv, hbr = np.array([0.01, 0, 0]), np.array([0, 10.0, 0]), 0.01
    tight = collision_probability(dr, dv, 0.01 * np.eye(3), hbr)
    loose = collision_probability(dr, dv, 25.0 * np.eye(3), hbr)
    assert loose < tight
