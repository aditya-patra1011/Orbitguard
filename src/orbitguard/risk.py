"""Probability of collision (Pc) for a short-duration encounter.

Standard 2-D formulation (Foster 1992, Alfano 2005, as used by NASA CARA):

- Encounters in LEO last well under a second, so relative motion is treated as a
  straight line and position uncertainty as constant during the encounter.
- Project the relative position and the combined position covariance of both
  objects onto the *encounter plane*, the plane perpendicular to the relative
  velocity at the time of closest approach (TCA).
- Pc is the integral of that 2-D Gaussian over a disc of radius HBR (the combined
  hard-body radius of the two objects), centred on the origin.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def encounter_plane(dr: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Return a (2, 3) matrix whose rows span the plane perpendicular to ``dv``.

    The first axis points along the miss vector, so the miss lies at (|dr_perp|, 0).
    """
    z = dv / np.linalg.norm(dv)
    x = dr - np.dot(dr, z) * z
    if np.linalg.norm(x) < 1e-12:
        # Zero miss distance: any axis perpendicular to dv will do.
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, helper)
    x /= np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x)])


def collision_probability(dr: np.ndarray, dv: np.ndarray, cov: np.ndarray, hbr_km: float) -> float:
    """Pc for relative position ``dr`` (km), relative velocity ``dv`` (km/s) at TCA,
    combined 3x3 position covariance ``cov`` (km^2) and hard-body radius ``hbr_km``."""
    proj = encounter_plane(np.asarray(dr, float), np.asarray(dv, float))
    mean = proj @ dr
    cov2 = proj @ cov @ proj.T

    # Rotate to the principal axes of the 2-D covariance so the Gaussian separates.
    var, axes = np.linalg.eigh(cov2)
    if var.min() <= 0:
        raise ValueError("Projected covariance must be positive definite")
    mx, my = axes.T @ mean
    sx, sy = np.sqrt(var)

    def strip(x: float) -> float:
        # Probability mass in the vertical chord of the disc at x.
        h = np.sqrt(max(hbr_km**2 - x**2, 0.0))
        density = np.exp(-0.5 * ((x - mx) / sx) ** 2) / (sx * np.sqrt(2 * np.pi))
        return density * (ndtr((h - my) / sy) - ndtr((-h - my) / sy))

    points = [mx] if -hbr_km < mx < hbr_km else None
    pc, _ = quad(strip, -hbr_km, hbr_km, points=points, epsabs=0.0, epsrel=1e-9, limit=200)
    return float(min(max(pc, 0.0), 1.0))
