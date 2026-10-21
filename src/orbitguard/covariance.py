"""Position uncertainty model for SGP4 / GP element sets.

Public GP data carries no covariance, so uncertainty is modelled in the RTN frame
(Radial, Transverse/along-track, Normal) as a 1-sigma error that grows linearly
with the age of the element set:

    sigma_axis(age) = sigma0_axis + growth_axis * |age_days|

Along-track error dominates because small errors in the mean motion and drag
accumulate along the orbit. The default parameters are order-of-magnitude
assumptions for LEO; ``orbitguard calibrate`` replaces them with values measured
from successive element sets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def rtn_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """3x3 matrix whose columns are the R, T, N unit vectors for state (r, v)."""
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.column_stack([r_hat, t_hat, n_hat])


@dataclass(frozen=True)
class CovarianceModel:
    sigma0_km: tuple[float, float, float]           # R, T, N at epoch
    growth_km_per_day: tuple[float, float, float]   # R, T, N
    source: str = "assumed"

    def sigma_rtn(self, age_days: float) -> np.ndarray:
        return np.asarray(self.sigma0_km) + np.asarray(self.growth_km_per_day) * abs(age_days)

    def covariance(self, r: np.ndarray, v: np.ndarray, age_days: float) -> np.ndarray:
        """3x3 position covariance (km^2) in the frame of (r, v), e.g. TEME."""
        basis = rtn_basis(r, v)
        return basis @ np.diag(self.sigma_rtn(age_days) ** 2) @ basis.T

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CovarianceModel":
        d = json.loads(path.read_text())
        return cls(tuple(d["sigma0_km"]), tuple(d["growth_km_per_day"]), d.get("source", str(path)))


# Order-of-magnitude LEO values (along-track ~0.5 km at epoch, growing ~1 km/day).
DEFAULT_MODEL = CovarianceModel(
    sigma0_km=(0.1, 0.5, 0.1),
    growth_km_per_day=(0.05, 1.0, 0.05),
    source="assumed",
)
