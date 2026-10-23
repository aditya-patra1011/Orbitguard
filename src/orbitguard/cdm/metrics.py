"""Competition metric: L = MSE on high-risk events / F2 (lower is better).

- F2 treats events as high risk (final log10 risk >= -6) or low risk. F-beta with
  beta = 2 weighs recall 4x as much as precision: missing a dangerous event is worse
  than a false alarm.
- MSE_HR is the squared error on the truly high-risk events only; for low-risk
  events the exact value does not matter.
- Predictions below -6 are clipped to -6.001, as in the competition, so a model is
  not penalised for exactly how low it puts a low-risk event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitguard.cdm.data import HIGH_RISK

CLIP_EPS = 0.001


@dataclass(frozen=True)
class Score:
    loss: float     # MSE_HR / F2
    mse_hr: float
    f2: float
    precision: float
    recall: float
    n_high: int

    def __str__(self) -> str:
        return (f"L={self.loss:.3f}  MSE_HR={self.mse_hr:.3f}  F2={self.f2:.3f}  "
                f"(precision {self.precision:.2f}, recall {self.recall:.2f}, {self.n_high} high-risk)")


def clip_predictions(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.where(pred < HIGH_RISK, HIGH_RISK - CLIP_EPS, pred)


def score(true_risk, pred_risk) -> Score:
    r = np.asarray(true_risk, dtype=float)
    p = clip_predictions(pred_risk)
    hr_true, hr_pred = r >= HIGH_RISK, p >= HIGH_RISK
    tp = np.sum(hr_true & hr_pred)
    fp = np.sum(~hr_true & hr_pred)
    fn = np.sum(hr_true & ~hr_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    mse_hr = float(np.mean((r[hr_true] - p[hr_true]) ** 2)) if hr_true.any() else 0.0
    loss = mse_hr / f2 if f2 > 0 else float("inf")
    return Score(float(loss), mse_hr, float(f2), float(precision), float(recall), int(hr_true.sum()))
