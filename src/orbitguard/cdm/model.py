"""Final-risk model: the latest-risk forecast plus two learned corrections.

Why this shape: with only 66 high-risk events among 8,293 training events, models that
predict the final risk from scratch learn that "risky-looking events usually end
safe", lower their predictions and lose the recall that F2 rewards most. Every such
model scored far worse than the naive latest-risk forecast (LRP) in cross-validation.
So the model keeps LRP as the backbone and only learns what LRP gets wrong:

- **A, value correction.** For events LRP flags (latest risk >= -6), a regressor
  predicts final - latest risk. It is trained only on flagged events that truly
  ended high, because only those enter MSE_HR. Its output never drops below -6, so
  it cannot turn a flagged event into a miss.
- **B, late risers.** For events LRP does not flag, a classifier estimates the chance
  that the risk still rises above -6. Events above a threshold (chosen by
  cross-validation) are flagged at the typical final risk of training late risers.

Both components also learn from the augmentation task (events excluded by the strict
competition rules), which quadruples the late-riser examples (19 -> 69).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from orbitguard.cdm.data import HIGH_RISK, Task
from orbitguard.cdm.features import CATEGORICAL, build_features
from orbitguard.cdm.metrics import CLIP_EPS, Score, score

LOW = HIGH_RISK - CLIP_EPS

REGRESSOR_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                        verbose=-1)
CLASSIFIER_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                         verbose=-1)


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray

    @classmethod
    def from_task(cls, task: Task) -> "Dataset":
        X = build_features(task.inputs)
        return cls(X, task.targets.loc[X.index].to_numpy(dtype=float))

    @property
    def last(self) -> np.ndarray:
        return self.X["last_risk"].to_numpy(dtype=float)

    def subset(self, mask) -> "Dataset":
        return Dataset(self.X[mask], self.y[mask])


def align_categories(*frames: pd.DataFrame, categories: dict | None = None) -> dict:
    """Give categorical columns identical categories across frames (in place)."""
    if categories is None:
        categories = {c: pd.api.types.union_categoricals([f[c] for f in frames]).categories
                      for c in CATEGORICAL}
    for f in frames:
        for c, cats in categories.items():
            f[c] = f[c].astype("object").astype(pd.CategoricalDtype(cats))
    return categories


@dataclass
class RiskModel:
    late_riser_threshold: float = 0.05
    seed: int = 0
    regressor: lgb.LGBMRegressor | None = field(default=None, repr=False)
    classifier: lgb.LGBMClassifier | None = field(default=None, repr=False)
    late_riser_value: float = HIGH_RISK + 0.3
    categories: dict | None = field(default=None, repr=False)

    def fit(self, train: Dataset, extra: Dataset | None = None) -> "RiskModel":
        frames = [train.X.copy()] + ([extra.X.copy()] if extra is not None else [])
        self.categories = align_categories(*frames)
        data = Dataset(pd.concat(frames),
                       np.concatenate([train.y] + ([extra.y] if extra is not None else [])))
        last, y = data.last, data.y
        flagged_high = (last >= HIGH_RISK) & (y >= HIGH_RISK)
        self.regressor = lgb.LGBMRegressor(**REGRESSOR_PARAMS, random_state=self.seed).fit(
            data.X[flagged_high], y[flagged_high] - last[flagged_high])

        unflagged = last < HIGH_RISK
        risers = (y >= HIGH_RISK).astype(int)
        self.classifier = lgb.LGBMClassifier(**CLASSIFIER_PARAMS, random_state=self.seed).fit(
            data.X[unflagged], risers[unflagged])
        if (unflagged & (y >= HIGH_RISK)).any():
            self.late_riser_value = float(np.median(y[unflagged & (y >= HIGH_RISK)]))
        return self

    def late_riser_probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.copy()
        align_categories(X, categories=self.categories)
        last = X["last_risk"].to_numpy(dtype=float)
        flag = last >= HIGH_RISK
        pred = np.full(len(X), LOW)
        if flag.any():
            pred[flag] = np.maximum(last[flag] + self.regressor.predict(X[flag]), HIGH_RISK)
        if (~flag).any():
            rises = self.late_riser_probability(X[~flag]) >= self.late_riser_threshold
            idx = np.flatnonzero(~flag)[rises]
            pred[idx] = max(self.late_riser_value, HIGH_RISK)
        return pd.Series(pred, index=X.index, name="predicted_risk")

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RiskModel":
        return joblib.load(path)


def latest_risk_prediction(X: pd.DataFrame) -> np.ndarray:
    last = X["last_risk"].to_numpy(dtype=float)
    return np.where(last >= HIGH_RISK, last, LOW)


@dataclass
class CVResult:
    threshold: float | None  # None for the LRP baseline
    scores: list[Score]

    @property
    def mean_loss(self) -> float:
        return float(np.mean([s.loss for s in self.scores]))


def cross_validate(train: Dataset, extra: Dataset | None, thresholds=(0.05, 0.1, 0.2, 0.5),
                   n_splits: int = 5, n_repeats: int = 4, seed: int = 1) -> list[CVResult]:
    """Repeated stratified CV on the strict training events only. Augmentation events
    are added to every training fold and never appear in a validation fold."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results = {None: CVResult(None, [])} | {t: CVResult(t, []) for t in thresholds}
    for fold, (tr, va) in enumerate(cv.split(train.X, train.y >= HIGH_RISK)):
        fold_train, fold_val = train.subset(np.isin(np.arange(len(train.y)), tr)), train.subset(
            np.isin(np.arange(len(train.y)), va))
        results[None].scores.append(score(fold_val.y, latest_risk_prediction(fold_val.X)))
        model = RiskModel(seed=fold).fit(fold_train, extra)
        for t in thresholds:
            model.late_riser_threshold = t
            results[t].scores.append(score(fold_val.y, model.predict(fold_val.X)))
    return list(results.values())


def select_threshold(results: list[CVResult]) -> float:
    """Threshold with the lowest mean CV loss (the LRP baseline is not a candidate)."""
    return min((r for r in results if r.threshold is not None), key=lambda r: r.mean_loss).threshold
