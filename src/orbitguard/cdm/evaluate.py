"""Train the final model and evaluate it once on the held-out test set.

The test set is used here and nowhere else: model design and the late-riser
threshold come from training-set cross-validation (``model.cross_validate``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from orbitguard.cdm.baselines import constant_risk, latest_risk
from orbitguard.cdm.data import HIGH_RISK, Task
from orbitguard.cdm.metrics import Score, score
from orbitguard.cdm.model import Dataset, RiskModel

# Uriot et al. 2020, Table 3: final test-set ranking (top 10 of 97 teams).
LEADERBOARD = {
    "sesc (1st)": 0.556, "dietmarw (2nd)": 0.571, "Magpies (3rd)": 0.585,
    "Vidente": 0.610, "DeCRA": 0.615, "Valis": 0.628, "DunderMifflin": 0.628,
    "madks": 0.634, "vhrique": 0.649, "Spacemeister": 0.649,
}


@dataclass
class Evaluation:
    model: Score
    latest_risk: Score
    constant_risk: Score
    confusion: dict
    leaderboard_rank: int        # position among the published top 10 + LRP
    top_features: dict

    def to_dict(self) -> dict:
        return asdict(self)


def confusion(true, pred) -> dict:
    t, p = np.asarray(true) >= HIGH_RISK, np.asarray(pred) >= HIGH_RISK
    return {"true_positive": int(np.sum(t & p)), "false_positive": int(np.sum(~t & p)),
            "false_negative": int(np.sum(t & ~p)), "true_negative": int(np.sum(~t & ~p))}


def top_features(model: RiskModel, n: int = 10) -> dict:
    out = {}
    for name, est in [("value_correction", model.regressor), ("late_riser", model.classifier)]:
        gain = pd.Series(est.booster_.feature_importance("gain"), index=est.booster_.feature_name())
        out[name] = (gain / gain.sum()).sort_values(ascending=False).head(n).round(4).to_dict()
    return out


def train_final_model(strict: Task, augmentation: Task, threshold: float, seed: int = 0) -> RiskModel:
    return RiskModel(late_riser_threshold=threshold, seed=seed).fit(
        Dataset.from_task(strict), Dataset.from_task(augmentation))


def evaluate(model: RiskModel, test: Task) -> Evaluation:
    data = Dataset.from_task(test)
    pred = model.predict(data.X)
    s = score(data.y, pred)
    rank = 1 + sum(v < s.loss for v in LEADERBOARD.values()) + int(
        score(data.y, latest_risk(test).loc[data.X.index]).loss < s.loss)
    return Evaluation(
        model=s,
        latest_risk=score(data.y, latest_risk(test).loc[data.X.index]),
        constant_risk=score(data.y, constant_risk(test).loc[data.X.index]),
        confusion=confusion(data.y, pred),
        leaderboard_rank=rank,
        top_features=top_features(model),
    )
