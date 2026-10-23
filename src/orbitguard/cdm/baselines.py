"""Reference predictors from the competition paper."""

from __future__ import annotations

import pandas as pd

from orbitguard.cdm.data import Task


def latest_risk(task: Task) -> pd.Series:
    """LRP baseline: predict the risk of the most recent usable CDM (naive forecast)."""
    return task.inputs.groupby("event_id")["risk"].last().loc[task.event_ids]


def constant_risk(task: Task, value: float = -5.0) -> pd.Series:
    """CRP baseline: the same risk for every event."""
    return pd.Series(value, index=task.event_ids)
