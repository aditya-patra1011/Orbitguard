"""Per-event features from an event's CDM time series.

Only CDMs issued at least 2 days before TCA are ever passed in (see ``data.py``), so
every feature is available at decision time.

Three families:
- **latest**: every numeric CDM field from the most recent message (what an analyst
  would read today);
- **trend**: how key quantities evolved across messages (risk, miss distance,
  uncertainty, Mahalanobis distance): first-to-last change, last step, slope;
- **history**: number of messages, time span, and summary statistics of risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEGLIGIBLE_RISK = -30.0  # value ESA uses for "no measurable risk"

# Quantities whose evolution between messages carries information.
TREND_COLUMNS = [
    "risk", "max_risk_estimate", "miss_distance", "mahalanobis_distance",
    "t_sigma_t", "c_sigma_t", "c_sigma_r", "c_sigma_n", "c_position_covariance_det",
]
CATEGORICAL = ["c_object_type", "mission_id"]
EXCLUDE_LATEST = {"event_id"}


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of value against days-before-TCA (per day, toward TCA)."""
    ok = ~np.isnan(values)
    if ok.sum() < 2 or np.ptp(times[ok]) == 0:
        return np.nan
    # time_to_tca counts down, so negate to get change per day of elapsed time.
    return float(np.polyfit(-times[ok], values[ok], 1)[0])


def build_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """One row per event_id. ``inputs`` must be sorted chronologically within events."""
    grouped = inputs.groupby("event_id", sort=False)

    latest = grouped.last(skipna=False)
    numeric = [c for c in latest.columns if c not in EXCLUDE_LATEST and c not in CATEGORICAL]
    feats = latest[numeric].add_prefix("last_")
    for col in CATEGORICAL:
        feats[col] = latest[col].astype("category")

    first = grouped.first(skipna=False)
    feats["n_cdms"] = grouped.size()
    feats["span_days"] = first["time_to_tca"] - latest["time_to_tca"]

    previous = grouped.nth(-2).set_index("event_id")
    for col in TREND_COLUMNS:
        feats[f"delta_{col}"] = latest[col] - first[col]
        feats[f"step_{col}"] = latest[col] - previous[col].reindex(latest.index)
        feats[f"slope_{col}"] = grouped[["time_to_tca", col]].apply(
            lambda g, c=col: _slope(g["time_to_tca"].to_numpy(), g[c].to_numpy()))

    risk = grouped["risk"]
    feats["risk_max"] = risk.max()
    feats["risk_min"] = risk.min()
    feats["risk_mean"] = risk.mean()
    feats["risk_std"] = risk.std()
    feats["last_risk_negligible"] = (latest["risk"] <= NEGLIGIBLE_RISK).astype(int)
    feats["frac_negligible"] = grouped["risk"].apply(lambda s: float((s <= NEGLIGIBLE_RISK).mean()))
    # Ratio of along-track uncertainty now vs. at the first message: shrinking
    # uncertainty means later CDMs will sharpen (and often lower) the risk.
    feats["c_sigma_t_ratio"] = latest["c_sigma_t"] / first["c_sigma_t"]
    return feats
