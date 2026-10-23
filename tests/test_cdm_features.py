import numpy as np
import pandas as pd
import pytest

from orbitguard.cdm.data import sort_cdms
from orbitguard.cdm.features import TREND_COLUMNS, build_features


def event(event_id, times, risks, miss, object_type="DEBRIS"):
    n = len(times)
    rows = {c: np.linspace(1.0, 2.0, n) for c in TREND_COLUMNS}
    rows.update({
        "event_id": event_id, "time_to_tca": times, "risk": risks, "miss_distance": miss,
        "c_object_type": object_type, "mission_id": 3, "relative_speed": 10000.0,
    })
    return pd.DataFrame(rows)


@pytest.fixture
def feats():
    raw = pd.concat([
        event(1, [6.0, 4.0, 2.0], [-10.0, -8.0, -5.0], [900.0, 600.0, 300.0]),
        event(2, [3.0], [-30.0], [5000.0], object_type="PAYLOAD"),
    ])
    return build_features(sort_cdms(raw.sample(frac=1.0, random_state=0)))


def test_one_row_per_event_with_latest_values(feats):
    assert list(feats.index) == [1, 2]
    assert feats.loc[1, "last_risk"] == -5.0
    assert feats.loc[1, "last_time_to_tca"] == 2.0
    assert feats.loc[2, "c_object_type"] == "PAYLOAD"
    assert feats["c_object_type"].dtype == "category"


def test_history_and_trend_features(feats):
    assert feats.loc[1, "n_cdms"] == 3
    assert feats.loc[1, "span_days"] == 4.0
    assert feats.loc[1, "delta_risk"] == 5.0              # -10 -> -5
    assert feats.loc[1, "step_risk"] == 3.0               # -8 -> -5
    assert feats.loc[1, "slope_miss_distance"] == pytest.approx(-150.0)  # m per day
    assert feats.loc[1, "risk_max"] == -5.0


def test_single_message_events_have_undefined_trends(feats):
    assert feats.loc[2, "n_cdms"] == 1
    assert np.isnan(feats.loc[2, "step_risk"])
    assert np.isnan(feats.loc[2, "slope_risk"])
    assert feats.loc[2, "last_risk_negligible"] == 1
    assert feats.loc[2, "frac_negligible"] == 1.0
