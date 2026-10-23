import numpy as np
import pandas as pd
import pytest

from orbitguard.cdm.baselines import constant_risk, latest_risk
from orbitguard.cdm.data import DEFAULT_DATA_DIR, build_training_task, load_test_task, load_training_task
from orbitguard.cdm.metrics import score

HAVE_DATA = (DEFAULT_DATA_DIR / "test_data_private.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="ESA CDM dataset not downloaded")


def cdm_rows(event_id, times, risks):
    return pd.DataFrame({"event_id": event_id, "time_to_tca": times, "risk": risks})


def test_training_task_applies_competition_rules():
    raw = pd.concat([
        cdm_rows(1, [0.5, 2.5, 3.0], [-4.0, -9.0, -10.0]),   # eligible (shuffled order)
        cdm_rows(2, [1.5, 0.5], [-5.0, -5.0]),               # first CDM < 2 days: excluded
        cdm_rows(3, [3.0, 1.5], [-5.0, -5.0]),               # last CDM >= 1 day: excluded
    ])
    task = build_training_task(raw)
    assert list(task.event_ids) == [1]
    assert task.targets.loc[1] == -4.0                      # risk of the last CDM
    assert (task.inputs.time_to_tca >= 2).all()             # nothing within 2 days of TCA
    assert list(task.inputs.risk) == [-10.0, -9.0]          # chronological order
    assert latest_risk(task).loc[1] == -9.0


def test_metric_hand_computed_case():
    true = [-5.0, -7.0, -4.0, -20.0]
    pred = [-5.5, -5.0, -30.0, -25.0]   # TP, FP, FN (clipped to -6.001), TN
    s = score(true, pred)
    assert (s.precision, s.recall, s.f2) == pytest.approx((0.5, 0.5, 0.5))
    assert s.mse_hr == pytest.approx((0.5**2 + 2.001**2) / 2)
    assert s.loss == pytest.approx(s.mse_hr / 0.5)


def test_metric_perfect_and_degenerate_predictions():
    true = np.array([-5.0, -8.0, -3.0])
    perfect = score(true, true)
    assert perfect.loss == 0.0 and perfect.f2 == 1.0
    assert score(true, np.full(3, -30.0)).loss == float("inf")   # never flags high risk


@needs_data
def test_reproduces_published_baselines_on_test_set():
    # Uriot et al. 2020, Table 3: LRP L = 0.694 (MSE_HR 0.513, F2 0.739); CRP L = 2.5.
    task = load_test_task()
    lrp = score(task.targets, latest_risk(task))
    assert lrp.n_high == 150 and len(task.targets) == 2167
    assert (round(lrp.loss, 3), round(lrp.mse_hr, 3), round(lrp.f2, 3)) == (0.694, 0.513, 0.739)
    assert score(task.targets, constant_risk(task)).loss == pytest.approx(2.5, abs=0.01)


@needs_data
def test_training_task_is_leak_free():
    task = load_training_task()
    assert (task.inputs.time_to_tca >= 2).all()
    assert set(task.inputs.event_id) == set(task.event_ids)
