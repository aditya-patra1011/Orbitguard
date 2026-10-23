import numpy as np
import pandas as pd
import pytest

from orbitguard.cdm.data import HIGH_RISK
from orbitguard.cdm.model import LOW, Dataset, RiskModel, align_categories, latest_risk_prediction


def synthetic(n=600, seed=0):
    """Events where a hidden 'uncertainty' feature drives late rises in risk."""
    rng = np.random.default_rng(seed)
    last = rng.choice([-30.0, -12.0, -8.0, -6.5, -5.5, -4.5], size=n)
    noise = rng.normal(size=n)
    uncertain = rng.random(n)
    final = np.where(last >= HIGH_RISK, last - 0.3 + 0.1 * noise, last)
    rises = (last < HIGH_RISK) & (last > -9) & (uncertain > 0.8)
    final = np.where(rises, -5.2, final)
    X = pd.DataFrame({
        "last_risk": last, "uncertainty": uncertain, "noise": noise,
        "c_object_type": pd.Categorical(rng.choice(["DEBRIS", "PAYLOAD"], size=n)),
        "mission_id": pd.Categorical(rng.choice([1, 2], size=n)),
    })
    return Dataset(X, final)


@pytest.fixture(scope="module")
def model():
    return RiskModel(late_riser_threshold=0.5).fit(synthetic())


def test_flagged_events_never_drop_below_threshold(model):
    test = synthetic(seed=1)
    pred = model.predict(test.X).to_numpy()
    flagged = test.last >= HIGH_RISK
    assert np.all(pred[flagged] >= HIGH_RISK)


def test_value_correction_learns_systematic_drop(model):
    # Flagged events end 0.3 lower than their latest risk; the regressor should learn it.
    test = synthetic(seed=2)
    flagged = (test.last >= -5.0)
    pred = model.predict(test.X).to_numpy()
    assert np.mean(test.last[flagged] - pred[flagged]) == pytest.approx(0.3, abs=0.1)


def test_late_risers_are_caught_and_safe_events_stay_low(model):
    test = synthetic(seed=3)
    pred = model.predict(test.X).to_numpy()
    risers = (test.last < HIGH_RISK) & (test.y >= HIGH_RISK)
    stays_low = (test.last < -20)
    assert np.mean(pred[risers] >= HIGH_RISK) > 0.8
    assert np.all(pred[stays_low] == LOW)
    assert model.late_riser_value == pytest.approx(-5.2)


def test_latest_risk_prediction_clips_low_values():
    X = pd.DataFrame({"last_risk": [-4.0, -6.0, -7.0]})
    assert list(latest_risk_prediction(X)) == [-4.0, -6.0, LOW]


def test_unseen_category_at_prediction_time(model):
    test = synthetic(seed=4)
    test.X["c_object_type"] = pd.Categorical(["ROCKET BODY"] * len(test.y))
    assert model.predict(test.X).notna().all()


def test_save_load_round_trip(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    X = synthetic(seed=5).X
    assert RiskModel.load(path).predict(X).equals(model.predict(X))


def test_align_categories_unions_levels():
    a = pd.DataFrame({"c_object_type": pd.Categorical(["DEBRIS"]), "mission_id": pd.Categorical([1])})
    b = pd.DataFrame({"c_object_type": pd.Categorical(["PAYLOAD"]), "mission_id": pd.Categorical([2])})
    align_categories(a, b)
    assert list(a["c_object_type"].cat.categories) == list(b["c_object_type"].cat.categories)
