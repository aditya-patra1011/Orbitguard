import json
from datetime import timedelta

import pytest
from sqlalchemy.orm import Session

from orbitguard.pipeline import PipelineConfig, detect_drift, run_pipeline
from orbitguard.store import list_pipeline_runs, list_runs, make_engine, save_pipeline_run
from test_orbitguard import EPOCH, _crossing_pair, make_record

NOW = EPOCH  # the synthetic crossing happens ~16 min after epoch


def write_snapshot(cache_dir, group, records, when):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"gp_{group}_{when:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(records))


def no_network(group, cache_dir, refresh, now):
    """Stand-in for fetch_group: the test has already put fresh snapshots in the cache."""


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """The pipeline must never download during these tests."""
    def refuse(*args, **kwargs):
        raise AssertionError("unexpected network request")
    monkeypatch.setattr("orbitguard.ingest.requests.get", refuse)


@pytest.fixture
def setup(tmp_path):
    cfg = PipelineConfig(groups=["test"], hours=0.5, threshold_km=20.0, calibrate=False,
                         cache_dir=tmp_path / "raw", results_dir=tmp_path / "processed",
                         covariance_path=tmp_path / "cov.json")
    return cfg, make_engine("sqlite://")


def test_successful_run_stores_results_and_metrics(setup):
    cfg, engine = setup
    write_snapshot(cfg.cache_dir, "test", _crossing_pair(), NOW - timedelta(minutes=5))
    out = run_pipeline(cfg, engine, now=NOW, fetch=no_network)

    assert out.status == "success" and out.run_id is not None
    assert out.metrics["objects_screened"] == 2 and out.metrics["conjunctions"] == 1
    assert out.metrics["conjunctions_per_1000_objects"] == 500.0
    assert out.drift_alerts == [] and "needs 3 earlier" in out.message
    assert len(list(cfg.results_dir.glob("conjunctions_*.json"))) == 1
    with Session(engine) as s:
        assert len(list_runs(s)) == 1
        assert list_pipeline_runs(s)[0].status == "success"


def test_validation_failure_stops_before_screening(setup):
    cfg, engine = setup
    bad = [{**r, "ECCENTRICITY": 2.0} for r in _crossing_pair()]   # impossible orbits
    write_snapshot(cfg.cache_dir, "test", bad, NOW - timedelta(minutes=5))
    out = run_pipeline(cfg, engine, now=NOW, fetch=no_network)

    assert out.status == "failed_validation" and out.run_id is None
    assert "invalid_records" in out.message
    with Session(engine) as s:
        assert list_runs(s) == []                                    # nothing screened or stored
        assert list_pipeline_runs(s)[0].status == "failed_validation"


def test_unexpected_errors_are_recorded(setup):
    cfg, engine = setup

    def broken_fetch(group, cache_dir, refresh, now):
        raise ConnectionError("CelesTrak unreachable")

    out = run_pipeline(cfg, engine, now=NOW, fetch=broken_fetch)
    assert out.status == "error" and "CelesTrak unreachable" in out.message
    with Session(engine) as s:
        assert list_pipeline_runs(s)[0].status == "error"


def test_recalibration_runs_when_two_snapshots_exist(setup):
    cfg, engine = setup
    cfg.calibrate = True
    # 400 debris objects in two snapshots, the newer one a day later.
    old = [{**make_record(i, 70.0, float(i % 360)), "OBJECT_NAME": "TEST DEB"} for i in range(400)]
    new = [{**r, "EPOCH": (EPOCH + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")} for r in old]
    write_snapshot(cfg.cache_dir, "test", old, EPOCH + timedelta(minutes=5))
    write_snapshot(cfg.cache_dir, "test", new, EPOCH + timedelta(days=1, minutes=5))
    out = run_pipeline(cfg, engine, now=NOW + timedelta(days=1), fetch=no_network)

    assert out.status == "success", out.message
    assert out.metrics["calibration_pairs"] == 1
    assert "debris" in out.metrics["calibrated_sigma_t_2d_km"]
    assert cfg.covariance_path.exists()


def test_drift_alert_after_enough_history(setup):
    cfg, engine = setup
    with Session(engine) as s:
        for i in range(5):  # stable history: 2 objects, ~500 conjunctions per 1000 objects
            save_pipeline_run(s, started_at=NOW - timedelta(hours=6 * (i + 1)), finished_at=NOW,
                              status="success", groups="test",
                              metrics={"objects_screened": 2, "high_risk_events": 0,
                                       "conjunctions_per_1000_objects": 500.0 + i,
                                       "median_element_age_days": 0.0})
        s.commit()
    far = [{**r, "MEAN_MOTION": 13.0 + i} for i, r in enumerate(_crossing_pair())]  # no encounter
    write_snapshot(cfg.cache_dir, "test", far, NOW - timedelta(minutes=5))
    out = run_pipeline(cfg, engine, now=NOW, fetch=no_network)

    assert out.status == "success" and out.message == ""
    assert [a["metric"] for a in out.drift_alerts] == ["conjunctions_per_1000_objects"]


def test_detect_drift_is_robust_to_one_outlier_in_history():
    history = [{"m": v} for v in (100, 101, 99, 100, 5000)]          # one wild past value
    assert detect_drift({"m": 102}, history, metrics=("m",)) == []
    alert = detect_drift({"m": 160}, history, metrics=("m",))[0]
    assert alert.baseline_median == 100 and alert.robust_z > 3


def test_detect_drift_needs_minimum_history():
    assert detect_drift({"m": 1e9}, [{"m": 1}, {"m": 1}], metrics=("m",)) == []
