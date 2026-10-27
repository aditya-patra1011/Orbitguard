import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from orbitguard.api import DEFAULT_RISK_MODEL_PATH, create_app
from orbitguard.cdm import DEFAULT_DATA_DIR
from orbitguard.store import make_engine, save_result
from test_store import result

HAVE_MODEL = DEFAULT_RISK_MODEL_PATH.exists() and (DEFAULT_DATA_DIR / "test_data.csv").exists()


@pytest.fixture
def client(tmp_path):
    engine = make_engine("sqlite://")
    with Session(engine) as s:
        save_result(s, result("2026-10-24T00:00:00+00:00", []))
        save_result(s, result())
        s.commit()
    app = create_app(engine, risk_model_path=tmp_path / "no-model.joblib")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def empty_client(tmp_path):
    app = create_app(make_engine("sqlite://"), risk_model_path=tmp_path / "no-model.joblib")
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["latest_run_at"].startswith("2026-10-25T00:00:00")
    assert body["risk_model_loaded"] is False


def test_runs(client):
    runs = client.get("/runs").json()
    assert [r["n_conjunctions"] for r in runs] == [4, 0]           # newest first
    assert runs[0]["groups"] == ["starlink", "fengyun-1c-debris"]
    assert client.get("/runs/latest").json()["id"] == runs[0]["id"]
    assert client.get(f"/runs/{runs[1]['id']}").json()["n_conjunctions"] == 0
    assert client.get("/runs/999").status_code == 404


def test_conjunctions_default_is_latest_run_by_pc(client):
    page = client.get("/conjunctions").json()
    assert page["total"] == 3
    assert [c["pc"] for c in page["items"]] == [1e-3, 5e-4, 1e-6]
    assert page["items"][0]["tca"].endswith(("Z", "+00:00"))       # explicit UTC


def test_conjunction_filters_and_validation(client):
    assert client.get("/conjunctions", params={"min_pc": 1e-4}).json()["total"] == 2
    assert client.get("/conjunctions", params={"include_co_orbital": True}).json()["total"] == 4
    assert client.get("/conjunctions", params={"order_by": "miss", "limit": 1}).json()[
        "items"][0]["miss_distance_km"] == 0.1
    assert client.get("/conjunctions", params={"limit": 501}).status_code == 422
    assert client.get("/conjunctions", params={"order_by": "nope"}).status_code == 422
    assert client.get("/conjunctions", params={"run_id": 999}).status_code == 404


def test_single_conjunction_and_object_view(client):
    first = client.get("/conjunctions").json()["items"][0]
    assert client.get(f"/conjunctions/{first['id']}").json() == first
    assert client.get("/conjunctions/999999").status_code == 404
    # Object 1 is in three pairs, one co-orbital: the object view includes all of them.
    assert client.get("/objects/1/conjunctions").json()["total"] == 3
    assert client.get("/conjunctions", params={"norad_id": 1}).json()["total"] == 2


def test_empty_database(empty_client):
    assert empty_client.get("/runs/latest").status_code == 404
    assert empty_client.get("/conjunctions").json()["total"] == 0


def test_risk_forecast_without_model_is_unavailable(client):
    body = {"events": [{"event_id": 1, "cdms": [{"time_to_tca": 3.0, "risk": -5.0}]}]}
    assert client.post("/risk/forecast", json=body).status_code == 503


def test_risk_forecast_rejects_bad_input(client):
    missing = {"events": [{"event_id": 1, "cdms": [{"time_to_tca": 3.0}]}]}
    assert client.post("/risk/forecast", json=missing).status_code == 422
    assert client.post("/risk/forecast", json={"events": []}).status_code == 422


@pytest.mark.skipif(not HAVE_MODEL, reason="trained risk model or ESA dataset not available")
def test_risk_forecast_matches_offline_model(tmp_path):
    """The API must give exactly the predictions of the offline evaluation pipeline."""
    from orbitguard.cdm.data import load_test_task
    from orbitguard.cdm.model import Dataset, RiskModel

    test = load_test_task()
    ids = list(test.targets.index[:25]) + list(test.targets[test.targets >= -6].index[:15])
    cdms = test.inputs[test.inputs.event_id.isin(ids)]
    events = [{"event_id": int(e), "cdms": [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items() if k != "event_id"}
        for row in g.to_dict("records")]} for e, g in cdms.groupby("event_id")]

    app = create_app(make_engine("sqlite://"), risk_model_path=DEFAULT_RISK_MODEL_PATH)
    with TestClient(app) as c:
        resp = c.post("/risk/forecast", json={"events": events})
    assert resp.status_code == 200
    api = pd.Series({f["event_id"]: f["predicted_final_risk"] for f in resp.json()["forecasts"]})

    model = RiskModel.load(DEFAULT_RISK_MODEL_PATH)
    sub = test.inputs[test.inputs.event_id.isin(ids)]
    offline = model.predict(Dataset.from_task(type(test)(sub, test.targets.loc[ids])).X)
    pd.testing.assert_series_equal(api.sort_index(), offline.sort_index(), check_names=False)

    early_only = {"events": [{"event_id": 1, "cdms": [{"time_to_tca": 1.5, "risk": -5.0}]}]}
    with TestClient(app) as c:
        assert c.post("/risk/forecast", json=early_only).status_code == 422


def test_unloadable_model_is_reported_and_api_still_serves(tmp_path, caplog):
    bad = tmp_path / "broken.joblib"
    bad.write_bytes(b"not a pickle")
    app = create_app(make_engine("sqlite://"), risk_model_path=bad)
    with TestClient(app) as c:
        assert c.get("/health").json()["risk_model_loaded"] is False
        assert c.get("/conjunctions").status_code == 200
    assert "Could not load risk model" in caplog.text


def test_pipeline_runs_endpoint(client):
    from datetime import datetime, timezone
    from orbitguard.store import save_pipeline_run
    engine = client.app.state.engine
    with Session(engine) as s:
        save_pipeline_run(s, started_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
                          finished_at=datetime(2026, 10, 27, 0, 5, tzinfo=timezone.utc),
                          status="success", groups="oneweb,starlink",
                          metrics={"conjunctions": 10}, drift_alerts=[{"metric": "x"}])
        save_pipeline_run(s, started_at=datetime(2026, 10, 27, 6, tzinfo=timezone.utc),
                          finished_at=datetime(2026, 10, 27, 6, 1, tzinfo=timezone.utc),
                          status="failed_validation", groups="oneweb", message="stable_count")
        s.commit()
    runs = client.get("/pipeline/runs").json()
    assert [r["status"] for r in runs] == ["failed_validation", "success"]
    assert runs[1]["groups"] == ["oneweb", "starlink"] and runs[1]["metrics"] == {"conjunctions": 10}
    assert runs[1]["drift_alerts"] == [{"metric": "x"}]
    assert len(client.get("/pipeline/runs", params={"status": "success"}).json()) == 1
    assert client.get("/pipeline/runs", params={"status": "nope"}).status_code == 422


def test_run_elements_are_served_compressed_with_populations(tmp_path):
    from orbitguard.propagation import build_catalog, element_sets
    from orbitguard.store import save_run_elements
    from test_orbitguard import EPOCH, make_record

    catalog = build_catalog([make_record(44713, 53.0, 0.0, name="STARLINK-1007"),
                             make_record(30266, 98.0, 90.0, name="FENGYUN 1C DEB")], at=EPOCH)
    engine = make_engine("sqlite://")
    with Session(engine) as s:
        old = save_result(s, result("2026-10-24T00:00:00+00:00", []))
        run = save_result(s, result())
        filler = [{"norad_id": 90000 + i, "name": f"OBJECT {i}", "tle1": "1", "tle2": "2"}
                  for i in range(30)]   # enough bytes for the response to be compressed
        save_run_elements(s, run.id, element_sets(catalog) + filler)
        s.commit()
        old_id, run_id = old.id, run.id
    with TestClient(create_app(engine, risk_model_path=tmp_path / "none.joblib")) as c:
        r = c.get(f"/runs/{run_id}/elements", headers={"Accept-Encoding": "gzip"})
        assert r.status_code == 200 and r.headers["content-encoding"] == "gzip"
        body = r.json()
        assert body["count"] == 32
        assert [o["population"] for o in body["objects"][:3]] == ["starlink", "debris", "other"]
        assert body["objects"][0]["tle1"].startswith("1 44713U")
        assert c.get(f"/runs/{old_id}/elements").status_code == 404   # not kept
        assert c.get("/runs/999/elements").status_code == 404
