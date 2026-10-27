import json

from sqlalchemy.orm import Session

from orbitguard.site import export_site
from orbitguard.store import make_engine, save_pipeline_run, save_result, save_run_elements
from test_store import result


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_dashboard_and_latest_run_data(tmp_path):
    engine = make_engine("sqlite://")
    with Session(engine) as s:
        save_result(s, result("2026-10-24T00:00:00+00:00", []))
        run = save_result(s, result())
        save_run_elements(s, run.id, [{"norad_id": 1, "name": "STARLINK-1", "tle1": "1 x", "tle2": "2 x"}])
        save_pipeline_run(s, started_at=run.generated_at, finished_at=run.generated_at, status="success",
                          groups="starlink", run_id=run.id, metrics={"conjunctions": 4})
        s.commit()
        run_id = run.id

    out = tmp_path / "site"
    summary = export_site(engine, out, top=2)

    for name in ("index.html", "app.js", "worker.js", "style.css", "config.js", ".nojekyll"):
        assert (out / name).exists()
    assert summary["run"]["id"] == run_id and summary["pipeline"]["status"] == "success"
    assert summary["counts"] == {"1e-4": 2, "1e-6": 3, "0": 3}   # co-orbital pair excluded
    assert _load(out / "data" / "summary.json") == summary

    conj = _load(out / "data" / "conjunctions.json")
    assert conj["total_with_pc"] == 3
    assert [c["pc"] for c in conj["items"]] == [1e-3, 5e-4]       # top 2 by Pc
    elements = _load(out / "data" / "elements.json")
    assert elements["count"] == 1 and elements["objects"][0]["population"] == "starlink"

    assert 'window.ORBITGUARD_DATA = "static"' in (out / "config.js").read_text(encoding="utf-8")
    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'http-equiv="Content-Security-Policy"' in html and "frame-ancestors" not in html


def test_export_with_no_runs_still_writes_a_site(tmp_path):
    summary = export_site(make_engine("sqlite://"), tmp_path / "site")
    assert summary["run"] is None and summary["pipeline"] is None
    assert (tmp_path / "site" / "index.html").exists()
    assert not (tmp_path / "site" / "data" / "elements.json").exists()


def test_export_replaces_an_old_site(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    export_site(make_engine("sqlite://"), out)
    assert not (out / "stale.txt").exists()
