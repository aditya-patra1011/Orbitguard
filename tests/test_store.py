import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from orbitguard.store import (
    ConjunctionQuery, get_run_elements, import_files, latest_run, list_runs, make_engine,
    query_conjunctions, save_result, save_run_elements,
)


def conj(i, pc, miss, tca_hour, co_orbital=False, ids=(1, 2)):
    return {
        "norad_id_1": ids[0], "name_1": f"OBJ-{ids[0]}", "norad_id_2": ids[1] + i,
        "name_2": f"OBJ-{ids[1] + i}", "tca": f"2026-10-25T{tca_hour:02d}:00:00+00:00",
        "miss_distance_km": miss, "relative_speed_km_s": 10.0, "co_orbital": co_orbital, "pc": pc,
    }


def result(generated_at="2026-10-25T00:00:00+00:00", conjunctions=None):
    return {
        "generated_at": generated_at, "groups": ["starlink", "fengyun-1c-debris"],
        "objects_screened": 100, "window_hours": 24.0, "threshold_km": 5.0,
        "hard_body_radius_m": 10.0, "covariance_model": {"source": "test model"},
        "conjunctions": conjunctions if conjunctions is not None else [
            conj(0, 1e-3, 0.2, 5), conj(1, 1e-6, 0.1, 3), conj(2, None, 2.0, 1, co_orbital=True),
            conj(3, 5e-4, 1.5, 8, ids=(7, 8)),
        ],
    }


@pytest.fixture
def session():
    with Session(make_engine("sqlite://")) as s:
        yield s


def test_save_and_read_back_a_run(session):
    run = save_result(session, result())
    assert run.id is not None and run.n_conjunctions == 4
    assert run.groups == "starlink,fengyun-1c-debris" and run.covariance_source == "test model"
    assert latest_run(session).id == run.id


def test_latest_run_is_the_default_and_newest_wins(session):
    save_result(session, result("2026-10-24T00:00:00+00:00", [conj(0, 0.5, 0.1, 1)]))
    save_result(session, result("2026-10-25T00:00:00+00:00"))
    total, rows = query_conjunctions(session, ConjunctionQuery())
    assert total == 3                      # co-orbital pair excluded by default
    assert [r.pc for r in rows] == [1e-3, 5e-4, 1e-6]   # ordered by Pc, highest first
    assert len(list_runs(session)) == 2


def test_filters(session):
    save_result(session, result())
    q = lambda **kw: query_conjunctions(session, ConjunctionQuery(**kw))  # noqa: E731
    assert q(min_pc=1e-4)[0] == 2
    assert q(max_miss_km=0.5)[0] == 2
    assert q(norad_id=7)[0] == 1 and q(norad_id=11)[0] == 1    # either side (7 x 8+3)
    assert q(include_co_orbital=True)[0] == 4
    assert q(tca_from=datetime(2026, 10, 25, 4, tzinfo=timezone.utc))[0] == 2
    assert [r.miss_distance_km for r in q(order_by="miss")[1]] == [0.1, 0.2, 1.5]


def test_pagination_reports_the_full_total(session):
    save_result(session, result())
    total, rows = query_conjunctions(session, ConjunctionQuery(limit=1, offset=1))
    assert total == 3 and len(rows) == 1 and rows[0].pc == 5e-4


def test_empty_database_returns_nothing(session):
    assert query_conjunctions(session, ConjunctionQuery()) == (0, [])
    assert latest_run(session) is None


def test_import_files_skips_duplicates_and_reads_phase1_format(session, tmp_path):
    old = result("2026-10-19T21:29:14+00:00", [conj(0, None, 0.03, 2)])
    old["conjunctions"][0].pop("co_orbital")
    old["conjunctions"][0]["persistent"] = True        # Phase 1 key
    del old["hard_body_radius_m"], old["covariance_model"]
    paths = []
    for i, r in enumerate([old, result()]):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(r))
        paths.append(p)
    assert len(import_files(session, paths)) == 2
    assert import_files(session, paths) == []          # already stored
    phase1 = min(list_runs(session), key=lambda r: r.generated_at)
    assert phase1.conjunctions[0].co_orbital is True and phase1.hard_body_radius_m is None


def test_element_sets_are_kept_only_for_the_latest_runs():
    with Session(make_engine("sqlite://")) as session:
        ids = []
        for day in (20, 21, 22):
            run = save_result(session, result(f"2026-10-{day}T00:00:00+00:00", []))
            save_run_elements(session, run.id, [{"norad_id": day, "name": "X", "tle1": "1", "tle2": "2"}],
                              keep_runs=2)
            ids.append(run.id)
        assert get_run_elements(session, ids[0]) is None           # pruned
        assert get_run_elements(session, ids[2])[0]["norad_id"] == 22
        assert len(list_runs(session)) == 3                         # runs themselves are kept
