import json
from datetime import datetime, timedelta, timezone

import pytest

from orbitguard.validation import Thresholds, validate_latest, validate_snapshot
from test_orbitguard import EPOCH, make_record

NOW = EPOCH + timedelta(hours=6)


def snapshot(n, start_id=1, **overrides):
    return [{**make_record(start_id + i, 53.0, float(i)), **overrides} for i in range(n)]


def failed(report):
    return {c.name for c in report.checks if not c.passed}


def test_clean_snapshot_passes():
    r = validate_snapshot(snapshot(200), "g", NOW, previous=snapshot(190))
    assert r.ok and not r.warnings and failed(r) == set()


def test_empty_download_is_an_error():
    r = validate_snapshot([], "g", NOW)
    assert not r.ok and failed(r) == {"not_empty"}


def test_truncated_download_is_caught_by_count_change():
    r = validate_snapshot(snapshot(50), "g", NOW, previous=snapshot(200))
    assert not r.ok and "stable_count" in failed(r)
    assert "-75.0%" in next(c.detail for c in r.checks if c.name == "stable_count")


def test_corrupt_records_over_the_limit_are_an_error():
    recs = snapshot(100)
    for r in recs[:5]:
        r["ECCENTRICITY"] = 1.5                       # impossible for an orbit
    recs[5].pop("MEAN_MOTION")
    report = validate_snapshot(recs, "g", NOW)
    assert not report.ok and "invalid_records" in failed(report)
    assert "6 invalid" in next(c.detail for c in report.checks if c.name == "invalid_records")


def test_a_few_bad_records_within_tolerance_pass():
    recs = snapshot(200)
    recs[0]["MEAN_MOTION"] = 40.0                     # faster than any real orbit
    assert validate_snapshot(recs, "g", NOW).ok       # 0.5% <= 1% limit


def test_future_epochs_are_an_error():
    future = (NOW + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S.%f")
    r = validate_snapshot(snapshot(10, EPOCH=future), "g", NOW)
    assert not r.ok and "no_future_epochs" in failed(r)


def test_stale_data_is_a_warning_not_an_error():
    r = validate_snapshot(snapshot(10), "g", NOW + timedelta(days=20))
    assert r.ok
    assert {"fresh_data", "few_stale_elements"} <= {c.name for c in r.warnings}


def test_duplicate_ids_are_a_warning():
    r = validate_snapshot(snapshot(5) + snapshot(1), "g", NOW)
    assert r.ok and [c.name for c in r.warnings] == ["unique_ids"]


def test_thresholds_are_configurable():
    r = validate_snapshot(snapshot(150), "g", NOW, previous=snapshot(100),
                          thresholds=Thresholds(max_count_change=0.6))
    assert r.ok


def test_validate_latest_compares_with_previous_cached_snapshot(tmp_path):
    (tmp_path / "gp_g_20261019T100000Z.json").write_text(json.dumps(snapshot(200)))
    (tmp_path / "gp_g_20261019T160000Z.json").write_text(json.dumps(snapshot(20)))
    r = validate_latest("g", cache_dir=tmp_path)
    assert r.snapshot == "gp_g_20261019T160000Z.json"
    assert "stable_count" in failed(r)
    assert validate_latest("missing", cache_dir=tmp_path).ok is False
