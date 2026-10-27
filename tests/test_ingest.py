import json
from datetime import datetime, timezone

import pytest
import requests

from orbitguard import ingest
from test_orbitguard import make_record

NOW = datetime(2026, 10, 27, 12, tzinfo=timezone.utc)
GOOD = json.dumps([make_record(1, 53.0, 0.0), make_record(2, 53.0, 10.0)])


class FakeResponse:
    def __init__(self, text, status=200):
        self.text, self.status_code, self.content = text, status, text.encode()

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def server(monkeypatch):
    """Queue of responses (or exceptions) returned by successive requests.get calls."""
    queue = []

    def fake_get(*args, **kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(ingest.requests, "get", fake_get)
    return queue


def fetch(tmp_path, sleeps=None):
    return ingest.fetch_group("g", cache_dir=tmp_path, now=NOW,
                              sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_truncated_download_is_retried(server, tmp_path):
    server += [FakeResponse(GOOD[:150]), FakeResponse(GOOD)]   # cut off mid-array, then fine
    sleeps = []
    assert len(fetch(tmp_path, sleeps)) == 2
    assert sleeps == [ingest.RETRY_DELAYS_S[0]]
    assert len(list(tmp_path.glob("gp_g_*.json"))) == 1         # only the good download cached


def test_connection_errors_are_retried(server, tmp_path):
    server += [requests.ConnectionError("reset"), requests.Timeout("slow"), FakeResponse(GOOD)]
    sleeps = []
    assert len(fetch(tmp_path, sleeps)) == 2
    assert sleeps == list(ingest.RETRY_DELAYS_S)


def test_gives_up_with_a_clear_message(server, tmp_path):
    server += [FakeResponse(GOOD[:150])] * 3
    with pytest.raises(RuntimeError, match=r"after 3 attempts: download truncated after 150 bytes"):
        fetch(tmp_path)
    assert list(tmp_path.glob("gp_g_*.json")) == []             # nothing half-written cached


def test_unknown_group_is_not_retried(server, tmp_path):
    server += [FakeResponse('Invalid query: "GROUP=g" (GROUP=g not found)')]
    with pytest.raises(ValueError, match="no JSON"):
        fetch(tmp_path)
    assert server == []                                          # exactly one request


def test_rate_limit_falls_back_to_cache(server, tmp_path):
    (tmp_path / "gp_g_20261027T050000Z.json").write_text(GOOD)  # 7 h old: too old to reuse
    server += [FakeResponse("Forbidden", status=403)]
    assert len(fetch(tmp_path)) == 2
