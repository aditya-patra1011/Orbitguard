import pytest
from fastapi.testclient import TestClient

from orbitguard.api import create_app
from orbitguard.security import (
    MAX_BODY_BYTES, RateLimiter, SecuritySettings, client_address, is_heavy,
)
from orbitguard.store import make_engine

KEY = "k" * 40
FORECAST = {"events": [{"event_id": 1, "cdms": [{"time_to_tca": 3, "risk": -5}]}]}


def app_client(tmp_path, **settings):
    app = create_app(make_engine("sqlite://"), risk_model_path=tmp_path / "none.joblib",
                     security=SecuritySettings(**settings))
    return TestClient(app)


def test_forecast_needs_a_valid_key_when_keys_are_configured(tmp_path):
    with app_client(tmp_path, api_keys=(KEY,)) as c:
        assert c.post("/risk/forecast", json=FORECAST).status_code == 401
        assert c.post("/risk/forecast", json=FORECAST, headers={"X-API-Key": "wrong"}).status_code == 401
        # A valid key gets through authentication (503 only because no model is loaded here).
        assert c.post("/risk/forecast", json=FORECAST, headers={"X-API-Key": KEY}).status_code == 503
        assert c.get("/runs").status_code == 200        # reading stays public


def test_without_keys_the_forecast_stays_open_for_local_use(tmp_path):
    with app_client(tmp_path) as c:
        assert c.post("/risk/forecast", json=FORECAST).status_code == 503


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    with app_client(tmp_path, rate_limit_per_min=3) as c:
        codes = [c.get("/runs").status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        blocked = c.get("/runs")
        assert 0 < int(blocked.headers["retry-after"]) <= 60
        assert c.get("/health").status_code == 200      # health checks are never limited


def test_heavy_endpoints_get_a_tenth_of_the_limit():
    limiter = RateLimiter(per_minute=20)
    heavy = [limiter.allow("1.2.3.4", heavy=True, now=120.0)[0] for _ in range(3)]
    assert heavy == [True, True, False]                # 20 // 10 = 2 per minute
    assert limiter.allow("1.2.3.4", heavy=False, now=120.0)[0]   # separate bucket
    assert limiter.allow("1.2.3.4", heavy=True, now=180.0)[0]    # next minute resets
    assert is_heavy("/runs/7/elements", "GET") and is_heavy("/risk/forecast", "POST")
    assert not is_heavy("/conjunctions", "GET")


def test_proxy_header_is_only_trusted_when_configured():
    scope = {"client": ("10.0.0.1", 5000),
             "headers": [(b"x-forwarded-for", b"6.6.6.6, 203.0.113.9")]}
    assert client_address(scope, trust_proxy=False) == "10.0.0.1"
    assert client_address(scope, trust_proxy=True) == "203.0.113.9"   # appended by our proxy


def test_oversized_bodies_are_refused(tmp_path):
    with app_client(tmp_path) as c:
        big = b"x" * (MAX_BODY_BYTES + 1)
        r = c.post("/risk/forecast", content=big, headers={"content-type": "application/json"})
        assert r.status_code == 413


def test_security_headers_on_pages_but_docs_still_work(tmp_path):
    with app_client(tmp_path) as c:
        page = c.get("/")
        csp = page.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
        assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]   # no inline scripts
        assert page.headers["x-frame-options"] == "DENY"
        assert page.headers["x-content-type-options"] == "nosniff"
        docs = c.get("/docs")
        assert docs.status_code == 200 and "content-security-policy" not in docs.headers


def test_dashboard_has_no_inline_scripts_and_pins_cdn_files(tmp_path):
    with app_client(tmp_path) as c:
        html = c.get("/").text
    assert "<script>" not in html                       # CSP would block it
    assert html.count('integrity="sha384-') == 3        # Cesium, its CSS and satellite.js


@pytest.mark.parametrize("settings, db_url, problem", [
    (dict(public=True), "postgresql://u:" + "p" * 40 + "@db/x", "ORBITGUARD_API_KEYS"),
    (dict(public=True, api_keys=("short",)), "sqlite://", "at least 32"),
    (dict(public=True, api_keys=(KEY,)), "postgresql://orbitguard:orbitguard@db/x", "default password"),
    (dict(public=True, api_keys=(KEY,), rate_limit_per_min=0), "sqlite://", "rate limiting"),
])
def test_public_mode_refuses_weak_settings(settings, db_url, problem):
    with pytest.raises(RuntimeError, match=problem):
        SecuritySettings(**settings).check(db_url)


def test_public_mode_starts_with_strong_settings():
    SecuritySettings(public=True, api_keys=(KEY,)).check("postgresql://u:" + "p" * 40 + "@db/x")


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ORBITGUARD_PUBLIC", "1")
    monkeypatch.setenv("ORBITGUARD_API_KEYS", f"{KEY}, {'z' * 40} ,")
    monkeypatch.setenv("ORBITGUARD_RATE_LIMIT", "30")
    s = SecuritySettings.from_env()
    assert s.public and s.api_keys == (KEY, "z" * 40) and s.rate_limit_per_min == 30
    assert not s.trust_proxy
