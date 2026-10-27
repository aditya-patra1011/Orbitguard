"""Settings and middleware that make the API safe to expose publicly.

Everything is configured by environment variables and is permissive by default, so local
use needs no setup. ``ORBITGUARD_PUBLIC=1`` turns the checks into hard requirements:
the API refuses to start without strong API keys and a non-default database password.

- ``ORBITGUARD_API_KEYS``: comma-separated keys accepted in the ``X-API-Key`` header by
  endpoints that run the model. Unset: those endpoints are open (local use only).
- ``ORBITGUARD_RATE_LIMIT``: requests per minute per client (default 120); endpoints
  marked heavy get a tenth of that.
- ``ORBITGUARD_TRUST_PROXY=1``: take the client address from ``X-Forwarded-For``, as
  set by the hosting platform's proxy. Only enable it behind such a proxy, or clients
  could pick their own address and dodge the rate limit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from sqlalchemy.engine import make_url

log = logging.getLogger(__name__)

MIN_KEY_LENGTH = 32
DEFAULT_DB_PASSWORD = "orbitguard"
# Largest accepted request body. A real CDM event is ~45 KB of JSON (at most 54 KB in the
# ESA test set), so this fits a full forecast request of 100 events with room to spare.
MAX_BODY_BYTES = 6_000_000

# Pages are a static dashboard plus CDN libraries pinned by version and integrity hash.
# Cesium needs blob: workers and inline styles, and its widgets (clock, timeline) compile
# their templates with `new Function` and it loads WebAssembly, hence the two eval
# sources: checked in the browser, the globe does not render without them. No inline
# scripts are allowed, and the page only ever inserts API data as text. Swagger UI
# (/docs) is left alone because it relies on an inline script.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval' https://cdn.jsdelivr.net",
    "worker-src 'self' blob: https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "img-src 'self' data: blob: https://cdn.jsdelivr.net",
    "connect-src 'self' https://cdn.jsdelivr.net",
    "font-src 'self' data: https://cdn.jsdelivr.net",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


@dataclass
class SecuritySettings:
    public: bool = False
    api_keys: tuple[str, ...] = ()
    rate_limit_per_min: int = 120
    trust_proxy: bool = False

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        keys = tuple(k.strip() for k in os.environ.get("ORBITGUARD_API_KEYS", "").split(",") if k.strip())
        return cls(public=_flag("ORBITGUARD_PUBLIC"), api_keys=keys,
                   rate_limit_per_min=int(os.environ.get("ORBITGUARD_RATE_LIMIT", "120")),
                   trust_proxy=_flag("ORBITGUARD_TRUST_PROXY"))

    def check(self, db_url: str) -> None:
        """In public mode, refuse to run with settings that are only fit for a laptop."""
        problems = []
        if self.public:
            if not self.api_keys:
                problems.append("ORBITGUARD_API_KEYS is not set")
            elif any(len(k) < MIN_KEY_LENGTH for k in self.api_keys):
                problems.append(f"every API key must be at least {MIN_KEY_LENGTH} characters")
            password = make_url(db_url).password
            if password is not None and password in ("", DEFAULT_DB_PASSWORD):
                problems.append("the database uses the default password")
            if self.rate_limit_per_min <= 0:
                problems.append("rate limiting is disabled")
        if problems:
            raise RuntimeError("Refusing to start in public mode: " + "; ".join(problems))
        if not self.api_keys:
            log.warning("No API keys configured: model endpoints are open (fine for local use only)")


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                               description="Required when the server has API keys configured")


def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> None:
    """FastAPI dependency: the X-API-Key header must match a configured key."""
    keys = request.app.state.security.api_keys
    if not keys:
        return
    # Compare digests in constant time, so response timing reveals nothing about the keys.
    digest = hashlib.sha256((api_key or "").encode()).digest()
    if not any(hmac.compare_digest(digest, hashlib.sha256(k.encode()).digest()) for k in keys):
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "API-Key"})


def client_address(scope: dict, trust_proxy: bool) -> str:
    if trust_proxy:
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-for":
                # The platform's proxy appends the address it saw: the right-most entry.
                return value.decode("latin-1").split(",")[-1].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


@dataclass
class RateLimiter:
    """Fixed one-minute windows per client and bucket, in memory.

    Per process: enough for a single-container deployment. Several replicas would
    need a shared store such as Redis.
    """
    per_minute: int
    _counts: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, client: str, heavy: bool, now: float | None = None) -> tuple[bool, int]:
        """(allowed, seconds until the window resets)."""
        now = time.time() if now is None else now
        window = int(now // 60)
        limit = max(1, self.per_minute // 10) if heavy else self.per_minute
        key = (client, heavy)
        with self._lock:
            if len(self._counts) > 10_000:   # drop finished windows now and then
                self._counts = {k: v for k, v in self._counts.items() if v[0] == window}
            start, count = self._counts.get(key, (window, 0))
            if start != window:
                start, count = window, 0
            count += 1
            self._counts[key] = (start, count)
        return count <= limit, 60 - int(now % 60)


def is_heavy(path: str, method: str) -> bool:
    """Model calls and the ~1 MB element-set download get the tighter limit."""
    return (method == "POST" and path.startswith("/risk/")) or path.endswith("/elements")


class SecurityMiddleware:
    """Pure ASGI middleware: rate limits, request-size limit and security headers."""

    def __init__(self, app, settings: SecuritySettings):
        self.app = app
        self.settings = settings
        self.limiter = RateLimiter(settings.rate_limit_per_min) if settings.rate_limit_per_min > 0 else None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]

        if self.limiter is not None and path != "/health":
            ok, retry = self.limiter.allow(client_address(scope, self.settings.trust_proxy),
                                           is_heavy(path, method))
            if not ok:
                return await _reject(send, 429, "Too many requests; slow down", {"retry-after": str(retry)})

        if method in ("POST", "PUT", "PATCH"):
            length = dict(scope["headers"]).get(b"content-length")
            if length is not None and int(length) > MAX_BODY_BYTES:
                return await _reject(send, 413, f"Request body over {MAX_BODY_BYTES} bytes")
            receive = _limited(receive)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [(b"x-content-type-options", b"nosniff"),
                            (b"referrer-policy", b"strict-origin-when-cross-origin")]
                if not path.startswith(DOCS_PATHS):
                    headers += [(b"content-security-policy", CONTENT_SECURITY_POLICY.encode()),
                                (b"x-frame-options", b"DENY")]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except _BodyTooLarge:
            await _reject(send, 413, f"Request body over {MAX_BODY_BYTES} bytes")


class _BodyTooLarge(Exception):
    pass


def _limited(receive):
    """Count body bytes as they arrive: chunked uploads have no Content-Length to check."""
    seen = 0

    async def wrapped():
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > MAX_BODY_BYTES:
                raise _BodyTooLarge
        return message
    return wrapped


async def _reject(send, status: int, detail: str, extra: dict | None = None):
    body = json.dumps({"detail": detail}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers += [(k.encode(), v.encode()) for k, v in (extra or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
