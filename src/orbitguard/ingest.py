"""Fetch General Perturbations (GP) element sets from CelesTrak, with local caching.

CelesTrak updates GP data roughly every 2 hours and asks clients not to download
the same data more often than that, so every fetch goes through a file cache.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_CACHE_DIR = Path("data/raw")
MIN_REFRESH = timedelta(hours=2)
# Waits before the 2nd and 3rd download attempts (flaky connections, truncated transfers).
RETRY_DELAYS_S = (10.0, 30.0)

REQUIRED_FIELDS = (
    "OBJECT_NAME", "OBJECT_ID", "EPOCH", "MEAN_MOTION", "ECCENTRICITY",
    "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY",
    "NORAD_CAT_ID", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
)


def validate_records(records: list[dict]) -> list[dict]:
    """Drop records missing required fields or with physically invalid elements."""
    valid = []
    for rec in records:
        if any(rec.get(f) in (None, "") for f in REQUIRED_FIELDS):
            continue
        try:
            ecc = float(rec["ECCENTRICITY"])
            mean_motion = float(rec["MEAN_MOTION"])
            datetime.strptime(rec["EPOCH"], "%Y-%m-%dT%H:%M:%S.%f")
        except (TypeError, ValueError):
            continue
        if not (0.0 <= ecc < 1.0) or mean_motion <= 0.0:
            continue
        valid.append(rec)
    dropped = len(records) - len(valid)
    if dropped:
        log.warning("Dropped %d invalid GP records", dropped)
    return valid


def _cache_files(cache_dir: Path, group: str) -> list[Path]:
    return sorted(cache_dir.glob(f"gp_{group}_*.json"))


def fetch_group(
    group: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    now: datetime | None = None,
    sleep=time.sleep,
) -> list[dict]:
    """Return validated GP records for a CelesTrak group, using the cache when fresh.

    Dropped connections, timeouts and truncated downloads are retried; a plain-text
    answer (unknown group) or a rate-limit refusal is not.
    """
    now = now or datetime.now(timezone.utc)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = _cache_files(cache_dir, group)
    if cached and not refresh:
        newest = cached[-1]
        fetched_at = datetime.strptime(newest.stem.rsplit("_", 1)[1], "%Y%m%dT%H%M%SZ")
        if now - fetched_at.replace(tzinfo=timezone.utc) < MIN_REFRESH:
            log.info("Using cached %s (%s)", group, newest.name)
            return validate_records(json.loads(newest.read_text()))

    records = None
    for attempt in range(1, len(RETRY_DELAYS_S) + 2):
        log.info("Downloading CelesTrak group %r (attempt %d)", group, attempt)
        try:
            resp = requests.get(
                CELESTRAK_GP_URL, params={"GROUP": group, "FORMAT": "json"}, timeout=60
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            problem = f"connection failed: {exc}"
        else:
            if resp.status_code == 403:
                # CelesTrak blocks clients that re-download unchanged data too often.
                if cached:
                    log.warning("CelesTrak refused %r (rate limit); using stale cache %s",
                                group, cached[-1].name)
                    return validate_records(json.loads(cached[-1].read_text()))
                raise RuntimeError(
                    f"CelesTrak refused group {group!r} (403). It rate-limits repeat downloads; "
                    "wait up to 2 hours and retry."
                )
            resp.raise_for_status()
            try:
                records = resp.json()
                break
            except ValueError as exc:
                if not resp.text.lstrip().startswith("["):
                    # CelesTrak answers unknown groups with a plain-text message: not retryable.
                    raise ValueError(
                        f"CelesTrak returned no JSON for group {group!r}: {resp.text[:200]}") from exc
                # A JSON array that does not parse was cut off mid-transfer.
                problem = f"download truncated after {len(resp.content):,} bytes"
        if attempt <= len(RETRY_DELAYS_S):
            log.warning("Group %r: %s; retrying in %g s", group, problem, RETRY_DELAYS_S[attempt - 1])
            sleep(RETRY_DELAYS_S[attempt - 1])
    if records is None:
        raise RuntimeError(f"Could not download group {group!r} after "
                           f"{len(RETRY_DELAYS_S) + 1} attempts: {problem}")

    path = cache_dir / f"gp_{group}_{now:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(records))
    return validate_records(records)


def fetch_groups(groups: list[str], **kwargs) -> list[dict]:
    """Fetch several groups and de-duplicate objects by NORAD catalog number."""
    by_id: dict[int, dict] = {}
    for group in groups:
        for rec in fetch_group(group, **kwargs):
            by_id[int(rec["NORAD_CAT_ID"])] = rec
    return list(by_id.values())
