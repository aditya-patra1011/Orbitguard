"""Command-line entry point: ``python -m orbitguard screen``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from orbitguard.covariance import DEFAULT_MODEL, CovarianceModel
from orbitguard.ingest import DEFAULT_CACHE_DIR, fetch_groups
from orbitguard.propagation import build_catalog
from orbitguard.screening import screen

DEFAULT_COVARIANCE_PATH = Path("data/models/covariance.json")


def _load_covariance(path: Path | None) -> CovarianceModel:
    """Explicit path, else a calibrated model if one exists, else the assumed default."""
    if path is not None:
        return CovarianceModel.load(path)
    if DEFAULT_COVARIANCE_PATH.exists():
        return CovarianceModel.load(DEFAULT_COVARIANCE_PATH)
    return DEFAULT_MODEL


def _cmd_screen(args: argparse.Namespace) -> int:
    start = datetime.now(timezone.utc).replace(microsecond=0)
    records = fetch_groups(args.group, cache_dir=args.cache_dir, refresh=args.refresh)
    catalog = build_catalog(records, at=start, max_perigee_km=args.max_perigee_km)
    cov_model = _load_covariance(args.covariance)
    hbr_km = args.hbr_m / 1000.0

    t0 = time.perf_counter()
    conjunctions = screen(catalog, start, hours=args.hours, step_s=args.step,
                          threshold_km=args.threshold, cov_model=cov_model, hbr_km=hbr_km)
    elapsed = time.perf_counter() - t0

    events = [c for c in conjunctions if not c.co_orbital]
    if args.sort == "pc":
        events.sort(key=lambda c: c.pc, reverse=True)
    co_orbital = [c for c in conjunctions if c.co_orbital]
    print(f"\nScreened {len(catalog)} objects over {args.hours:g} h in {elapsed:.1f} s")
    print(f"{len(events)} conjunctions < {args.threshold:g} km "
          f"(+{len(co_orbital)} co-orbital pairs flying in formation)")
    print(f"Pc: covariance '{cov_model.source}', hard-body radius {args.hbr_m:g} m; "
          f"{sum(c.pc >= 1e-4 for c in events)} events with Pc >= 1e-4\n")
    print(f"{'TCA (UTC)':<20} {'miss km':>8} {'v_rel km/s':>10} {'Pc':>9}  objects")
    for c in events[:args.top]:
        print(f"{c.tca:%Y-%m-%d %H:%M:%S} {c.miss_distance_km:8.3f} {c.relative_speed_km_s:10.2f} "
              f"{c.pc:9.2e}  "
              f"{c.name_1} ({c.norad_id_1}) x {c.name_2} ({c.norad_id_2})")

    out = args.out or Path("data/processed") / f"conjunctions_{start:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": start.isoformat(),
        "groups": args.group,
        "objects_screened": len(catalog),
        "window_hours": args.hours,
        "threshold_km": args.threshold,
        "hard_body_radius_m": args.hbr_m,
        "covariance_model": asdict(cov_model),
        "conjunctions": [c.to_dict() for c in conjunctions],
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitguard")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("screen", help="Screen the catalog for close approaches")
    p.add_argument("--group", action="append",
                   help="CelesTrak group (repeatable). Default: active")
    p.add_argument("--hours", type=float, default=24.0, help="Screening window length")
    p.add_argument("--step", type=float, default=10.0, help="Coarse time step in seconds")
    p.add_argument("--threshold", type=float, default=5.0, help="Miss distance threshold in km")
    p.add_argument("--max-perigee-km", type=float, default=None,
                   help="Only screen objects with perigee below this altitude")
    p.add_argument("--hbr-m", type=float, default=10.0,
                   help="Combined hard-body radius in metres (default 10)")
    p.add_argument("--covariance", type=Path, default=None,
                   help=f"Covariance model JSON (default: {DEFAULT_COVARIANCE_PATH} if present, "
                        "else assumed values)")
    p.add_argument("--sort", choices=["pc", "miss"], default="pc",
                   help="Rank conjunctions by Pc (default) or miss distance")
    p.add_argument("--top", type=int, default=20, help="Rows to print")
    p.add_argument("--refresh", action="store_true", help="Ignore the 2-hour data cache")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=_cmd_screen)

    args = parser.parse_args(argv)
    if args.command == "screen" and not args.group:
        args.group = ["active"]
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)
