"""Command-line entry point: ``python -m orbitguard screen``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from orbitguard.ingest import DEFAULT_CACHE_DIR, fetch_groups
from orbitguard.propagation import build_catalog
from orbitguard.screening import screen


def _cmd_screen(args: argparse.Namespace) -> int:
    start = datetime.now(timezone.utc).replace(microsecond=0)
    records = fetch_groups(args.group, cache_dir=args.cache_dir, refresh=args.refresh)
    catalog = build_catalog(records, at=start, max_perigee_km=args.max_perigee_km)

    t0 = time.perf_counter()
    conjunctions = screen(catalog, start, hours=args.hours, step_s=args.step,
                          threshold_km=args.threshold)
    elapsed = time.perf_counter() - t0

    events = [c for c in conjunctions if not c.persistent]
    persistent = [c for c in conjunctions if c.persistent]
    print(f"\nScreened {len(catalog)} objects over {args.hours:g} h in {elapsed:.1f} s")
    print(f"{len(events)} conjunctions < {args.threshold:g} km "
          f"(+{len(persistent)} persistent pairs, e.g. formation flying)\n")
    print(f"{'TCA (UTC)':<20} {'miss km':>8} {'v_rel km/s':>10}  objects")
    for c in events[:args.top]:
        print(f"{c.tca:%Y-%m-%d %H:%M:%S} {c.miss_distance_km:8.3f} {c.relative_speed_km_s:10.2f}  "
              f"{c.name_1} ({c.norad_id_1}) x {c.name_2} ({c.norad_id_2})")

    out = args.out or Path("data/processed") / f"conjunctions_{start:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": start.isoformat(),
        "groups": args.group,
        "objects_screened": len(catalog),
        "window_hours": args.hours,
        "threshold_km": args.threshold,
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
