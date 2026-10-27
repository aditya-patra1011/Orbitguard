"""Command-line entry point: ``python -m orbitguard screen | calibrate | risk | db | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orbitguard.cdm.cli import add_risk_parser
from orbitguard.calibration import (
    Residuals, compute_residuals, fit_populations, load_snapshot_pairs,
)
from orbitguard.covariance import (
    DEFAULT_MODEL, CovarianceModel, PopulationCovariance, load_covariance,
)
from orbitguard.ingest import DEFAULT_CACHE_DIR, fetch_groups
from orbitguard.propagation import build_catalog
from orbitguard.screening import screen

DEFAULT_COVARIANCE_PATH = Path("data/models/covariance.json")
DB_URL_HELP = "SQLAlchemy database URL (default: $ORBITGUARD_DB_URL or sqlite:///data/orbitguard.db)"


def _load_covariance(path: Path | None) -> CovarianceModel | PopulationCovariance:
    """Explicit path, else a calibrated model if one exists, else the assumed default."""
    if path is not None:
        return load_covariance(path)
    if DEFAULT_COVARIANCE_PATH.exists():
        return load_covariance(DEFAULT_COVARIANCE_PATH)
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

    result = {
        "generated_at": start.isoformat(),
        "groups": args.group,
        "objects_screened": len(catalog),
        "window_hours": args.hours,
        "threshold_km": args.threshold,
        "hard_body_radius_m": args.hbr_m,
        "covariance_model": asdict(cov_model),
        "conjunctions": [c.to_dict() for c in conjunctions],
    }
    out = args.out or Path("data/processed") / f"conjunctions_{start:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")

    if args.store:
        from sqlalchemy.orm import Session

        from orbitguard.store import make_engine, save_result

        with Session(make_engine(args.db_url)) as session:
            run = save_result(session, result)
            session.commit()
            print(f"Stored as run {run.id} in {session.get_bind().url.render_as_string(hide_password=True)}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from orbitguard.validation import validate_latest

    groups = args.group or sorted({p.stem[len("gp_"):].rsplit("_", 1)[0]
                                   for p in args.cache_dir.glob("gp_*_*.json")})
    if not groups:
        print(f"No cached snapshots in {args.cache_dir}")
        return 1
    failed = False
    for group in groups:
        report = validate_latest(group, cache_dir=args.cache_dir)
        status = "OK" if report.ok and not report.warnings else ("WARN" if report.ok else "FAIL")
        print(f"[{status:4}] {group:<22} {report.n_records:>6} records  {report.snapshot}")
        for c in report.checks:
            if not c.passed or args.verbose_checks:
                mark = "pass" if c.passed else c.severity
                print(f"         {mark:<7} {c.name}: {c.detail}")
        failed |= not report.ok
    return 1 if failed else 0


def _cmd_db_import(args: argparse.Namespace) -> int:
    from sqlalchemy.orm import Session

    from orbitguard.store import import_files, make_engine

    paths = args.paths or sorted(Path("data/processed").glob("conjunctions_*.json"))
    with Session(make_engine(args.db_url)) as session:
        runs = import_files(session, paths)
        session.commit()
        print(f"Imported {len(runs)} new run(s) "
              f"({sum(r.n_conjunctions for r in runs)} conjunctions) from {len(paths)} file(s)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    # The app reads its configuration from the environment when it starts.
    if args.db_url:
        os.environ["ORBITGUARD_DB_URL"] = args.db_url
    if args.risk_model:
        os.environ["ORBITGUARD_RISK_MODEL"] = str(args.risk_model)
    print(f"OrbitGuard API on http://{args.host}:{args.port}  (docs: /docs)")
    uvicorn.run("orbitguard.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    pairs = load_snapshot_pairs(args.cache_dir)
    if not pairs:
        print(f"Need at least two cached snapshots of a group in {args.cache_dir}; "
              "run `orbitguard screen` again after CelesTrak publishes new data.")
        return 1
    parts = [compute_residuals(old, new) for old, new in pairs]
    res = Residuals(
        norad_ids=np.concatenate([p.norad_ids for p in parts]),
        names=[n for p in parts for n in p.names],
        age_days=np.concatenate([p.age_days for p in parts]),
        rtn_km=np.concatenate([p.rtn_km for p in parts]),
    )
    today = datetime.now(timezone.utc)
    model, summaries = fit_populations(
        res, source=f"calibrated {today:%Y-%m-%d} from {len(res.age_days)} element-set pairs")

    pooled = summaries["pooled"]
    print(f"\n{pooled['residuals']} residuals from {len(pairs)} snapshot pairs, "
          f"ages {pooled['age_days_range'][0]:.2f}-{pooled['age_days_range'][1]:.2f} days\n")
    print(f"{'population':<10} {'n':>6} {'outliers':>8}   {'sigma0 R,T,N (km)':<24} "
          f"{'growth R,T,N (km/day)':<24} {'sigma T @ 2 d':>12}")
    rows = [(pop, model.models[pop]) for pop in model.models] + [("pooled", model.default)]
    for pop, m in rows:
        s = summaries[pop]
        print(f"{pop:<10} {s['residuals']:6d} {s['outlier_fraction']:8.1%}   "
              f"{_fmt3(m.sigma0_km):<24} {_fmt3(m.growth_km_per_day):<24} "
              f"{m.sigma_rtn(2.0)[1]:9.2f} km")
    print(f"{'assumed':<10} {'':>6} {'':>8}   {_fmt3(DEFAULT_MODEL.sigma0_km):<24} "
          f"{_fmt3(DEFAULT_MODEL.growth_km_per_day):<24} {DEFAULT_MODEL.sigma_rtn(2.0)[1]:9.2f} km")

    out = args.out or DEFAULT_COVARIANCE_PATH
    model.save(out)
    out.with_suffix(".summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nWrote {out} (used automatically by `orbitguard screen`)")
    return 0


def _fmt3(values) -> str:
    return ", ".join(f"{v:.3f}" for v in values)


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
    p.add_argument("--store", action="store_true",
                   help="Also save the run to the results database (needs the [api] extra)")
    p.add_argument("--db-url", default=None, help=DB_URL_HELP)
    p.set_defaults(func=_cmd_screen)

    v = sub.add_parser("validate", help="Check the newest cached snapshot of each group")
    v.add_argument("--group", action="append", help="Group to check (default: all cached)")
    v.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    v.add_argument("--all", dest="verbose_checks", action="store_true", help="Show passing checks too")
    v.set_defaults(func=_cmd_validate)

    db = sub.add_parser("db", help="Results database")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    imp = db_sub.add_parser("import", help="Import screening result JSON files")
    imp.add_argument("paths", nargs="*", type=Path,
                     help="Result files (default: data/processed/conjunctions_*.json)")
    imp.add_argument("--db-url", default=None, help=DB_URL_HELP)
    imp.set_defaults(func=_cmd_db_import)

    s = sub.add_parser("serve", help="Run the HTTP API (needs the [api] extra)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--db-url", default=None, help=DB_URL_HELP)
    s.add_argument("--risk-model", type=Path, default=None,
                   help="Trained CDM risk model (default: data/models/cdm_risk_model.joblib)")
    s.add_argument("--reload", action="store_true", help="Restart on code changes (development)")
    s.set_defaults(func=_cmd_serve)

    c = sub.add_parser("calibrate",
                       help="Fit the covariance model from cached snapshots of the same objects")
    c.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    c.add_argument("--out", type=Path, default=None,
                   help=f"Model output path (default: {DEFAULT_COVARIANCE_PATH})")
    c.set_defaults(func=_cmd_calibrate)

    add_risk_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "screen" and not args.group:
        args.group = ["active"]
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)
