"""Command-line entry point: ``python -m orbitguard screen | calibrate | risk | db | serve``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from orbitguard.cdm.cli import add_risk_parser
from orbitguard.calibration import MAX_PAIR_AGE_DAYS, calibrate_from_cache
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

        from orbitguard.propagation import element_sets
        from orbitguard.store import make_engine, save_result, save_run_elements

        with Session(make_engine(args.db_url)) as session:
            run = save_result(session, result)
            save_run_elements(session, run.id, element_sets(catalog))
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


def _pipeline_config(args: argparse.Namespace):
    from orbitguard.pipeline import PipelineConfig

    return PipelineConfig(groups=args.group, hours=args.hours, threshold_km=args.threshold,
                          hbr_m=args.hbr_m, cache_dir=args.cache_dir,
                          calibrate=not args.no_calibrate, refresh=args.refresh)


def _cmd_pipeline_schedule(args: argparse.Namespace) -> int:
    import signal
    import threading

    from orbitguard.pipeline import run_on_schedule, run_pipeline
    from orbitguard.store import make_engine

    cfg, engine = _pipeline_config(args), make_engine(args.db_url)
    stop = threading.Event()

    def request_stop(signum, frame):
        print("Stop requested; finishing after the current run.", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)   # `docker compose stop`
    signal.signal(signal.SIGINT, request_stop)    # Ctrl+C
    print(f"Running the pipeline every {args.every_hours:g} h for {', '.join(cfg.groups)}", flush=True)

    def once():
        out = run_pipeline(cfg, engine)
        alerts = f", {len(out.drift_alerts)} drift alert(s)" if out.drift_alerts else ""
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z] pipeline run {out.pipeline_run_id}: "
              f"{out.status}{alerts} {out.message}".rstrip(), flush=True)
        return out

    run_on_schedule(once, args.every_hours * 3600, stop=stop)
    return 0


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    from orbitguard.pipeline import run_pipeline
    from orbitguard.store import make_engine

    out = run_pipeline(_pipeline_config(args), make_engine(args.db_url))
    print(f"Pipeline run {out.pipeline_run_id}: {out.status.upper()}")
    for report in out.validation:
        problems = [c for c in report["checks"] if not c["passed"]]
        print(f"  validate {report['group']:<20} {report['n_records']:>6} records"
              + ("" if not problems else "  " + "; ".join(f"{c['severity']}: {c['name']}" for c in problems)))
    m = out.metrics
    if out.status == "success":
        print(f"  screened {m['objects_screened']} objects: {m['conjunctions']} conjunctions, "
              f"{m['high_risk_events']} with Pc >= 1e-4 (stored as run {out.run_id}, "
              f"{m['screening_seconds']} s)")
        if "calibrated_sigma_t_2d_km" in m:
            sig = ", ".join(f"{k} {v} km" for k, v in m["calibrated_sigma_t_2d_km"].items())
            print(f"  recalibrated from {m['calibration_pairs']} snapshot pairs: along-track 1-sigma at 2 days: {sig}")
        for a in out.drift_alerts:
            print(f"  DRIFT {a['metric']}: {a['value']} vs median {a['baseline_median']} (z {a['robust_z']})")
        if not out.drift_alerts:
            print("  drift: none detected" + (f" ({out.message})" if out.message else ""))
    else:
        print(f"  {out.message}")
    return 0 if out.status == "success" else 1


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


def _cmd_export_site(args: argparse.Namespace) -> int:
    from orbitguard.site import export_site
    from orbitguard.store import make_engine

    summary = export_site(make_engine(args.db_url), args.out, top=args.top)
    if summary["run"] is None:
        print(f"Wrote {args.out}, but there is no screening run yet: the dashboard will say so")
        return 0
    run = summary["run"]
    print(f"Wrote {args.out}: run {run['id']} ({run['objects_screened']} objects, "
          f"{summary['counts']['1e-4']} encounters with Pc >= 1e-4)")
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
    fitted = calibrate_from_cache(args.cache_dir)
    if fitted is None:
        print(f"Need two cached snapshots of a group in {args.cache_dir}, fetched at most "
              f"{MAX_PAIR_AGE_DAYS:g} days apart; run `orbitguard screen` again after "
              "CelesTrak publishes new data.")
        return 1
    model, summaries, n_pairs = fitted

    pooled = summaries["pooled"]
    print(f"\n{pooled['residuals']} residuals from {n_pairs} snapshot pairs, "
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

    pl = sub.add_parser("pipeline", help="Fetch, validate, recalibrate, screen, store, monitor")
    pl_sub = pl.add_subparsers(dest="pipeline_command", required=True)
    plr = pl_sub.add_parser("run", help="Run the pipeline once (exit code 1 unless it succeeds)")
    pls = pl_sub.add_parser("schedule", help="Run the pipeline repeatedly until stopped")
    pls.add_argument("--every-hours", type=float, default=6.0,
                     help="Interval between run starts (minimum 2 h)")
    for parser_ in (plr, pls):
        parser_.add_argument("--group", action="append", required=True,
                             help="CelesTrak group (repeatable)")
        parser_.add_argument("--hours", type=float, default=24.0)
        parser_.add_argument("--threshold", type=float, default=5.0)
        parser_.add_argument("--hbr-m", type=float, default=10.0)
        parser_.add_argument("--no-calibrate", action="store_true", help="Skip recalibration")
        parser_.add_argument("--refresh", action="store_true", help="Ignore the 2-hour data cache")
        parser_.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
        parser_.add_argument("--db-url", default=None, help=DB_URL_HELP)
    plr.set_defaults(func=_cmd_pipeline_run)
    pls.set_defaults(func=_cmd_pipeline_schedule)

    db = sub.add_parser("db", help="Results database")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    imp = db_sub.add_parser("import", help="Import screening result JSON files")
    imp.add_argument("paths", nargs="*", type=Path,
                     help="Result files (default: data/processed/conjunctions_*.json)")
    imp.add_argument("--db-url", default=None, help=DB_URL_HELP)
    imp.set_defaults(func=_cmd_db_import)

    ex = sub.add_parser("export-site", help="Write the dashboard and latest run as a static site")
    ex.add_argument("--out", type=Path, default=Path("site"), help="Output folder (replaced)")
    ex.add_argument("--top", type=int, default=1000, help="Highest-Pc encounters to include")
    ex.add_argument("--db-url", default=None, help=DB_URL_HELP)
    ex.set_defaults(func=_cmd_export_site)

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
