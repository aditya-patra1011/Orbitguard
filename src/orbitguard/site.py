"""Export the dashboard and the latest run as a static site (for GitHub Pages).

The JSON files use the API's own response models, so the dashboard reads them exactly as
it reads the live API:

    index.html, app.js, ...     the dashboard, switched to static mode by config.js
    data/summary.json           latest run, latest pipeline run, encounter counts
    data/conjunctions.json      the highest-Pc encounters of the latest run
    data/elements.json          the run's element sets, for drawing every object

GitHub Pages cannot send HTTP headers, so the content security policy goes into a
<meta> tag (which cannot carry frame-ancestors; Pages pages can be framed).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from orbitguard.api import (
    WEB_DIR, ConjunctionOut, ElementSetOut, PipelineRunOut, RunElementsOut, RunOut, _as_utc,
)
from orbitguard.covariance import population
from orbitguard.security import CONTENT_SECURITY_POLICY
from orbitguard.store import (
    ConjunctionQuery, ConjunctionRow, get_run_elements, latest_run, list_pipeline_runs,
    query_conjunctions,
)

TOP_ENCOUNTERS = 1000
COUNT_THRESHOLDS = {"1e-4": 1e-4, "1e-6": 1e-6, "0": 0.0}


def export_site(engine: Engine, out: Path, top: int = TOP_ENCOUNTERS) -> dict:
    """Write the static site to ``out`` (replacing its contents); return the summary."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(WEB_DIR, out)
    data = out / "data"
    data.mkdir()

    with Session(engine) as session:
        run = latest_run(session)
        pipeline = list_pipeline_runs(session, limit=1)
        summary = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "run": RunOut.model_validate(run).model_dump(mode="json") if run else None,
            "pipeline": (PipelineRunOut.model_validate(pipeline[0]).model_dump(mode="json")
                         if pipeline else None),
            "counts": {},
        }
        if run is not None:
            for key, min_pc in COUNT_THRESHOLDS.items():
                summary["counts"][key] = session.scalar(
                    select(func.count()).select_from(ConjunctionRow).where(
                        ConjunctionRow.run_id == run.id, ConjunctionRow.co_orbital.is_(False),
                        ConjunctionRow.pc >= min_pc))
            _, rows = query_conjunctions(session, ConjunctionQuery(
                run_id=run.id, min_pc=0.0, order_by="pc", limit=top))
            _write(data / "conjunctions.json", {
                "run_id": run.id, "total_with_pc": summary["counts"]["0"],
                "items": [ConjunctionOut.model_validate(r).model_dump(mode="json") for r in rows]})
            objects = get_run_elements(session, run.id)
            if objects is not None:   # absent for runs stored before element sets were kept
                _write(data / "elements.json", RunElementsOut(
                    run_id=run.id, generated_at=_as_utc(run.generated_at), count=len(objects),
                    objects=[ElementSetOut(population=population(o["name"]), **o) for o in objects],
                ).model_dump(mode="json"))
    _write(data / "summary.json", summary)

    # Explicit UTF-8 throughout: the page uses characters such as "≥", and Windows would
    # otherwise use its legacy code page.
    config = (WEB_DIR / "config.js").read_text(encoding="utf-8")
    (out / "config.js").write_text(config + '\n// Static export: read data/*.json instead of the API.\n'
                                            'window.ORBITGUARD_DATA = "static";\n', encoding="utf-8")
    meta_csp = "; ".join(d for d in CONTENT_SECURITY_POLICY.split("; ")
                         if not d.startswith("frame-ancestors"))
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    meta = f'<meta charset="utf-8">\n  <meta http-equiv="Content-Security-Policy" content="{meta_csp}">'
    (out / "index.html").write_text(html.replace('<meta charset="utf-8">', meta, 1), encoding="utf-8")
    (out / ".nojekyll").write_text("")   # serve files as they are, no Jekyll processing
    return summary


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
