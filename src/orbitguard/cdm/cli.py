"""``orbitguard risk cv | train | evaluate``: the CDM final-risk model.

ML dependencies are imported inside the commands, so ``orbitguard screen`` keeps
working without the optional [ml] extra installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL_PATH = Path("data/models/cdm_risk_model.joblib")
DEFAULT_EVAL_PATH = Path("data/models/cdm_test_evaluation.json")


def _cmd_cv(args: argparse.Namespace) -> int:
    import numpy as np

    from orbitguard.cdm.data import load_training_tasks
    from orbitguard.cdm.model import Dataset, cross_validate, select_threshold

    strict, aug = load_training_tasks(args.data_dir)
    train, extra = Dataset.from_task(strict), Dataset.from_task(aug)
    print(f"Training events: {len(train.y)} strict ({int((train.y >= -6).sum())} high-risk), "
          f"{len(extra.y)} augmentation ({int((extra.y >= -6).sum())} high-risk)")
    results = cross_validate(train, extra, n_repeats=args.repeats)
    base = np.array([s.loss for s in results[0].scores])
    print(f"\n{'candidate':<18} {'L':>6} {'MSE_HR':>7} {'F2':>6} {'recall':>7}  {'paired dL vs LRP':>18}")
    for r in results:
        loss = np.array([s.loss for s in r.scores])
        d = loss - base
        name = "LRP baseline" if r.threshold is None else f"threshold {r.threshold:g}"
        print(f"{name:<18} {loss.mean():6.3f} {np.mean([s.mse_hr for s in r.scores]):7.3f} "
              f"{np.mean([s.f2 for s in r.scores]):6.3f} {np.mean([s.recall for s in r.scores]):7.3f}  "
              f"{d.mean():+7.3f} (sd {d.std():.3f})")
    print(f"\nSelected late-riser threshold: {select_threshold(results):g}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from orbitguard.cdm.data import load_training_tasks
    from orbitguard.cdm.evaluate import train_final_model

    strict, aug = load_training_tasks(args.data_dir)
    model = train_final_model(strict, aug, threshold=args.threshold)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    print(f"Trained on {len(strict.targets)} strict + {len(aug.targets)} augmentation events; "
          f"late-riser threshold {args.threshold:g}\nWrote {args.model}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from orbitguard.cdm.data import load_test_task
    from orbitguard.cdm.evaluate import evaluate
    from orbitguard.cdm.model import RiskModel

    ev = evaluate(RiskModel.load(args.model), load_test_task(args.data_dir))
    print("Held-out test set. Use it to report results, never to tune the model.\n")
    print(f"  model         {ev.model}")
    print(f"  latest risk   {ev.latest_risk}")
    print(f"  constant -5   {ev.constant_risk}")
    c = ev.confusion
    print(f"\n  dangerous events caught {c['true_positive']}, missed {c['false_negative']}; "
          f"false alarms {c['false_positive']}")
    print(f"  position among the published top 10 + baseline: {ev.leaderboard_rank}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ev.to_dict(), indent=2))
    print(f"\nWrote {args.out}")
    return 0


def add_risk_parser(sub) -> None:
    from orbitguard.cdm.data import DEFAULT_DATA_DIR

    risk = sub.add_parser("risk", help="Final-risk model on ESA conjunction data messages")
    risk_sub = risk.add_subparsers(dest="risk_command", required=True)

    cv = risk_sub.add_parser("cv", help="Cross-validate on training data and select the threshold")
    cv.add_argument("--repeats", type=int, default=4, help="Repeats of 5-fold CV")
    cv.set_defaults(func=_cmd_cv)

    tr = risk_sub.add_parser("train", help="Train the final model on all training data")
    tr.add_argument("--threshold", type=float, default=0.05,
                    help="Late-riser probability threshold (from `risk cv`)")
    tr.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    tr.set_defaults(func=_cmd_train)

    ev = risk_sub.add_parser("evaluate", help="Score the trained model on the held-out test set")
    ev.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ev.add_argument("--out", type=Path, default=DEFAULT_EVAL_PATH)
    ev.set_defaults(func=_cmd_evaluate)

    for p in (cv, tr, ev):
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
