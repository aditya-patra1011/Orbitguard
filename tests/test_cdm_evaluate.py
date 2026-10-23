from orbitguard.cdm.evaluate import LEADERBOARD, confusion, top_features
from test_cdm_model import model, synthetic  # noqa: F401  (fixture reuse)


def test_confusion_counts():
    true = [-5.0, -7.0, -4.0, -20.0]
    pred = [-5.5, -5.0, -6.001, -6.001]
    assert confusion(true, pred) == {"true_positive": 1, "false_positive": 1,
                                     "false_negative": 1, "true_negative": 1}


def test_top_features_are_normalised_gains(model):  # noqa: F811
    feats = top_features(model, n=3)
    assert set(feats) == {"value_correction", "late_riser"}
    for gains in feats.values():
        assert len(gains) <= 3 and all(v >= 0 for v in gains.values())


def test_leaderboard_is_the_published_top_ten():
    assert len(LEADERBOARD) == 10
    assert min(LEADERBOARD.values()) == 0.556


def test_cli_does_not_import_ml_libraries():
    # `orbitguard screen` must work without the optional [ml] extra.
    import subprocess
    import sys
    code = ("import sys, orbitguard.cli; "
            "print(any(m in sys.modules for m in ('pandas', 'lightgbm', 'sklearn')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
