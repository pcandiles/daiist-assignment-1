"""
Automated evaluation of the saved models. Run after `main.py train`:

    uv run pytest tests/test_evaluation.py -v

Loads the artifacts in models/ (never retrains) and checks that the three
methods beat the naive baseline and agree with each other on the test set.
Metrics are written to evaluation_metrics.json (and to the GitHub Actions
job summary when running in CI).
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Quality gates. Deliberately loose: they catch a broken pipeline, not a
# slightly worse random draw.
MIN_R2_LOG = 0.45           # each model must explain a large share of the log-price variance
MIN_MAE_IMPROVEMENT = 0.20  # each model must cut the baseline MAE (EUR) by at least 20%
MAX_METHOD_GAP_LOG = 0.15   # mean |difference| in log-price between methods (PyTorch is currently ~0.10 off sklearn)


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


@pytest.fixture(scope="module")
def preds_log():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    sklearn_model = joblib.load(MODELS_DIR / "sklearn_ridge.joblib")
    manual = torch.load(MODELS_DIR / "manual_pytorch.pt")
    nn_state = torch.load(MODELS_DIR / "nn_pytorch.pt")
    baseline = joblib.load(MODELS_DIR / "baseline.joblib")
    test_set = joblib.load(MODELS_DIR / "test_set.joblib")

    X = preprocessor.transform(test_set["X_test"]).astype("float32")
    X = np.nan_to_num(X, nan=0.0)  # same handling as app.py
    X_t = torch.tensor(X)

    model_nn = nn.Linear(nn_state["weight"].shape[1], 1)
    model_nn.load_state_dict(nn_state)
    model_nn.eval()

    with torch.no_grad():
        preds = {
            "Naive baseline": np.full(len(X), baseline["median_log_price"]),
            "Scikit-learn Ridge": sklearn_model.predict(X),
            "Manual PyTorch": (X_t @ manual["w"] + manual["b"]).squeeze().numpy(),
            "Standard PyTorch": model_nn(X_t).squeeze().numpy(),
        }
    preds["y_true"] = np.asarray(test_set["y_test_log"])
    return preds


@pytest.fixture(scope="module")
def metrics(preds_log):
    y = preds_log["y_true"]
    out = {}
    for name, p in preds_log.items():
        if name == "y_true":
            continue
        out[name] = {
            "R2 (log)": r2(y, p),
            "RMSE (log)": rmse(y, p),
            "MAE (EUR)": mae(np.expm1(y), np.expm1(p)),
            "RMSE (EUR)": rmse(np.expm1(y), np.expm1(p)),
        }
    (ROOT / "evaluation_metrics.json").write_text(json.dumps(out, indent=2))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        lines = ["## Test-set evaluation", "",
                 "| Model | R2 (log) | RMSE (log) | MAE (EUR) | RMSE (EUR) |",
                 "|---|---|---|---|---|"]
        for name, m in out.items():
            lines.append(f"| {name} | {m['R2 (log)']:.3f} | {m['RMSE (log)']:.3f} "
                         f"| {m['MAE (EUR)']:.2f} | {m['RMSE (EUR)']:.2f} |")
        with open(summary, "a") as f:
            f.write("\n".join(lines) + "\n")
    return out


MODELS = ["Scikit-learn Ridge", "Manual PyTorch", "Standard PyTorch"]


def test_predictions_are_finite(preds_log):
    for name, p in preds_log.items():
        assert np.isfinite(p).all(), f"{name} produced NaN/inf predictions"


@pytest.mark.parametrize("name", MODELS)
def test_model_explains_variance(metrics, name):
    assert metrics[name]["R2 (log)"] >= MIN_R2_LOG


@pytest.mark.parametrize("name", MODELS)
def test_model_beats_baseline(metrics, name):
    base = metrics["Naive baseline"]["MAE (EUR)"]
    assert metrics[name]["MAE (EUR)"] <= base * (1 - MIN_MAE_IMPROVEMENT)


def test_three_methods_agree(preds_log):
    sk = preds_log["Scikit-learn Ridge"]
    for other in ["Manual PyTorch", "Standard PyTorch"]:
        gap = mae(sk, preds_log[other])
        assert gap <= MAX_METHOD_GAP_LOG, f"{other} differs from sklearn by {gap:.3f} (log)"
