"""
Sanity checks on app/model_metrics.json — the file the app reads live to display
AUC/precision/recall instead of a hardcoded number. This doesn't re-run training;
it just checks the file that's checked in is well-formed and in sane ranges, so a
bad hand-edit or a broken training run can't silently ship a nonsense number.
"""
import json
import os

import pytest

METRICS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "model_metrics.json"
)

REQUIRED_KEYS = {
    "auc_roc", "precision_high_risk", "recall_high_risk", "n_features", "seed",
}


@pytest.fixture(scope="module")
def metrics():
    if not os.path.exists(METRICS_PATH):
        pytest.skip("app/model_metrics.json not present — run app/run_modeling.py first")
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_required_keys_present(metrics):
    missing = REQUIRED_KEYS - set(metrics)
    assert not missing, f"model_metrics.json is missing keys: {missing}"


def test_auc_in_valid_range(metrics):
    # A meaningful binary classifier should beat random (0.5); this also guards
    # against a metrics file left over from a broken/partial training run.
    assert 0.5 < metrics["auc_roc"] < 1.0


def test_precision_recall_in_valid_range(metrics):
    assert 0.0 <= metrics["precision_high_risk"] <= 1.0
    assert 0.0 <= metrics["recall_high_risk"] <= 1.0

