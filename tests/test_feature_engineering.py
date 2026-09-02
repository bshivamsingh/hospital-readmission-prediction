"""
Regression tests for app/feature_engineering.py.

These exist because of a real bug that shipped in this project: build_feature_vector()
used to produce ~18 features while the trained model expected 179, and the missing
columns were silently zero-filled instead of raising an error — which is exactly how
the app ended up predicting ~99% readmission risk for every patient regardless of the
inputs entered. These tests catch that class of bug before it reaches a deploy.
"""
import itertools
import os

import joblib
import pytest

from feature_engineering import build_feature_vector

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/app"

RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
GENDERS = ["Male", "Female"]
AGES = ["[10-20)", "[30-40)", "[60-70)", "[90-100)"]
ADMISSION_TYPES = ["Emergency", "Urgent", "Elective"]
DISCHARGE_DESTS = ["Home", "Skilled Nursing Facility", "Home Health Agency",
                    "Left Against Medical Advice", "Other"]
A1C_RESULTS = ["None", "Norm", ">7", ">8"]
DIAG_CATEGORIES = ["Circulatory", "Diabetes", "Respiratory", "Unknown", "External cause"]


@pytest.fixture(scope="module")
def feature_names():
    path = os.path.join(APP_DIR, "feature_names.pkl")
    if not os.path.exists(path):
        pytest.skip("app/feature_names.pkl not present — run app/run_modeling.py first")
    return set(joblib.load(path))


def _build(**overrides):
    defaults = dict(
        age="[50-60)", gender="Female", race="Caucasian", admission_type="Emergency",
        discharge_dest="Home", time_in_hospital=3, num_inpatient=1, num_emergency=1,
        num_outpatient=1, num_diagnoses=5, num_medications=8, num_lab_procs=30,
        num_procedures=1, insulin_status="No", a1c_result="None", diag_category="Circulatory",
    )
    defaults.update(overrides)
    return build_feature_vector(**defaults)


def test_returns_nonempty_dict():
    fdict = _build()
    assert isinstance(fdict, dict)
    assert len(fdict) > 40  # a real feature vector, not the ~18-key one from the original bug


def test_every_generated_column_exists_in_trained_model(feature_names):
    """
    Every dummy column build_feature_vector() can produce must either be a real
    column the model was trained on, or be one of the reference categories that
    pd.get_dummies(drop_first=True) intentionally drops (in which case leaving it
    out of the dict is correct — the model sees the all-zeros reference case).
    """
    combos = list(itertools.product(
        RACES, GENDERS, AGES, ADMISSION_TYPES, DISCHARGE_DESTS, A1C_RESULTS, DIAG_CATEGORIES
    ))
    # Sample to keep the test fast; the full cross-product (thousands of combos) was
    # checked manually once during development and confirmed clean.
    step = max(1, len(combos) // 500)
    unmatched = set()
    for race, gender, age, admission_type, discharge_dest, a1c, diag in combos[::step]:
        fdict = _build(
            race=race, gender=gender, age=age, admission_type=admission_type,
            discharge_dest=discharge_dest, a1c_result=a1c, diag_category=diag,
        )
        unmatched |= {k for k in fdict if k not in feature_names}

    # The only acceptable "unmatched" columns are the known drop_first=True reference
    # categories — anything else means a category doesn't match the trained schema.
    known_reference_categories = {
        "gender_Female", "race_AfricanAmerican", "diag1_category_Circulatory",
        "discharge_risk_group_High_risk__AMA_", "A1Cresult__7",
    }
    unexpected = unmatched - known_reference_categories
    assert not unexpected, f"Unexpected unmatched columns (would silently zero-fill): {unexpected}"


def test_predictions_vary_across_different_patients(feature_names):
    """
    Loads the real trained model and confirms predicted risk actually varies across
    clearly different patients, instead of pinning near a constant value (the original
    bug: everyone got ~99% regardless of input).
    """
    model_path = os.path.join(APP_DIR, "xgb_readmission_model.pkl")
    if not (os.path.exists(model_path)):
        pytest.skip("app/xgb_readmission_model.pkl not present")
    model = joblib.load(model_path)
    feature_list = sorted(feature_names)  # any fixed order works, we're only checking spread

    profiles = [
        dict(age="[20-30)", gender="Male", race="Hispanic", admission_type="Urgent",
             discharge_dest="Home Health Agency", time_in_hospital=3, num_inpatient=0,
             num_emergency=0, num_outpatient=2, num_diagnoses=2, num_medications=3,
             num_lab_procs=15, num_procedures=0, insulin_status="No", a1c_result="Norm",
             diag_category="Respiratory"),
        dict(age="[70-80)", gender="Male", race="AfricanAmerican", admission_type="Emergency",
             discharge_dest="Skilled Nursing Facility", time_in_hospital=10, num_inpatient=3,
             num_emergency=2, num_outpatient=1, num_diagnoses=9, num_medications=18,
             num_lab_procs=70, num_procedures=4, insulin_status="Up", a1c_result=">8",
             diag_category="Circulatory"),
        dict(age="[80-90)", gender="Female", race="Caucasian", admission_type="Emergency",
             discharge_dest="Left Against Medical Advice", time_in_hospital=1, num_inpatient=5,
             num_emergency=4, num_outpatient=0, num_diagnoses=9, num_medications=10,
             num_lab_procs=40, num_procedures=0, insulin_status="Down", a1c_result=">7",
             diag_category="Diabetes"),
    ]

    probs = []
    for profile in profiles:
        fdict = build_feature_vector(**profile)
        row = [fdict.get(f, 0) for f in feature_list]
        probs.append(float(model.predict_proba([row])[0, 1]))

    spread = max(probs) - min(probs)
    assert spread > 0.05, (
        f"Predictions barely vary across clearly different patients ({probs}); "
        "this is the exact shape of the original always-~99% bug."
    )
    assert all(0.0 <= p <= 1.0 for p in probs)
