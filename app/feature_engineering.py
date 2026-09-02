"""
feature_engineering.py — Encodes a single patient's form inputs into the exact
feature vector the trained model expects.

Pulled out of streamlit_app.py so it can be unit-tested directly (tests/test_feature_engineering.py)
without importing the Streamlit UI. Keep this in sync with app/run_modeling.py's
pd.get_dummies() column naming — see the comments below for how they're kept aligned.
"""
import re

# ── Encoding tables that mirror run_modeling.py's pd.get_dummies() exactly ──
# (see notebooks/03_feature_engineering.ipynb + app/run_modeling.py)
ADMISSION_TYPE_ID = {"Emergency": 1, "Urgent": 2, "Elective": 3}
DISCHARGE_DISPOSITION_ID = {
    "Home": 1,
    "Skilled Nursing Facility": 3,
    "Home Health Agency": 6,
    "Left Against Medical Advice": 7,
    "Other": 18,
}
# discharge_risk_group is the clinical risk tier run_modeling.py actually one-hot
# encodes (grouped from discharge_disposition_id) — this mirrors that same grouping.
DISCHARGE_RISK_GROUP = {
    "Home": "Moderate (home)",
    "Skilled Nursing Facility": "Lower (SNF/rehab)",
    "Home Health Agency": "Lower (home health)",
    "Left Against Medical Advice": "High risk (AMA)",
    "Other": "Other",
}
A1C_CATEGORY = {"None": "Unknown", "Norm": "Norm", ">7": ">7", ">8": ">8"}
MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}

# The other 20 diabetes medication columns the model was trained on.
# This simplified UI only asks about insulin, so they default to "No" (0) —
# same fallback run_modeling.py itself uses for missing values.
OTHER_MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone',
    'metformin_rosiglitazone', 'metformin_pioglitazone',
]

DIAG1_CATEGORIES = [
    'Circulatory', 'Other', 'Respiratory', 'Digestive', 'Injury',
    'Musculoskeletal', 'Genitourinary', 'Neoplasm', 'Mental',
    'Supplementary', 'Diabetes', 'Unknown', 'External cause',
]


def _dummy_col(col, category):
    """
    Reproduce run_modeling.py's column naming exactly:
    pd.get_dummies() names a column "{col}_{category}", then the script
    sanitizes every column name with df.columns.str.replace(r"[^A-Za-z0-9_]", "_").
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", f"{col}_{category}")


def build_feature_vector(age, gender, race, admission_type, discharge_dest,
                          time_in_hospital, num_inpatient, num_emergency,
                          num_outpatient, num_diagnoses, num_medications,
                          num_lab_procs, num_procedures, insulin_status, a1c_result,
                          diag_category):
    """
    Build a feature vector that matches the model's expected input —
    including the one-hot encoded categorical columns (race, gender, age,
    primary diagnosis category, discharge risk group, medical specialty, lab
    results) that run_modeling.py creates via pd.get_dummies(), plus the raw
    numeric admission/discharge codes and comorbidity_count it keeps as
    plain features (matching notebooks/03_feature_engineering.ipynb, which
    is what the deployed model is actually trained to replicate — see
    run_modeling.py's module docstring for why this matters).

    medical_specialty and max_glu_serum aren't collected by this simplified
    UI at all, so they default to "Unknown" — the same fallback the training
    pipeline applies to missing values. admission_source_id is inferred from
    admission type. comorbidity_count is approximated as min(num_diagnoses, 3)
    since this UI doesn't collect individual diagnosis codes. Only insulin
    dose is tracked individually among the 21 medication columns; the other
    20 default to "No", same as an unprescribed drug.
    """
    age_map = {
        '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }

    prior_util = num_inpatient * 3 + num_emergency * 2 + num_outpatient
    insulin_code = MED_MAP[insulin_status]
    insulin_changed = 1 if insulin_status in ('Up', 'Down') else 0
    a1c_abnormal = 1 if a1c_result in ('>7', '>8') else 0
    a1c_tested = 1 if a1c_result != 'None' else 0
    admitted_er = 1 if admission_type == "Emergency" else 0

    features_dict = {
        'age_numeric': age_map.get(age, 65),
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procs,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': num_outpatient,
        'number_emergency': num_emergency,
        'number_inpatient': num_inpatient,
        'number_diagnoses': num_diagnoses,
        'comorbidity_count': min(num_diagnoses, 3),
        'prior_utilization_score': prior_util,
        'medication_changes': insulin_changed,  # only insulin change is known here
        'insulin_changed': insulin_changed,
        'insulin': insulin_code,
        'num_active_meds': max(1, num_medications // 3) + (1 if insulin_code > 0 else 0),
        'high_lab_burden': 1 if num_lab_procs > 54 else 0,
        'admitted_from_er': admitted_er,
        'emergency_admission': admitted_er,
        'a1c_abnormal': a1c_abnormal,
        'a1c_tested': a1c_tested,
        # Raw numeric admission/discharge codes — the model does NOT one-hot
        # encode these (it uses discharge_risk_group for the discharge signal
        # instead), it keeps them as plain ordinal features.
        'admission_type_id': ADMISSION_TYPE_ID.get(admission_type, 1),
        'discharge_disposition_id': DISCHARGE_DISPOSITION_ID.get(discharge_dest, 18),
        'admission_source_id': 7 if admission_type == "Emergency" else 1,
    }
    for med in OTHER_MED_COLS:
        features_dict[med] = 0  # "No" — not collected individually by this UI

    # ── One-hot categorical encodings — must match run_modeling.py exactly ──
    categorical_selections = {
        'race': race,
        'gender': gender,
        'age': age,
        'diag1_category': diag_category,
        'discharge_risk_group': DISCHARGE_RISK_GROUP.get(discharge_dest, "Other"),
        'medical_specialty': "Unknown",   # not collected by this UI
        'max_glu_serum': "Unknown",       # not collected by this UI
        'A1Cresult': A1C_CATEGORY.get(a1c_result, "Unknown"),
    }
    for col, category in categorical_selections.items():
        # If this category was the one pd.get_dummies(drop_first=True) dropped
        # during training, there's no matching column — that's correct: it's
        # the reference case, which is represented by all dummies staying 0.
        features_dict[_dummy_col(col, category)] = 1

    return features_dict
