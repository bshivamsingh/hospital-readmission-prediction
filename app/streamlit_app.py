"""
Hospital Readmission Risk Scorer
Streamlit App — Project 1: Healthcare Analytics

Deploy to Streamlit Community Cloud (free):
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo and point to app/streamlit_app.py

Requirements: see requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital Readmission Risk Scorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load model artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('app/xgb_readmission_model.pkl')
    features = joblib.load('app/feature_names.pkl')
    return model, features, None

@st.cache_data
def load_metrics():
    # Real, measured performance from the last run_modeling.py training run —
    # NOT a hardcoded claim. See app/model_metrics.json.
    try:
        with open('app/model_metrics.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

try:
    model, FEATURE_NAMES, explainer = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

METRICS = load_metrics()

# ── Sidebar — Patient Input ───────────────────────────────────
st.sidebar.header("🧑‍⚕️ Patient Information")
st.sidebar.caption("Enter patient details to compute 30-day readmission risk")

with st.sidebar:
    # Demographics
    st.subheader("Demographics")
    age = st.selectbox("Age group", [
        '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)',
        '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ], index=5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race   = st.selectbox("Race", [
        "Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"
    ])

    st.divider()

    # Admission details
    st.subheader("Admission Details")
    admission_type = st.selectbox("Admission type", [
        "Emergency", "Urgent", "Elective"
    ])
    discharge_dest = st.selectbox("Discharge destination", [
        "Home", "Skilled Nursing Facility", "Home Health Agency",
        "Left Against Medical Advice", "Other"
    ])
    time_in_hospital = st.slider("Length of stay (days)", 1, 14, 4)

    st.divider()

    # Prior utilization
    st.subheader("Prior Year Utilization")
    num_inpatient  = st.number_input("Prior inpatient visits",  0, 20, 0)
    num_emergency  = st.number_input("Prior emergency visits",  0, 20, 0)
    num_outpatient = st.number_input("Prior outpatient visits", 0, 40, 1)

    st.divider()

    # Clinical
    st.subheader("Clinical Features")
    num_diagnoses    = st.slider("Number of diagnoses", 1, 9, 5)
    diag_category    = st.selectbox("Primary diagnosis category", [
        "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
        "Musculoskeletal", "Genitourinary", "Neoplasm", "Mental",
        "Supplementary", "External cause", "Other", "Unknown"
    ], help="Broad ICD-9 category for the primary diagnosis (most encounters in the training data are Circulatory)")
    num_medications  = st.slider("Number of medications", 1, 81, 15)
    num_lab_procs    = st.slider("Lab procedures", 1, 132, 44)
    num_procedures   = st.slider("Non-lab procedures", 0, 6, 1)
    insulin_status   = st.selectbox("Insulin status", ["No", "Steady", "Up", "Down"])
    a1c_result       = st.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"])

    predict_btn = st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True)

# ── Main Page ─────────────────────────────────────────────────
st.title("🏥 Hospital Readmission Risk Scorer")
st.caption("Predicts 30-day readmission risk for diabetic patients using XGBoost + SHAP explainability")

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model", "XGBoost")
with col_info2:
    auc_display = f"{METRICS['auc_roc']:.2f}" if METRICS else "N/A"
    st.metric("AUC-ROC", auc_display, help="Measured on a held-out test split by the last run_modeling.py training run — see app/model_metrics.json.")
with col_info3:
    st.metric("Dataset", "71.5K encounters")

st.divider()

if not MODEL_LOADED:
    st.warning("""
    ⚠️ **Model not loaded.** 

    To run this app with a real model:
    1. Run notebooks 01–04 in order to train and save the model
    2. Ensure `app/xgb_readmission_model.pkl` exists

    **For demo purposes**, the app will use a mock risk score based on input features.
    """)

import re

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


def mock_risk_score(features_dict):
    """
    Deterministic mock score for demo (when model not loaded).
    Mimics the general direction of the real model.
    """
    score = 0.10  # base rate
    score += features_dict['number_inpatient'] * 0.025
    score += features_dict['number_emergency'] * 0.015
    score += features_dict['prior_utilization_score'] * 0.005
    score += features_dict['number_diagnoses'] * 0.008
    score += features_dict['time_in_hospital'] * 0.003
    score += features_dict['insulin_changed'] * 0.02
    score -= features_dict['a1c_tested'] * 0.01  # tested = better managed
    return min(max(score, 0.02), 0.95)

def risk_tier(prob):
    if prob < 0.15:
        return "Low", "#27500A", "✅"
    elif prob < 0.25:
        return "Moderate", "#854F0B", "⚠️"
    else:
        return "High", "#A32D2D", "🚨"

# ── Results Panel ─────────────────────────────────────────────
if predict_btn:
    features_dict = build_feature_vector(
        age, gender, race, admission_type, discharge_dest,
        time_in_hospital, num_inpatient, num_emergency, num_outpatient,
        num_diagnoses, num_medications, num_lab_procs, num_procedures,
        insulin_status, a1c_result, diag_category
    )

    # Get risk score
    if MODEL_LOADED:
        # Build a proper DataFrame matching the model's features
        # This is simplified — full version needs all feature engineering columns
        feat_df = pd.DataFrame([features_dict])
        for col in FEATURE_NAMES:
            if col not in feat_df.columns:
                feat_df[col] = 0
        feat_df = feat_df[FEATURE_NAMES]
        risk_prob = model.predict_proba(feat_df)[0, 1]
    else:
        risk_prob = mock_risk_score(features_dict)

    tier, tier_color, tier_icon = risk_tier(risk_prob)

    # ── Risk Score Display ──
    st.subheader("📊 Risk Assessment")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f"""
        <div style='background-color: #f0f2f6; border-radius: 12px; padding: 24px; text-align: center;'>
            <div style='font-size: 48px; font-weight: bold; color: {tier_color};'>
                {risk_prob:.0%}
            </div>
            <div style='font-size: 18px; color: {tier_color}; font-weight: 600;'>
                {tier_icon} {tier} Risk
            </div>
            <div style='font-size: 13px; color: #666; margin-top: 8px;'>
                30-day readmission probability
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Risk gauge (simple bar)
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh([0], [1], height=0.4, color='#e8e8e8', zorder=1)
        bar_color = '#27500A' if risk_prob < 0.15 else '#854F0B' if risk_prob < 0.25 else '#A32D2D'
        ax.barh([0], [risk_prob], height=0.4, color=bar_color, zorder=2)
        ax.axvline(0.15, color='#EF9F27', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(0.25, color='#E24B4A', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Predicted probability')
        ax.text(0.15, 0.5, 'Mod', ha='center', va='bottom', fontsize=8,
                color='#854F0B', transform=ax.get_xaxis_transform())
        ax.text(0.25, 0.5, 'High', ha='center', va='bottom', fontsize=8,
                color='#A32D2D', transform=ax.get_xaxis_transform())
        ax.set_title('Risk gauge', fontsize=10)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("**Key risk factors entered:**")
        key_facts = [
            f"Prior inpatient visits: **{num_inpatient}**",
            f"Diagnoses: **{num_diagnoses}**",
            f"Length of stay: **{time_in_hospital} days**",
            f"Discharge: **{discharge_dest}**",
            f"Insulin: **{insulin_status}**"
        ]
        for fact in key_facts:
            st.markdown(f"• {fact}")

    st.divider()

    # ── Clinical Recommendations ──
    st.subheader("💊 Recommended Care Actions")

    actions_by_tier = {
        "Low": [
            "Standard discharge planning — confirm patient understands medications",
            "Schedule follow-up appointment within 7–14 days",
            "Provide diabetic care education materials"
        ],
        "Moderate": [
            "Arrange structured follow-up call within 48–72 hours of discharge",
            "Ensure patient has a primary care physician appointment within 7 days",
            "Review medication adherence — consider pill organiser or blister packs",
            "Flag for nurse care manager outreach if any concerning signs"
        ],
        "High": [
            "🚨 **Prioritise for care transitions program enrollment**",
            "Schedule follow-up call within 24 hours of discharge",
            "Consider home health agency referral",
            "Medication reconciliation with pharmacist before discharge",
            "Assign care manager for 30-day post-discharge monitoring",
            "Schedule in-person clinic visit within 3 days"
        ]
    }

    for action in actions_by_tier[tier]:
        st.markdown(f"- {action}")

    st.divider()

    # ── SHAP Explanation (if model loaded) ──
    if MODEL_LOADED:
        st.subheader("🔬 Model Explanation (SHAP)")
        st.caption("Why did the model predict this risk score?")
        try:
            shap_vals = explainer.shap_values(feat_df)
            shap_exp = shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=feat_df.iloc[0],
                feature_names=FEATURE_NAMES
            )
            fig_shap, _ = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(shap_exp, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()
        except Exception as e:
            st.info(f"SHAP waterfall unavailable: {e}")
    else:
        st.info("ℹ️ Train and save the model (notebook 04) to see SHAP explanations here.")

    # ── Footer note ──
    st.caption("""
    ⚠️ **Disclaimer**: This tool is for educational and portfolio demonstration purposes only.  
    It is not a clinical decision support system and should not be used for real patient care.  
    Model trained on historical data from the Diabetes 130-US Hospitals dataset (UCI/Kaggle).
    """)

else:
    # Landing state — show description and instructions
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ### How to use this app
        1. Enter patient details in the **left sidebar**
        2. Click **Calculate Risk Score**
        3. Review the risk tier, gauge, and clinical recommendations
        4. (With trained model) View SHAP explanation of key risk drivers

        ---

        ### About this model
        - **Algorithm**: XGBoost classifier with SMOTE oversampling
        - **Training data**: 71,518 first-encounter diabetic patients across 130 US hospitals
        - **Validation**: single stratified 80/20 train/test split
        - **Performance**: AUC-ROC {auc} on held-out test set (precision {prec} / recall {rec} on the high-risk class at a 0.5 threshold) — modest, and it's the real, current number: an earlier "0.74" claim in this project was never actually produced by any run and has been corrected
        - **Explainability**: SHAP TreeExplainer for per-patient feature attribution
        """.format(
            auc=f"{METRICS['auc_roc']:.3f}" if METRICS else "N/A",
            prec=f"{METRICS['precision_high_risk']:.3f}" if METRICS else "N/A",
            rec=f"{METRICS['recall_high_risk']:.3f}" if METRICS else "N/A",
        ))

    with col_b:
        st.markdown("""
        ### Top readmission risk factors
        Based on SHAP analysis of the trained model:

        | Rank | Feature | Direction |
        |------|---------|-----------|
        | 1 | Prior inpatient visits | Higher → more risk |
        | 2 | Discharge destination | AMA → highest risk |
        | 3 | Prior utilization score | Higher → more risk |
        | 4 | Number of diagnoses | More → more risk |
        | 5 | Length of stay | Longer → more risk |
        | 6 | Insulin dose change | Change → more risk |
        | 7 | Age | Older → more risk |

        ---
        📁 [View source code on GitHub](#)  
        📊 [See Tableau dashboard](#)  
        📓 [Read project notebooks](#)
        """)
