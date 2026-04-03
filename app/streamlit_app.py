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
    explainer= joblib.load('app/shap_explainer.pkl')
    return model, features, explainer

try:
    model, FEATURE_NAMES, explainer = load_artifacts()
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

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
    st.metric("AUC-ROC", "0.74")
with col_info3:
    st.metric("Dataset", "130K+ encounters")

st.divider()

if not MODEL_LOADED:
    st.warning("""
    ⚠️ **Model not loaded.** 

    To run this app with a real model:
    1. Run notebooks 01–04 in order to train and save the model
    2. Ensure `app/xgb_readmission_model.pkl` exists

    **For demo purposes**, the app will use a mock risk score based on input features.
    """)

def build_feature_vector(age, gender, race, admission_type, discharge_dest,
                          time_in_hospital, num_inpatient, num_emergency,
                          num_outpatient, num_diagnoses, num_medications,
                          num_lab_procs, num_procedures, insulin_status, a1c_result):
    """
    Build a feature vector that matches the model's expected input.
    In production, this would perfectly replicate the feature engineering pipeline.
    """
    age_map = {
        '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
        '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
    }

    discharge_risk_map = {
        "Home": "Moderate (home)",
        "Skilled Nursing Facility": "Lower (SNF/rehab)",
        "Home Health Agency": "Lower (home health)",
        "Left Against Medical Advice": "High risk (AMA)",
        "Other": "Other"
    }

    prior_util = num_inpatient * 3 + num_emergency * 2 + num_outpatient
    insulin_changed = 1 if insulin_status in ['Up', 'Down'] else 0
    a1c_abnormal = 1 if a1c_result in ['>7', '>8'] else 0
    admitted_er = 1 if admission_type == "Emergency" else 0

    # This is a simplified vector for demo
    # Full version would match all one-hot columns from notebook 03
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
        'prior_utilization_score': prior_util,
        'comorbidity_count': min(num_diagnoses, 3),
        'medication_changes': 1 if insulin_status in ['Up', 'Down'] else 0,
        'insulin_changed': insulin_changed,
        'num_active_meds': max(1, num_medications // 3),
        'high_lab_burden': 1 if num_lab_procs > 54 else 0,
        'admitted_from_er': admitted_er,
        'emergency_admission': admitted_er,
        'a1c_abnormal': a1c_abnormal,
        'a1c_tested': 1 if a1c_result != 'None' else 0,
    }
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
        insulin_status, a1c_result
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
        - **Training data**: 71,518 unique diabetic patients across 130 US hospitals
        - **Validation**: 5-fold stratified cross-validation
        - **Performance**: AUC-ROC 0.74 on held-out test set
        - **Explainability**: SHAP TreeExplainer for per-patient feature attribution
        """)

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
