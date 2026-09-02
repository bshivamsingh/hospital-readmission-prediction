"""
run_modeling.py — Final fixed version
Explicitly encodes ALL string columns before modeling.
Feature engineering matches notebooks/03_feature_engineering.ipynb (diag1_category,
comorbidity_count, discharge_risk_group) and hyperparameters match
notebooks/04_modeling_shap.ipynb, which is what originally produced the ~0.74 AUC —
the earlier version of this script skipped those three engineered features entirely
(it dropped diag_1/2/3 with no replacement) and trained a materially weaker model.
Run: /opt/homebrew/bin/python3.11 app/run_modeling.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, joblib, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE

SEED = 42

print("[1/7] Loading data...")
df = pd.read_csv('data/diabetic_first_encounter.csv')
df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
print(f"      {df.shape[0]:,} rows | readmission rate: {df['readmitted_30'].mean():.1%}")

print("[2/7] Engineering diagnosis-based features (before dropping diag_1/2/3)...")

def icd9_category(code):
    """Map raw ICD-9 code to clinical category (matches notebook 03)."""
    if pd.isna(code) or code == '?':
        return 'Unknown'
    code = str(code)
    if code.startswith('V'):
        return 'Supplementary'
    if code.startswith('E'):
        return 'External cause'
    try:
        num = float(code)
    except ValueError:
        return 'Other'
    if 390 <= num <= 459 or num == 785:
        return 'Circulatory'
    elif 460 <= num <= 519 or num == 786:
        return 'Respiratory'
    elif 520 <= num <= 579 or num == 787:
        return 'Digestive'
    elif num == 250:
        return 'Diabetes'
    elif 800 <= num <= 999:
        return 'Injury'
    elif 710 <= num <= 739:
        return 'Musculoskeletal'
    elif 580 <= num <= 629 or num == 788:
        return 'Genitourinary'
    elif 140 <= num <= 239:
        return 'Neoplasm'
    elif 290 <= num <= 319:
        return 'Mental'
    else:
        return 'Other'

def discharge_risk(disp_id):
    """Group disposition IDs into clinical risk tiers (matches notebook 03)."""
    home_dispositions = [1, 2, 8]
    snf_dispositions  = [3, 5, 12, 14]
    hha_dispositions  = [6, 10]
    ama_dispositions  = [7]
    if disp_id in ama_dispositions:
        return 'High risk (AMA)'
    elif disp_id in home_dispositions:
        return 'Moderate (home)'
    elif disp_id in snf_dispositions:
        return 'Lower (SNF/rehab)'
    elif disp_id in hha_dispositions:
        return 'Lower (home health)'
    else:
        return 'Other'

df['comorbidity_count'] = (
    df[['diag_1', 'diag_2', 'diag_3']]
    .apply(lambda col: col.notna() & (col != '?'))
    .sum(axis=1)
)
df['diag1_category'] = df['diag_1'].apply(icd9_category)
df['discharge_risk_group'] = df['discharge_disposition_id'].apply(discharge_risk)

print("[3/7] Dropping unused columns...")
DROP = ['encounter_id','patient_nbr','weight','payer_code','readmitted','diag_1','diag_2','diag_3']
df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)

print("[4/7] Engineering remaining features...")
age_map = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,
           '[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
df['age_numeric'] = df['age'].map(age_map).fillna(65)
df['prior_utilization_score'] = df['number_inpatient']*3 + df['number_emergency']*2 + df['number_outpatient']
df['high_lab_burden']     = (df['num_lab_procedures'] > df['num_lab_procedures'].quantile(0.75)).astype(int)
df['admitted_from_er']    = (df['admission_source_id'] == 7).astype(int)
df['emergency_admission'] = (df['admission_type_id'] == 1).astype(int)
df['a1c_abnormal']        = df['A1Cresult'].isin(['>7','>8']).astype(int)
df['a1c_tested']          = df['A1Cresult'].notna().astype(int)  # raw data uses NaN, not the string 'None', for untested

print("[5/7] Encoding medication and categorical columns...")
MED_COLS = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
            'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
            'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','insulin',
            'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone']
MED_MAP = {'No':0,'Steady':1,'Up':2,'Down':3}
for col in MED_COLS:
    if col in df.columns:
        df[col] = df[col].map(MED_MAP).fillna(0).astype(int)

df['medication_changes'] = (df[[c for c in MED_COLS if c in df.columns]].isin([2,3])).sum(axis=1)
df['insulin_changed']    = df['insulin'].isin([2,3]).astype(int)
df['num_active_meds']    = (df[[c for c in MED_COLS if c in df.columns]] > 0).sum(axis=1)

# admission_type_id / admission_source_id / discharge_disposition_id are left as raw
# numeric codes (not one-hot encoded) — matches notebook 03, which only one-hot encodes
# the categorical fields below and lets discharge_risk_group carry the discharge signal.
CAT_COLS = ['race','gender','age','medical_specialty','max_glu_serum','A1Cresult',
            'diag1_category','discharge_risk_group']
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)
df = pd.get_dummies(df, columns=[c for c in CAT_COLS if c in df.columns], drop_first=True, dtype=int)

print("[6/7] Final cleanup...")
cols_to_drop = []
for col in df.columns:
    if col == 'readmitted_30': continue
    try:
        df[col] = pd.to_numeric(df[col], errors='raise')
    except Exception:
        cols_to_drop.append(col)
if cols_to_drop:
    print(f"      Dropping non-numeric: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
df.fillna(0, inplace=True)
df.columns = df.columns.str.replace("[^A-Za-z0-9_]", "_", regex=True)
print(f"      Final shape: {df.shape}")

print("[7/7] Training model...")
X = df.drop(columns=['readmitted_30'])
y = df['readmitted_30']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=SEED,stratify=y)
X_res,y_res = SMOTE(random_state=SEED).fit_resample(X_train,y_train)
print(f"      After SMOTE: {X_res.shape} | positive: {y_res.mean():.1%}")

model = xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,min_child_weight=5,
    gamma=0.1,reg_alpha=0.1,reg_lambda=1.0,
    eval_metric='auc',random_state=SEED,n_jobs=-1,verbosity=0)
model.fit(X_res,y_res,eval_set=[(X_test,y_test)],verbose=50)

y_prob = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_prob)
y_pred = (y_prob >= 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['Not Readmitted','Readmitted <30d'], output_dict=True)
precision_high_risk = report['Readmitted <30d']['precision']
recall_high_risk = report['Readmitted <30d']['recall']
print(f"\n      AUC-ROC: {auc:.3f}")
print(f"      High-risk precision: {precision_high_risk:.3f} | recall: {recall_high_risk:.3f}")

os.makedirs('reports',exist_ok=True)
os.makedirs('app',exist_ok=True)

fig,ax = plt.subplots(figsize=(7,5))
fpr,tpr,_ = roc_curve(y_test,y_prob)
ax.plot(fpr,tpr,color='#1D9E75',lw=2,label=f'XGBoost (AUC={auc:.3f})')
ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve',fontweight='bold'); ax.legend()
plt.tight_layout(); plt.savefig('reports/10_roc_curve.png',dpi=150,bbox_inches='tight'); plt.close()

fi = pd.DataFrame({'feature':X.columns,'importance':model.feature_importances_})
fi = fi.sort_values('importance',ascending=True).tail(15)
fig,ax = plt.subplots(figsize=(9,6))
ax.barh(fi['feature'],fi['importance'],color='#1D9E75',alpha=0.85)
ax.set_title('Top 15 features',fontweight='bold')
plt.tight_layout(); plt.savefig('reports/12_feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()

joblib.dump(model,'app/xgb_readmission_model.pkl')
joblib.dump(X.columns.tolist(),'app/feature_names.pkl')
with open('app/model_metrics.json', 'w') as f:
    json.dump({
        'auc_roc': round(float(auc), 3),
        'precision_high_risk': round(float(precision_high_risk), 3),
        'recall_high_risk': round(float(recall_high_risk), 3),
        'n_features': int(X.shape[1]),
        'seed': SEED,
    }, f, indent=2)

print("\n" + "="*50)
print(f"  DONE!  AUC-ROC = {auc:.3f}")
print("  Model saved to app/")
print("  Charts saved to reports/")
print("="*50)
