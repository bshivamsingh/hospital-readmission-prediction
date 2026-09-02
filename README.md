# 🏥 Hospital Readmission Prediction — Diabetic Patients

> **Can we predict which diabetic patients will be readmitted within 30 days of discharge — and explain why?**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey) ![Sklearn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red) ![License](https://img.shields.io/badge/License-MIT-lightgrey) ![Tests](https://github.com/bshivamsingh/hospital-readmission-prediction/actions/workflows/ci.yml/badge.svg)

![App Screenshot](assets/app_screenshot.png)

## Who this is for

I built this because I'm interested in data work that actually helps people, and healthcare is one of the places that matters most. Diabetic patients readmitted to hospital within 30 days are a real, costly, and often preventable problem — for the patient as much as for the hospital. This project is a portfolio piece, not a clinical tool, but the goal behind it is genuine: give someone a rough, honest sense of their own readmission risk and the factors driving it, built on a model whose numbers I've personally verified rather than just written down.

---

## Business Problem

Hospitals in the US face **financial penalties** from CMS (Centers for Medicare & Medicaid Services) when patients are readmitted within 30 days of discharge. For diabetic patients — one of the highest-risk cohorts — readmission rates can exceed 20%.

This project builds an **end-to-end predictive analytics pipeline** to:
1. Identify key clinical and operational drivers of 30-day readmission
2. Score patients at discharge with a calibrated risk probability
3. Surface explainable, actionable insights for care teams

---

## Key Findings

- **Top readmission drivers**: number of inpatient visits in the prior year, discharge disposition (to home vs. SNF), number of diagnoses, and insulin dosage changes
- **XGBoost + SMOTE model** achieves **AUC-ROC: 0.588** on a held-out test set, vs. a logistic regression baseline of **0.575** — modest predictive power, honestly reported (see the correction note under Results Summary below)
- Patients with **3+ inpatient visits** in the prior year have a **2.8× higher readmission rate**
- Discharges to **skilled nursing facilities** are readmitted at the *highest* rate of any discharge group (13.9%, vs. 7.2% for home discharges) — likely a marker that SNF patients were sicker to begin with, not a protective effect (see `reports/key_findings.md` for the corrected breakdown and why an earlier, wrong version of this finding said the opposite)

---

## Dataset

**Source**: [Diabetes 130-US Hospitals (UCI ML Repository / Kaggle)](https://www.kaggle.com/datasets/brandao/diabetes)

- 100,000+ patient encounters across 130 US hospitals (1999–2008)
- Features: demographics, diagnoses (ICD-9), medications, lab results, prior utilization
- Target: `readmitted` — whether the patient was readmitted in <30 days

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data storage & profiling | SQLite, SQL |
| EDA & feature engineering | Python, pandas, seaborn, matplotlib |
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Dashboard | Tableau Public |
| Deployed app | Streamlit |

---

## Project Structure

```
hospital-readmission-prediction/
├── README.md
├── LICENSE
├── requirements.txt / requirements-dev.txt
├── .github/workflows/ci.yml        # Runs tests + a syntax check on every push/PR
├── data/
│   └── data_dictionary.md          # Feature descriptions
├── notebooks/
│   ├── 01_sql_profiling.ipynb      # SQL-based data quality checks
│   ├── 02_eda.ipynb                # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling_shap.ipynb      # XGBoost + SHAP explainability
├── sql/
│   └── readmission_queries.sql     # All SQL profiling & analysis queries
├── app/
│   ├── streamlit_app.py            # Live risk scoring app
│   ├── feature_engineering.py      # build_feature_vector() — unit-tested, imported by the app
│   ├── run_modeling.py             # Retrains the model, writes model_metrics.json fresh every run
│   └── model_metrics.json          # Real metrics from the last training run — read live by the app
├── tests/                          # pytest — feature engineering + model_metrics.json sanity checks
├── dashboard/
│   └── README.md                   # Tableau Public build spec (not yet published)
└── reports/
    └── key_findings.md
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/bshivamsingh/hospital-readmission-prediction
cd hospital-readmission-prediction

# 2. Install dependencies
pip install -r requirements.txt   # or requirements-dev.txt to also get pytest for running tests

# 3. Download the dataset
# Go to https://www.kaggle.com/datasets/brandao/diabetes
# Place diabetic_data.csv in the data/ folder

# 4. Run notebooks in order (01 → 04)
jupyter notebook

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Live Demo

🔗 [Streamlit App — Patient Risk Scorer](https://hospital-readmission-prediction-pro1.streamlit.app) — live, deployed on Streamlit Community Cloud  
📊 Tableau Dashboard — not yet published; see `dashboard/README.md` for the full build spec (views, calculated fields, data sources)

---

## Results Summary

| Model | AUC-ROC | Precision (High Risk) | Recall (High Risk) |
|---|---|---|---|
| Logistic Regression (baseline) | 0.575 | — | — |
| XGBoost + SMOTE (deployed model) | 0.588 | 0.155 | 0.093 |

> **Note**: For clinical use cases, recall matters more than precision — missing a high-risk patient is costlier than a false alarm. At a 0.5 probability threshold this model's recall on the high-risk class is low (0.093); `app/run_modeling.py` also reports precision/recall at a lower, recall-favoring threshold (0.3) for exactly this reason — see `app/model_metrics.json` after a training run, or the app's "About this model" panel.
>
> An earlier version of this README and app claimed **AUC-ROC 0.74** — that number was never real. See "Data Integrity & Lessons Learned" below for the full story of what was wrong and how it was caught.

---

## Data Integrity & Lessons Learned

This section exists because this project's numbers were wrong twice, in two different ways, and I think how that happened — and how it got caught — is more useful to show than to hide.

**The model's real AUC-ROC is 0.588, not the 0.74 originally claimed.** The 0.74 figure was never produced by any executed code. The notebook (`notebooks/04_modeling_shap.ipynb`) that supposedly measured it fails on an `ImportError` in its very first cell — every cell after that has `execution_count: None` in the saved notebook file, meaning it never ran. The number was typed into a markdown "Results Summary" cell by hand. Separately, the script that actually trains the deployed model (`app/run_modeling.py`) was silently dropping three engineered features the feature-engineering notebook builds (`diag1_category`, `comorbidity_count`, `discharge_risk_group`), so even a genuine run of the old script would have trained on a weaker feature set than intended. Both are now fixed: the training script builds the full feature set, and every number it produces is written live to `app/model_metrics.json` and read by the app — nothing is hand-typed.

**A "Key Finding" about discharge destination was backwards.** `reports/key_findings.md` originally claimed patients discharged to a Skilled Nursing Facility (SNF) had a *lower* 30-day readmission rate than those discharged home, calling it "counter-intuitive." Recomputing it directly from the data shows the opposite: SNF discharges have the *highest* readmission rate of any group (13.9%), not the lowest, and home discharges have the lowest (7.2%) — likely because SNF patients were sicker to begin with, which is the intuitive explanation, not a counter-intuitive one.

**How these were caught**: by not trusting a number just because it was written down, and instead re-deriving it from the data or the notebook's own saved execution metadata every time it was used for something that mattered (fixing the app, writing this README, answering "rate this project"). That's a slower way to work, but it's the difference between a project that looks credible and one that actually is — which matters more here than in most portfolio projects, because the subject is people's health.

**The headline "SHAP explainability" feature didn't actually work.** While making these fixes, I also found that the app's SHAP waterfall chart was silently broken — the code that was supposed to build the SHAP explainer never actually built one (it passed `None` instead), so every prediction's explanation quietly failed and fell back to an error message instead of a chart. It's fixed now (`app/streamlit_app.py`'s `load_artifacts()` builds a real `shap.TreeExplainer`), but it's a good reminder that a feature listed in a README isn't verified just because the code path exists — it has to actually be exercised and checked.

---

## Limitations & Ethical Considerations

This is a portfolio and educational project, **not a clinical decision-support tool**, and shouldn't be used to make real decisions about a real patient's care. A few things worth being explicit about:

- **Model performance is modest.** AUC-ROC 0.588 is only somewhat better than chance (0.5). At the default 0.5 threshold, recall on the high-risk class is low (0.093) — the model misses most patients who are actually readmitted. It should not be trusted as a reliable individual risk estimate.
- **Race and age are used as model inputs.** They're genuine predictors in this dataset, but using demographic attributes in a health risk model raises fairness questions this project doesn't attempt to fully address — a production system would need a real fairness/bias audit (e.g., checking for disparate error rates across race and age groups) before this would be responsible to deploy.
- **The training data is a public research dataset** (Diabetes 130-US Hospitals, 1999–2008, UCI/Kaggle), not current clinical data, and reflects the practices and population of hospitals from that era — it may not generalize to a different hospital, patient population, or time period.
- **Calibration is unverified for real-world use.** The app reports a Brier score and a reliability table (see `app/model_metrics.json`), which is a start, but "42% risk" from this app should be read as a rough model output, not a validated clinical probability.

---

## Testing & CI

Run the test suite locally with `pytest tests/ -v` (see `requirements-dev.txt`). GitHub Actions runs the same tests, plus a syntax check on the app, on every push and pull request — see `.github/workflows/ci.yml`.

---

## License

MIT — see [LICENSE](LICENSE). Free to use, fork, and build on; just don't use it as an actual clinical tool (see Limitations above).
