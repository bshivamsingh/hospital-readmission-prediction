# Tableau Dashboard — Hospital Readmission Risk

## Live Dashboard
🔗 **[View on Tableau Public](https://public.tableau.com/app/profile/b.shivam.singh/viz/HospitalReadmissionDashboard_17888227883740/View1ExecutiveSummary)**

---

## Dashboard Structure

The dashboard has **3 views** (tabs):

### View 1 — Executive Summary
- **KPI cards**: Overall readmission rate, total encounters, high-risk patient count
- **Readmission rate by age group** (bar + line combo)
- **Readmission by discharge disposition** (horizontal bar, sorted by rate)
- **Filter**: Age group, admission type (date range filter dropped — see note below)

### View 2 — Risk Factor Deep Dive
- **Prior utilisation vs readmission** (scatter with trend line, aggregated by prior-visit count so it isn't dominated by the ~74% of patients with zero prior visits)
- **Medication change impact** (three small-multiple bar charts — insulin, metformin, glipizide — instead of one grouped bar, to avoid a Tableau column-pivot step; same information, lower risk of breaking the workbook)
- **HbA1c and glucose test results** (heat map by test result × readmission, with an explicit "Not Tested" category since 82%/95% of patients were never tested for these)
- **Filter**: Risk tier (low / moderate / high), discharge destination

### View 3 — Model Results
- **SHAP feature importance** (horizontal bar — top 15 features)
- **Risk score distribution** (histogram split by true readmission status)
- **Confusion matrix** (4-quadrant highlight table)
- **Threshold trade-off** (precision vs. recall at the 0.5 default threshold vs. a 0.3 recall-optimized threshold) — replaces the "Model comparison table (LR vs XGBoost)" originally planned here; see note below for why.

---

## Notes on deviations from the original spec

- **No date range filter on View 1.** The dataset has no date or timestamp field at all (only `admission_type_id`, `admission_source_id`, and `time_in_hospital`, none of which are calendar dates), so this filter was dropped rather than faked with a placeholder.
- **No LR vs. XGBoost comparison table on View 3.** The only "logistic regression" numbers that ever existed for this project (AUC 0.67 in `reports/key_findings.md`) come from `notebooks/04_modeling_shap.ipynb`, which — per the "Data Integrity & Lessons Learned" section above — fails on an `ImportError` in its first cell and never actually ran; that number was hand-typed, exactly the kind of thing this project already caught and fixed once for the model's real AUC (0.588). Rather than repeat that mistake, View 3 instead shows a real, derivable-from-data threshold trade-off chart built directly from `model_predictions.csv`, computed live in Tableau with no hand-typed numbers.

---

## How to Build This in Tableau

### Data sources needed (from your notebooks):
1. `diabetic_first_encounter.csv` — raw features + readmission label
2. `model_predictions.csv` — patient ID, risk score, true label (export from notebook 04)
3. `shap_summary.csv` — feature name, mean_abs_shap (export from notebook 04)

### Export SHAP summary and predictions for Tableau
**Already done — these are real, generated files, not something you need to build.**
`app/run_modeling.py` step 9/9 writes `data/model_predictions.csv` (every held-out test
patient's true label, predicted probability, and predicted class at the 0.5 threshold)
and `data/shap_summary.csv` (the top 15 features by mean |SHAP value|) on every run, the
same way it already writes `app/model_metrics.json` — nothing here is hand-exported once
and left to drift out of sync with the model.

(This intentionally lives in `run_modeling.py`, not `notebooks/04_modeling_shap.ipynb` —
see the README's "Data Integrity & Lessons Learned" section for why that notebook isn't
the trusted source of anything in this project.)

Just run:
```bash
python app/run_modeling.py
```
and both CSVs will be sitting in `data/`, ready to connect to in Tableau.

### Tableau connection steps:
1. Open Tableau Desktop or Tableau Public (free)
2. Connect to Text File → select `diabetic_first_encounter.csv`
3. Add additional data sources: `shap_summary.csv`, `model_predictions.csv`
4. Build calculated fields:
   - `Readmitted_Binary` = IF [readmitted] = '<30' THEN 1 ELSE 0 END
   - `Risk_Tier` = IF [predicted_prob] < 0.15 THEN 'Low' ELSEIF [predicted_prob] < 0.25 THEN 'Moderate' ELSE 'High' END
5. Publish to Tableau Public (free) → paste the URL in this README

### Recommended colour palette:
- High risk: `#A32D2D`
- Moderate risk: `#854F0B`
- Low risk: `#27500A`
- Primary accent: `#1D9E75`
- Secondary: `#378ADD`
