# Tableau Dashboard — Hospital Readmission Risk

## Live Dashboard
🔗 **[View on Tableau Public](#)** ← Paste your Tableau Public URL here after publishing

---

## Dashboard Structure

The dashboard has **3 views** (tabs):

### View 1 — Executive Summary
- **KPI cards**: Overall readmission rate, total encounters, high-risk patient count
- **Readmission rate by age group** (bar + line combo)
- **Readmission by discharge disposition** (horizontal bar, sorted by rate)
- **Filter**: Date range, age group, admission type

### View 2 — Risk Factor Deep Dive
- **Prior utilisation vs readmission** (scatter with trend line)
- **Medication change impact** (grouped bar: insulin, metformin, glipizide)
- **HbA1c and glucose test results** (heat map by test result × readmission)
- **Filter**: Risk tier (low / moderate / high), discharge destination

### View 3 — Model Results
- **SHAP feature importance** (horizontal bar — top 15 features)
- **Risk score distribution** (histogram split by true readmission status)
- **Confusion matrix** (4-quadrant highlight table)
- **Model comparison table** (LR vs XGBoost metrics)

---

## How to Build This in Tableau

### Data sources needed (from your notebooks):
1. `diabetic_first_encounter.csv` — raw features + readmission label
2. `model_predictions.csv` — patient ID, risk score, true label (export from notebook 04)
3. `shap_summary.csv` — feature name, mean_abs_shap (export from notebook 04)

### Export SHAP summary for Tableau
Add this to notebook 04:
```python
shap_df = pd.DataFrame({
    'feature': X_sample.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False).head(15)
shap_df.to_csv('../data/shap_summary.csv', index=False)

# Export predictions
pred_df = pd.DataFrame({
    'true_label': y_test.values,
    'predicted_prob': y_prob_xgb,
    'predicted_class': (y_prob_xgb >= 0.5).astype(int)
})
pred_df.to_csv('../data/model_predictions.csv', index=False)
```

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
