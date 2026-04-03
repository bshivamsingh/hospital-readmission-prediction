# Key Findings — Hospital Readmission Prediction

## Project Summary

**Business question**: Which diabetic patients are at highest risk of being readmitted to hospital within 30 days of discharge?

**Why it matters**: CMS (Centers for Medicare & Medicaid Services) financially penalises hospitals for excess readmissions under the Hospital Readmissions Reduction Program (HRRP). A $1 readmission costs on average $15,000+; preventing even a small percentage has significant financial impact.

---

## Finding 1 — Prior Utilisation Is the Strongest Predictor

Patients with 3+ inpatient visits in the prior year are readmitted at **2.8× the rate** of those with no prior inpatient visits.

| Prior Inpatient Visits | 30-day Readmission Rate |
|---|---|
| 0 | ~8.5% |
| 1 | ~14.2% |
| 2 | ~18.6% |
| 3+ | ~23.9% |

**Implication**: Care management programs should be targeted at high-utilisation patients *before* they are discharged, not reactively after readmission.

---

## Finding 2 — Discharge Destination Matters More Than Expected

Counter-intuitively, patients discharged to Skilled Nursing Facilities (SNFs) have *lower* readmission rates than those discharged home. This is likely because SNF patients are actively monitored.

| Discharge Destination | 30-day Readmission Rate |
|---|---|
| Left Against Medical Advice (AMA) | ~22.1% |
| Home | ~11.8% |
| Home Health Agency | ~10.9% |
| Skilled Nursing Facility | ~8.7% |

**Implication**: "Home" should not mean "discharged and forgotten." Home patients need structured follow-up.

---

## Finding 3 — HbA1c Testing Is Underutilised

Only ~16% of patients had an HbA1c test during their hospital stay — despite this being a key diabetes management metric. Among those tested with abnormal results (>8), readmission rates were elevated, but testing itself was associated with lower readmission rates (selection effect: better-managed patients are more likely to be tested).

**Implication**: Increasing systematic HbA1c testing during admission could identify high-risk patients earlier.

---

## Finding 4 — Class Imbalance Requires Careful Handling

30-day readmissions make up only ~11.2% of encounters. A naive model that predicts "never readmitted" achieves 88.8% accuracy — useless clinically. We addressed this with SMOTE oversampling, and evaluated using AUC-ROC and recall rather than accuracy.

**Lesson for analysts**: Always check your class distribution before modelling. Accuracy is a misleading metric for imbalanced clinical outcomes.

---

## Model Performance

| Model | AUC-ROC | Precision (pos class) | Recall (pos class) |
|---|---|---|---|
| Naive baseline (all negative) | 0.50 | — | 0.0% |
| Logistic Regression | 0.67 | 0.41 | 55.2% |
| XGBoost (SMOTE) | 0.74 | 0.48 | 61.4% |

**At a 0.3 probability threshold** (optimised for recall in a clinical setting):
- Recall: 71% of actual readmissions are flagged
- Precision: 31% of flagged patients are actually readmitted
- The tradeoff: for every 3 patients flagged for intervention, ~1 truly needs it

This tradeoff is acceptable in healthcare — the cost of a missed readmission far exceeds the cost of an unnecessary phone call.

---

## Top SHAP Features (XGBoost)

1. `number_inpatient` — prior inpatient utilisation
2. `discharge_risk_group_High risk (AMA)` — left against medical advice
3. `prior_utilization_score` — weighted composite utilisation
4. `number_diagnoses` — clinical complexity
5. `time_in_hospital` — length of stay
6. `insulin_changed` — insulin dose adjusted during stay
7. `age_numeric` — patient age
8. `emergency_admission` — admitted via emergency
9. `num_medications` — number of active medications
10. `a1c_tested` — whether HbA1c was tested (negative weight — protective)

---

## Limitations

- **Historical data**: Model trained on 1999–2008 data; clinical practices have changed
- **Missing features**: Weight (97% missing) and payer code (40% missing) were excluded
- **No cost data**: Could not model cost-effectiveness of interventions
- **Single encounter**: Only first encounter per patient used — doesn't capture full longitudinal history
- **External validity**: Trained on US hospital data — may not generalise internationally

---

## Next Steps (Production Recommendations)

1. Retrain on more recent data (post-2010)
2. Add real-time EHR integration (FHIR API)
3. Incorporate social determinants of health (SDOH) features
4. A/B test: randomise care management interventions to measure actual impact
5. Build a fairness audit — check model performance across race and gender subgroups
