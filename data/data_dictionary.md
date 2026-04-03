# Data Dictionary — Diabetes 130-US Hospitals Dataset

## Source
UCI Machine Learning Repository / Kaggle
https://www.kaggle.com/datasets/brandao/diabetes

## Download Instructions
1. Go to the Kaggle link above
2. Download `diabetic_data.csv` and `IDs_mapping.csv`
3. Place both files in this `data/` folder

---

## Target Variable

| Column | Type | Description |
|---|---|---|
| `readmitted` | string | `<30` = readmitted within 30 days, `>30` = readmitted after 30 days, `NO` = not readmitted. **We binarize this: 1 if `<30`, else 0** |

---

## Patient Demographics

| Column | Type | Description |
|---|---|---|
| `encounter_id` | int | Unique identifier for each encounter |
| `patient_nbr` | int | Unique patient identifier (patients can have multiple encounters) |
| `race` | string | Patient race (Caucasian, AfricanAmerican, Hispanic, Asian, Other) |
| `gender` | string | Male / Female / Unknown |
| `age` | string | Age in 10-year buckets: [0-10), [10-20), ... [90-100) |
| `weight` | string | Weight in pounds — highly missing (~97%), drop this column |

---

## Admission & Discharge

| Column | Type | Description |
|---|---|---|
| `admission_type_id` | int | 1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, 5=Not Available |
| `discharge_disposition_id` | int | Where patient went after discharge (1=Home, 3=SNF, 6=Home Health, etc.) |
| `admission_source_id` | int | Where the patient came from (7=Emergency Room, 1=Physician Referral, etc.) |
| `time_in_hospital` | int | Days from admission to discharge (1–14) |

---

## Utilization (Prior Visits)

| Column | Type | Description |
|---|---|---|
| `num_lab_procedures` | int | Number of lab tests performed |
| `num_procedures` | int | Number of non-lab procedures |
| `num_medications` | int | Number of distinct medications administered |
| `number_outpatient` | int | Outpatient visits in the prior year |
| `number_emergency` | int | Emergency visits in the prior year |
| `number_inpatient` | int | Inpatient visits in the prior year — **strong readmission predictor** |
| `number_diagnoses` | int | Number of diagnoses entered to the system |

---

## Diagnoses (ICD-9 Codes)

| Column | Type | Description |
|---|---|---|
| `diag_1` | string | Primary diagnosis (ICD-9 code) |
| `diag_2` | string | Secondary diagnosis |
| `diag_3` | string | Tertiary diagnosis |

We engineer grouped diagnosis categories from these (circulatory, respiratory, digestive, diabetes, etc.)

---

## Medications (23 columns)

Each medication column (e.g. `metformin`, `insulin`, `glipizide`) takes one of:
- `No` — not prescribed
- `Steady` — no change in dose
- `Up` — dose increased
- `Down` — dose decreased

Key medication columns: `insulin`, `metformin`, `glipizide`, `glyburide`, `pioglitazone`, `rosiglitazone`

---

## Lab Results

| Column | Type | Description |
|---|---|---|
| `max_glu_serum` | string | Glucose serum test result: None, Norm, >200, >300 |
| `A1Cresult` | string | HbA1c test result: None, Norm, >7, >8 |

---

## Engineered Features (created in notebook 03)

| Feature | Description |
|---|---|
| `prior_utilization_score` | Weighted sum of inpatient + emergency + outpatient visits |
| `comorbidity_count` | Count of non-null/non-unknown diagnosis codes |
| `medication_changes` | Number of medications with Up or Down dose change |
| `insulin_changed` | 1 if insulin dose was changed |
| `age_numeric` | Midpoint of age bucket (e.g. [30-40) → 35) |
| `discharge_to_snf` | 1 if discharge_disposition_id == 3 |
| `admitted_from_er` | 1 if admission_source_id == 7 |
| `diag1_category` | Grouped ICD-9 category for primary diagnosis |
| `high_lab_burden` | 1 if num_lab_procedures > 75th percentile |
