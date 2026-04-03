-- ============================================================
-- Hospital Readmission Analytics — SQL Profiling Queries
-- Dataset: Diabetes 130-US Hospitals
-- Run in: SQLite (via Python), PostgreSQL, or DBeaver
-- ============================================================
-- HOW TO USE: Run notebook 01_sql_profiling.ipynb which loads
-- the CSV into SQLite and executes these queries via pandas.
-- ============================================================


-- ------------------------------------------------------------
-- SECTION 1: DATA QUALITY & PROFILING
-- ------------------------------------------------------------

-- 1.1 Row count and distinct patients
SELECT
    COUNT(*)                        AS total_encounters,
    COUNT(DISTINCT patient_nbr)     AS distinct_patients,
    COUNT(DISTINCT encounter_id)    AS distinct_encounters,
    ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT patient_nbr), 2) AS avg_encounters_per_patient
FROM diabetic_data;


-- 1.2 Missing values per column (key columns)
-- SQLite treats '?' as a string — replace with NULL during ingestion
SELECT
    SUM(CASE WHEN race = '?'   OR race IS NULL   THEN 1 ELSE 0 END) AS missing_race,
    SUM(CASE WHEN weight = '?' OR weight IS NULL  THEN 1 ELSE 0 END) AS missing_weight,
    SUM(CASE WHEN diag_1 = '?' OR diag_1 IS NULL  THEN 1 ELSE 0 END) AS missing_diag1,
    SUM(CASE WHEN diag_2 = '?' OR diag_2 IS NULL  THEN 1 ELSE 0 END) AS missing_diag2,
    SUM(CASE WHEN diag_3 = '?' OR diag_3 IS NULL  THEN 1 ELSE 0 END) AS missing_diag3,
    SUM(CASE WHEN payer_code = '?' OR payer_code IS NULL THEN 1 ELSE 0 END) AS missing_payer,
    COUNT(*) AS total_rows
FROM diabetic_data;


-- 1.3 Target variable distribution
SELECT
    readmitted,
    COUNT(*)                                            AS encounter_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_of_total
FROM diabetic_data
GROUP BY readmitted
ORDER BY encounter_count DESC;


-- 1.4 Class imbalance check (binary target: <30 days vs. other)
SELECT
    CASE WHEN readmitted = '<30' THEN 'Readmitted <30 days'
         ELSE 'Not readmitted <30 days' END AS readmission_class,
    COUNT(*)                                            AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
FROM diabetic_data
GROUP BY readmission_class;


-- ------------------------------------------------------------
-- SECTION 2: DEMOGRAPHIC ANALYSIS
-- ------------------------------------------------------------

-- 2.1 Readmission rate by age group
SELECT
    age                                                          AS age_group,
    COUNT(*)                                                     AS total_encounters,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END)         AS readmitted_30,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY age
ORDER BY age;


-- 2.2 Readmission rate by gender
SELECT
    gender,
    COUNT(*)                                                     AS total,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END)         AS readmitted_30,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
WHERE gender != 'Unknown/Invalid'
GROUP BY gender;


-- 2.3 Readmission rate by race
SELECT
    CASE WHEN race = '?' THEN 'Unknown' ELSE race END           AS race,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY race
ORDER BY readmission_rate_pct DESC;


-- ------------------------------------------------------------
-- SECTION 3: CLINICAL & UTILIZATION ANALYSIS
-- ------------------------------------------------------------

-- 3.1 Readmission rate by discharge disposition
-- disposition_id 1=Home, 3=SNF, 6=Home Health Agency
SELECT
    discharge_disposition_id,
    COUNT(*)                                                     AS total,
    SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END)         AS readmitted_30,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY discharge_disposition_id
HAVING COUNT(*) > 200
ORDER BY readmission_rate_pct DESC;


-- 3.2 Readmission rate by number of prior inpatient visits
SELECT
    number_inpatient                                             AS prior_inpatient_visits,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY number_inpatient
ORDER BY number_inpatient;


-- 3.3 Average utilization metrics by readmission status
SELECT
    CASE WHEN readmitted = '<30' THEN 'Readmitted <30d' ELSE 'Other' END AS readmission_group,
    ROUND(AVG(time_in_hospital), 2)     AS avg_los_days,
    ROUND(AVG(num_medications), 2)      AS avg_medications,
    ROUND(AVG(num_lab_procedures), 2)   AS avg_lab_procedures,
    ROUND(AVG(number_diagnoses), 2)     AS avg_diagnoses,
    ROUND(AVG(number_inpatient), 2)     AS avg_prior_inpatient,
    ROUND(AVG(number_emergency), 2)     AS avg_prior_emergency
FROM diabetic_data
GROUP BY readmission_group;


-- 3.4 Time in hospital distribution
SELECT
    time_in_hospital                                             AS los_days,
    COUNT(*)                                                     AS encounters,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY time_in_hospital
ORDER BY time_in_hospital;


-- ------------------------------------------------------------
-- SECTION 4: MEDICATION ANALYSIS
-- ------------------------------------------------------------

-- 4.1 Insulin usage vs readmission
SELECT
    insulin,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY insulin
ORDER BY readmission_rate_pct DESC;


-- 4.2 Patients with medication dose changes vs readmission
-- (Any medication changed Up or Down)
SELECT
    CASE
        WHEN (metformin IN ('Up','Down') OR insulin IN ('Up','Down')
              OR glipizide IN ('Up','Down') OR glyburide IN ('Up','Down'))
        THEN 'Had medication change'
        ELSE 'No medication change'
    END AS medication_change,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY medication_change;


-- 4.3 HbA1c test result vs readmission
SELECT
    A1Cresult,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY A1Cresult
ORDER BY readmission_rate_pct DESC;


-- ------------------------------------------------------------
-- SECTION 5: DIAGNOSIS ANALYSIS
-- ------------------------------------------------------------

-- 5.1 Most common primary diagnoses (ICD-9 code groups)
-- This requires Python-side ICD-9 grouping — see notebook 03
-- Here we look at raw top codes
SELECT
    diag_1,
    COUNT(*)                                                     AS frequency,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
WHERE diag_1 != '?'
GROUP BY diag_1
HAVING COUNT(*) > 500
ORDER BY frequency DESC
LIMIT 20;


-- 5.2 Number of diagnoses vs readmission rate
SELECT
    number_diagnoses,
    COUNT(*)                                                     AS total,
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS readmission_rate_pct
FROM diabetic_data
GROUP BY number_diagnoses
ORDER BY number_diagnoses;


-- ------------------------------------------------------------
-- SECTION 6: PATIENT-LEVEL AGGREGATION
-- (For patients with multiple encounters — take first encounter)
-- ------------------------------------------------------------

-- 6.1 Flag patients with multiple encounters
SELECT
    patient_nbr,
    COUNT(*) AS encounter_count
FROM diabetic_data
GROUP BY patient_nbr
HAVING COUNT(*) > 1
ORDER BY encounter_count DESC
LIMIT 20;


-- 6.2 Create a deduplicated dataset (first encounter per patient)
-- Use this as the base for modeling to avoid data leakage
CREATE TABLE IF NOT EXISTS diabetic_first_encounter AS
SELECT *
FROM diabetic_data
WHERE encounter_id IN (
    SELECT MIN(encounter_id)
    FROM diabetic_data
    GROUP BY patient_nbr
);

SELECT COUNT(*) AS deduplicated_rows FROM diabetic_first_encounter;


-- ------------------------------------------------------------
-- SECTION 7: SUMMARY STATS FOR REPORT
-- ------------------------------------------------------------

-- 7.1 Overall readmission rate (the number you lead with)
SELECT
    ROUND(SUM(CASE WHEN readmitted = '<30' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 2)
    AS overall_readmission_rate_pct
FROM diabetic_data;


-- 7.2 Top 5 insights table (copy into your README)
SELECT '3+ prior inpatient visits'      AS segment, ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) AS readmission_rate FROM diabetic_data WHERE number_inpatient >= 3
UNION ALL
SELECT '0 prior inpatient visits',       ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) FROM diabetic_data WHERE number_inpatient = 0
UNION ALL
SELECT 'Insulin dose changed',           ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) FROM diabetic_data WHERE insulin IN ('Up','Down')
UNION ALL
SELECT 'Insulin steady / not prescribed',ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) FROM diabetic_data WHERE insulin NOT IN ('Up','Down')
UNION ALL
SELECT 'Age 70-80',                      ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) FROM diabetic_data WHERE age = '[70-80)'
UNION ALL
SELECT 'Age 20-30',                      ROUND(AVG(CASE WHEN readmitted='<30' THEN 1.0 ELSE 0 END)*100,2) FROM diabetic_data WHERE age = '[20-30)';
