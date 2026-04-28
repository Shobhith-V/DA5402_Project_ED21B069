"""
Airflow DAG — SepsisWatch Data Pipeline
--------------------------------------

This DAG orchestrates the data processing pipeline for SepsisWatch.

Each step in the pipeline (validation, imputation, feature engineering, and baseline computation)
is executed as an Airflow task. The actual processing logic is implemented in
src/pipeline/feature_pipeline.py — this file only handles orchestration.

You can trigger the pipeline:
- From the Airflow UI: http://localhost:8080
- Or via CLI: airflow dags trigger sepsis_data_pipeline
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Adding project root to path for imports
sys.path.insert(0, "/opt/airflow")

# Overriding ROOT for Docker environment
os.environ["SEPSIS_ROOT"] = "/opt/airflow"

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.feature_pipeline import (
    validate_raw_data,
    impute_features,
    extract_features,
    compute_baseline_stats,
)

# ── Default args applied to every task ───────────────────────────
default_args = {
    "owner":            "sepsis-watch",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ── DAG definition ────────────────────────────────────────────────
with DAG(
    dag_id="sepsis_data_pipeline",
    default_args=default_args,
    description="Ingest → impute → engineer features → baseline",
    schedule_interval=None,      # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["sepsis-watch", "data-pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="validate_raw_data",
        python_callable=validate_raw_data,
    )

    t2 = PythonOperator(
        task_id="impute_features",
        python_callable=impute_features,
    )

    t3 = PythonOperator(
        task_id="extract_features",
        python_callable=extract_features,
    )

    t4 = PythonOperator(
        task_id="compute_baseline_stats",
        python_callable=compute_baseline_stats,
    )

    t1 >> t2 >> t3 >> t4