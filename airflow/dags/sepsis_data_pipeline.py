"""
Airflow DAG — Sepsis Watch Data Pipeline
=========================================
Wraps feature_pipeline.py functions as Airflow tasks.
The actual logic lives in src/pipeline/feature_pipeline.py —
this file is purely orchestration.

Trigger manually:
  airflow dags trigger sepsis_data_pipeline

Or from Airflow UI at http://localhost:8080
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Make src importable inside Airflow container
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

    # Strict sequential dependency
    t1 >> t2 >> t3 >> t4