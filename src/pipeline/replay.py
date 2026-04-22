"""
Hospital B Replay Script — Sepsis Watch
========================================
Simulates a live ICU feed by streaming Hospital B patient records
through the FastAPI /predict endpoint one hourly row at a time.

This script serves three purposes:
  1. Integration test — verifies the full pipeline end to end
  2. Demo script — run this during demonstration
  3. Drift trigger — Hospital B data causes drift to fire

Usage:
    # Normal (100ms between rows — good for demos)
    python src/pipeline/replay.py

    # Fast (no delay — triggers drift quickly for testing)
    python src/pipeline/replay.py --fast

    # Limit patients
    python src/pipeline/replay.py --n-patients 200
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "data" / "processed"
CONFIG   = ROOT / "src" / "config.json"
API_URL  = "http://localhost:8000"

with open(CONFIG) as f:
    CFG = json.load(f)


def check_api_ready(max_retries=15):
    for i in range(max_retries):
        try:
            r = requests.get(f"{API_URL}/ready", timeout=2)
            if r.status_code == 200:
                log.info("API is ready")
                return True
        except requests.exceptions.ConnectionError:
            pass
        log.info(f"Waiting for API... ({i+1}/{max_retries})")
        time.sleep(2)
    log.error("API not ready — is uvicorn running?")
    return False


def replay(fast=False, n_patients=None):
    path = DATA / "hospital_B_features.parquet"
    if not path.exists():
        log.error("Hospital B features not found — run feature_pipeline.py first")
        return

    df       = pd.read_parquet(path)
    pid_col  = CFG["patient_id_col"]
    patients = df[pid_col].unique()

    if n_patients:
        patients = patients[:n_patients]

    log.info(f"Replaying {len(patients):,} Hospital B patients")
    log.info(f"Mode: {'fast' if fast else 'demo (100ms delay)'}")
    log.info("─" * 60)

    total_sent    = 0
    total_flagged = 0
    errors        = 0

    for patient_id in patients:
        patient_df = df[df[pid_col] == patient_id].sort_values("ICULOS")

        for _, row in patient_df.iterrows():
            # Build payload with vital signs
            payload = {"patient_id": str(patient_id)}

            for col in CFG["vital_cols"]:
                val = row.get(col)
                if pd.notna(val):
                    payload[col] = float(val)

            # Add a few lab values if available
            for col in ["Lactate", "Glucose", "Creatinine", "WBC"]:
                val = row.get(col)
                if pd.notna(val):
                    payload[col] = float(val)

            # Demographics
            for col in ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]:
                val = row.get(col)
                if pd.notna(val):
                    payload[col] = float(val)

            if pd.notna(row.get("ICULOS")):
                payload["ICULOS"] = int(row["ICULOS"])

            # Include SepsisLabel for feedback loop
            # In real deployment this would be absent
            # In replay it enables real recall/precision calculation
            label = row.get(CFG["target_col"])
            if pd.notna(label):
                payload["SepsisLabel"] = int(label)

            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=5,
                )
                resp.raise_for_status()
                result = resp.json()

                total_sent += 1
                if result["flagged"]:
                    total_flagged += 1
                    top = result["top_features"]
                    feat_str = ""
                    if top:
                        feat_str = f"  [{top[0]['name']}={top[0]['value']}]"
                    log.warning(
                        f"FLAGGED  {patient_id:<12} "
                        f"score={result['risk_score']:.3f}{feat_str}"
                    )

                elif total_sent % 500 == 0:
                    rate = total_flagged / total_sent * 100
                    log.info(
                        f"{total_sent:>6} sent · "
                        f"{total_flagged} flagged ({rate:.1f}%) · "
                        f"{errors} errors"
                    )

            except requests.exceptions.RequestException as e:
                errors += 1
                if errors <= 3:
                    log.error(f"Request failed: {e}")

            if not fast:
                time.sleep(0.1)

        if not fast:
            time.sleep(0.2)

    log.info("─" * 60)
    log.info(f"Replay complete:")
    log.info(f"  Sent:    {total_sent:,}")
    log.info(f"  Flagged: {total_flagged:,} ({total_flagged/max(total_sent,1)*100:.1f}%)")
    log.info(f"  Errors:  {errors}")
    log.info(f"  Check Grafana: http://localhost:3001")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",       action="store_true")
    parser.add_argument("--n-patients", type=int, default=None)
    args = parser.parse_args()

    if not check_api_ready():
        exit(1)

    replay(fast=args.fast, n_patients=args.n_patients)