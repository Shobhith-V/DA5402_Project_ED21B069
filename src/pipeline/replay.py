"""
Hospital B Replay Script — Sepsis Watch
========================================
Streams Hospital B pre-engineered features through the API.
Uses hospital_B_features.parquet which has all 77 features
including rolling stats and missingness indicators —
exactly what the model was trained on.

Usage:
    python src/pipeline/replay.py
    python src/pipeline/replay.py --fast
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

ROOT        = Path(__file__).parent.parent.parent
DATA        = ROOT / "data" / "processed"
CONFIG      = ROOT / "src" / "config.json"
FEATURE_COLS_PATH = ROOT / "data" / "baseline" / "feature_cols.json"
API_URL     = "http://localhost:8000"

with open(CONFIG) as f:
    CFG = json.load(f)

with open(FEATURE_COLS_PATH) as f:
    FEATURE_COLS = json.load(f)


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


def replay(fast=False, n_patients=None, start_patient=0):
    # Load pre-engineered Hospital B features
    # This file has all 77 features the model expects:
    # raw vitals + rolling stats + missingness indicators
    path = DATA / "hospital_B_features.parquet"
    if not path.exists():
        log.error("hospital_B_features.parquet not found")
        log.error("Run src/pipeline/feature_pipeline.py first")
        return

    df      = pd.read_parquet(path)
    pid_col = CFG["patient_id_col"]
    target  = CFG["target_col"]

    patients = df[pid_col].unique()
    patients = patients[start_patient:]      # skip already-replayed patients
    if n_patients:
        patients = patients[:n_patients]

    log.info(f"Starting from patient index {start_patient}")

    log.info(f"Replaying {len(patients):,} Hospital B patients")
    log.info(f"Features per row: {len(FEATURE_COLS)}")
    log.info(f"Mode: {'fast' if fast else 'demo (100ms delay)'}")
    log.info("─" * 60)

    total_sent    = 0
    total_flagged = 0
    errors        = 0

    for patient_id in patients:
        patient_df = df[df[pid_col] == patient_id].sort_values("ICULOS")

        for _, row in patient_df.iterrows():
            # Build payload with all engineered features
            payload = {"patient_id": str(patient_id)}

            for col in FEATURE_COLS:
                val = row.get(col)
                if pd.notna(val):
                    payload[col] = float(val)

            # Include ground truth label for feedback loop
            # In real deployment this field would be absent
            label = row.get(target)
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
                        f"FLAGGED  {str(patient_id):<12} "
                        f"score={result['risk_score']:.4f}{feat_str}"
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
    log.info(f"  Flagged: {total_flagged:,} "
             f"({total_flagged/max(total_sent,1)*100:.1f}%)")
    log.info(f"  Errors:  {errors}")
    log.info(f"  Check Grafana: http://localhost:3001")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",          action="store_true")
    parser.add_argument("--n-patients",    type=int, default=None)
    parser.add_argument("--start-patient", type=int, default=0,
                        help="Start from this patient index (0-based)")
    args = parser.parse_args()

    if not check_api_ready():
        exit(1)

    replay(fast=args.fast, n_patients=args.n_patients, start_patient=args.start_patient)