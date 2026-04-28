"""
FastAPI Service — Sepsis Watch
================================
Serves sepsis risk predictions via REST API.
Instruments Prometheus metrics for monitoring.
Runs background jobs for feedback loop and drift detection.

Endpoints:
  POST /predict   — score one patient's hourly row
  GET  /health    — liveness check
  GET  /ready     — readiness check
  GET  /metrics   — Prometheus metrics

Usage (local):
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from scipy import stats
from starlette.responses import Response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent.parent
CONFIG    = ROOT / "src" / "config.json"
BASELINE  = ROOT / "data" / "baseline"
CONFIRMED = ROOT / "data" / "confirmed"
PRED_LOG  = CONFIRMED / "prediction_log.jsonl"
POOL_PATH = CONFIRMED / "hospital_B_confirmed.parquet"

CONFIRMED.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────
with open(CONFIG) as f:
    CFG = json.load(f)

with open(BASELINE / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

with open(BASELINE / "hospital_a_baseline.json") as f:
    BASELINE_STATS = json.load(f)

# ── Patient history buffer ─────────────────────────────────────────
# Stores last 6 rows per patient to compute rolling statistics.
# Without history, rolling mean = current value (no trend signal).
# With history, model gets the temporal patterns it was trained on.
_patient_history = defaultdict(lambda: deque(maxlen=6))

# ── Prometheus metrics ─────────────────────────────────────────────
RISK_SCORE = Gauge(
    "sepsis_risk_score",
    "Most recent risk score",
    ["patient_id"]
)
PREDICTIONS_TOTAL = Counter(
    "sepsis_predictions_total",
    "Total predictions made",
    ["risk_tier"]
)
ROLLING_RECALL = Gauge(
    "model_rolling_recall",
    "Rolling recall vs SepsisLabel ground truth"
)
ROLLING_PRECISION = Gauge(
    "model_rolling_precision",
    "Rolling precision vs SepsisLabel ground truth"
)
DRIFT_KS_PVALUE = Gauge(
    "drift_ks_pvalue",
    "KS test p-value vs Hospital A baseline",
    ["feature"]
)
DRIFT_DETECTED = Counter(
    "drift_detected_total",
    "Times drift confirmed on 2+ features"
)
CONFIRMED_POOL_SIZE = Gauge(
    "confirmed_pool_size",
    "Rows in confirmed Hospital B pool"
)
RETRAINING_TRIGGERED = Counter(
    "retraining_triggered_total",
    "Retraining jobs triggered",
    ["outcome"]
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Model inference time only",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
)

# ── Model loading ──────────────────────────────────────────────────
_model       = None
_model_ready = False


def load_model():
    global _model, _model_ready
    try:
        model_path = BASELINE / "model_local.json"
        if not model_path.exists():
            log.error(f"Model not found at {model_path}")
            log.error("Run src/training/train.py first")
            return
        _model = xgb.XGBClassifier()
        _model.load_model(str(model_path))
        _model_ready = True
        log.info(f"Model loaded from {model_path}")
    except Exception as e:
        log.error(f"Model load failed: {e}")
        _model_ready = False


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="SepsisWatch API",
    description="Sepsis early-warning — real-time ICU risk scoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


# ── Request schema ─────────────────────────────────────────────────
# extra="allow" accepts all 77 engineered features dynamically.
# We don't list them all explicitly — feature_cols.json defines
# the exact set and order the model expects.
class PatientRow(BaseModel):
    model_config = {"extra": "allow"}
    patient_id:  str
    SepsisLabel: Optional[int] = None


class PredictionResponse(BaseModel):
    patient_id:   str
    risk_score:   float
    risk_tier:    str
    flagged:      bool
    top_features: list
    timestamp:    float


# ── Startup ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()
    start_background_jobs()
    log.info("SepsisWatch API started")


# ── Health and readiness ───────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/ready", tags=["ops"])
def ready():
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model": "sepsis-watch-classifier"}

@app.get("/patients", tags=["inference"])
def get_patients(limit: int = 10000):
    """
    Returns the most recently scored patients from the prediction log.
    Frontend polls this to show real Hospital B patients.
    """
    if not PRED_LOG.exists():
        return {"patients": []}

    try:
        lines   = PRED_LOG.read_text().strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]

        # Get latest entry per patient
        latest = {}
        for e in entries:
            pid = e["patient_id"]
            if pid not in latest or e["timestamp"] > latest[pid]["timestamp"]:
                latest[pid] = e

        patients = sorted(
            latest.values(),
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]

        return {
            "patients": [
                {
                    "patient_id":  p["patient_id"],
                    "risk_score":  p["risk_score"],
                    "risk_tier":   p["risk_tier"],
                    "flagged":     p["risk_score"] >= 0.3,
                    "top_features": [],
                    "timestamp":   p["timestamp"],
                    "features":    {
                        k: v for k, v in p.get("features", {}).items()
                        if k in CFG["vital_cols"]
                    }
                }
                for p in patients
            ]
        }
    except Exception as e:
        log.error(f"Error reading patients: {e}")
        return {"patients": []}

# ── Prediction ─────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(row: PatientRow):
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Extract all fields — PatientRow accepts any extra fields
    # Extract all fields from request
    row_dict = row.model_dump()
    row_dict.pop("patient_id", None)
    label = row_dict.pop("SepsisLabel", None)

    # Check what we actually received
    roll_features_received = {k:v for k,v in row_dict.items() if 'roll' in k}
    log.info(f"Rolling features in request: {len(roll_features_received)}")

    # Update history buffer
    _patient_history[row.patient_id].append(row_dict.copy())
    history = list(_patient_history[row.patient_id])

    # Only compute rolling stats if NOT already in payload
    for col in CFG["vital_cols"]:
        if f"{col}_roll_mean" not in row_dict:
            vals = [h.get(col) for h in history if h.get(col) is not None]
            row_dict[f"{col}_roll_mean"] = float(np.mean(vals)) if vals else 0.0
            row_dict[f"{col}_roll_std"]  = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Only compute drawn indicators if NOT already in payload
    for col in CFG["lab_cols"]:
        if f"{col}_drawn" not in row_dict:
            row_dict[f"{col}_drawn"] = 1 if row_dict.get(col) is not None else 0

    # Build DataFrame aligned to training order
    feature_df = pd.DataFrame([row_dict])
    for col in FEATURE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[FEATURE_COLS].fillna(0)

    # Inference — timed for Prometheus histogram
    t0         = time.time()
    risk_score = float(_model.predict_proba(feature_df)[0, 1])
    INFERENCE_LATENCY.observe(time.time() - t0)

    # Risk tier
    if risk_score >= 0.3:
        risk_tier = "high"
    elif risk_score >= 0.1:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    # Update Prometheus
    RISK_SCORE.labels(patient_id=row.patient_id).set(risk_score)
    PREDICTIONS_TOTAL.labels(risk_tier=risk_tier).inc()

    # Top contributing features — deviation from baseline
    top_features = []
    for col in CFG["vital_cols"]:
        val = row_dict.get(col)
        if val is not None and col in BASELINE_STATS:
            mean = BASELINE_STATS[col]["mean"]
            std  = BASELINE_STATS[col]["std"] or 1
            contribution = abs(val - mean) / std
            top_features.append({
                "name":         col,
                "value":        round(float(val), 2),
                "contribution": round(contribution, 3)
            })
    top_features = sorted(
        top_features, key=lambda x: x["contribution"], reverse=True
    )[:3]

    # Log prediction for feedback loop
    log_entry = {
        "patient_id":  row.patient_id,
        "timestamp":   time.time(),
        "risk_score":  risk_score,
        "risk_tier":   risk_tier,
        "sepsis_label": label,
        "features":    {k: float(v) if isinstance(v, (int, float)) else v
                       for k, v in row_dict.items()},
    }
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return PredictionResponse(
        patient_id   = row.patient_id,
        risk_score   = round(risk_score, 4),
        risk_tier    = risk_tier,
        flagged      = risk_score >= 0.3,
        top_features = top_features,
        timestamp    = time.time(),
    )


# ── Background jobs ────────────────────────────────────────────────
def feedback_loop():
    """
    Compares predictions against SepsisLabel ground truth.
    Updates rolling recall and precision Prometheus metrics.
    Accumulates confirmed rows for retraining pool.
    Runs every 5 minutes.
    """
    try:
        if not PRED_LOG.exists():
            return

        lines   = PRED_LOG.read_text().strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]

        labelled = [e for e in entries if e.get("sepsis_label") is not None]
        if len(labelled) < 50:
            return

        window = labelled[-1000:]
        y_true = np.array([e["sepsis_label"] for e in window])
        y_pred = np.array([1 if e["risk_score"] >= 0.3 else 0 for e in window])

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        recall    = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        ROLLING_RECALL.set(recall)
        ROLLING_PRECISION.set(precision)

        # Append to confirmed pool for retraining
        new_rows = []
        for e in labelled[-100:]:
            row = e["features"].copy()
            row["patient_id"]  = e["patient_id"]
            row["SepsisLabel"] = e["sepsis_label"]
            row["timestamp"]   = e["timestamp"]
            new_rows.append(row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if POOL_PATH.exists():
                existing = pd.read_parquet(POOL_PATH)
                combined = pd.concat([existing, new_df]).drop_duplicates(
                    subset=["patient_id", "timestamp"]
                )
            else:
                combined = new_df
            combined.to_parquet(POOL_PATH, index=False)
            CONFIRMED_POOL_SIZE.set(len(combined))

        log.info(
            f"Feedback loop: recall={recall:.3f} "
            f"precision={precision:.3f} "
            f"pool={len(combined) if new_rows else '—'}"
        )

    except Exception as e:
        log.error(f"Feedback loop error: {e}")


def drift_check():
    """
    KS test on rolling window of incoming features vs
    Hospital A baseline. Fires drift alert when 2+ features drift.
    Runs every 10 minutes.
    """
    try:
        if not PRED_LOG.exists():
            return

        lines   = PRED_LOG.read_text().strip().split("\n")
        entries = [json.loads(l) for l in lines[-500:] if l.strip()]

        if len(entries) < CFG["min_ks_window"]:
            log.info(
                f"Drift check: {len(entries)} entries, "
                f"need {CFG['min_ks_window']}"
            )
            return

        drift_count = 0
        for col in CFG["vital_cols"]:
            prod_vals = [
                e["features"].get(col)
                for e in entries
                if e["features"].get(col) is not None
            ]
            if len(prod_vals) < 100 or col not in BASELINE_STATS:
                continue

            b_mean = BASELINE_STATS[col]["mean"]
            b_std  = BASELINE_STATS[col]["std"] or 1
            ref    = np.random.normal(b_mean, b_std, len(prod_vals))

            stat, pvalue = stats.ks_2samp(prod_vals, ref)
            DRIFT_KS_PVALUE.labels(feature=col).set(pvalue)

            if pvalue < CFG["drift_threshold"]:
                drift_count += 1
                log.warning(f"Drift on {col}: p={pvalue:.4f}")

        if drift_count >= 2:
            DRIFT_DETECTED.inc()
            log.warning(f"DRIFT CONFIRMED: {drift_count} features")

    except Exception as e:
        log.error(f"Drift check error: {e}")

def check_retrain_trigger():
    """
    Automatically triggers retraining when:
    - Confirmed pool >= 1000 rows AND
    - Rolling recall < 0.85 (model degrading)
    Runs every 2 minutes.
    """
    try:
        # Check pool size first
        if not POOL_PATH.exists():
            return

        pool = pd.read_parquet(POOL_PATH)
        if len(pool) < CFG["min_confirmed_pool_for_retrain"]:
            log.info(
                f"Retrain check: pool {len(pool)} < "
                f"{CFG['min_confirmed_pool_for_retrain']} — skipping"
            )
            return

        # Check recall from prediction log
        if not PRED_LOG.exists():
            return

        lines    = PRED_LOG.read_text().strip().split("\n")
        entries  = [json.loads(l) for l in lines if l.strip()]
        labelled = [e for e in entries if e.get("sepsis_label") is not None]

        if len(labelled) < 50:
            return

        window = labelled[-1000:]
        y_true = np.array([e["sepsis_label"] for e in window])
        y_pred = np.array([1 if e["risk_score"] >= 0.3 else 0 for e in window])

        tp     = np.sum((y_pred == 1) & (y_true == 1))
        fn     = np.sum((y_pred == 0) & (y_true == 1))
        recall = tp / max(tp + fn, 1)

        log.info(f"Retrain check: pool={len(pool)} recall={recall:.3f}")

        if recall < CFG["recall_alert_threshold"]:
            log.warning(
                f"Recall {recall:.3f} below {CFG['recall_alert_threshold']} "
                f"— triggering auto-retrain"
            )

            import subprocess
            result = subprocess.run(
                ["python", "src/training/retrain.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                log.info("Auto-retrain completed — reloading model")
                RETRAINING_TRIGGERED.labels(outcome="promoted").inc()
                # Reload the model
                load_model()
            else:
                log.error(f"Auto-retrain failed:\n{result.stderr}")
                RETRAINING_TRIGGERED.labels(outcome="failed").inc()

    except Exception as e:
        log.error(f"Retrain trigger error: {e}")

def start_background_jobs():
    scheduler = BackgroundScheduler()
    scheduler.add_job(feedback_loop,         "interval", minutes=5,  id="feedback")
    scheduler.add_job(drift_check,           "interval", minutes=10, id="drift")
    scheduler.add_job(check_retrain_trigger, "interval", minutes=2,  id="retrain_check")
    scheduler.start()
    log.info("Background jobs: feedback(5m), drift(10m), retrain_check(2m)")