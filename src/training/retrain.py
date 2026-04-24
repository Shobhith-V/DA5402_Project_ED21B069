"""
Retrain Script — Sepsis Watch
================================
Triggered when drift is detected or recall drops below threshold.
Combines Hospital A training data with confirmed Hospital B pool,
trains a new model, evaluates against held-out set, and promotes
to Production if it beats the current model.

Usage:
    python src/training/retrain.py

    # Or via MLproject:
    mlflow run . -e retrain
"""

import json
import logging
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent.parent
DATA      = ROOT / "data" / "processed"
CONFIRMED = ROOT / "data" / "confirmed"
BASELINE  = ROOT / "data" / "baseline"
CONFIG    = ROOT / "src" / "config.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "sepsis-watch"
MODEL_NAME          = "sepsis-watch-classifier"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"

def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def load_combined_training_data(cfg):
    """
    Combines Hospital A training data with confirmed Hospital B pool.

    Why combine rather than replace:
    The model must still perform on Hospital A patients — replacing
    the training set with only Hospital B data would cause it to
    forget Hospital A patterns (catastrophic forgetting).
    Combining preserves Hospital A knowledge while adapting to B.

    Why we need minimum pool size:
    Retraining on too few Hospital B examples can degrade performance
    rather than improve it. We enforce a minimum before retraining.
    """
    log.info("Loading combined training data...")
    pid    = cfg["patient_id_col"]
    target = cfg["target_col"]

    # Hospital A training patients
    train_patients = pd.read_parquet(
        DATA / "train_hospital_a.parquet",
        columns=[pid]
    )[pid].unique()

    hosp_a_features = pd.read_parquet(DATA / "hospital_A_features.parquet")
    train_a = hosp_a_features[
        hosp_a_features[pid].isin(train_patients)
    ].copy()

    log.info(f"Hospital A training rows: {len(train_a):,}")

    # Confirmed Hospital B pool
    pool_path = CONFIRMED / "hospital_B_confirmed.parquet"
    if not pool_path.exists():
        log.warning("No confirmed pool found — retraining on Hospital A only")
        return train_a, None

    pool = pd.read_parquet(pool_path)
    log.info(f"Confirmed Hospital B pool: {len(pool):,} rows")

    # Check minimum pool size
    min_pool = cfg["min_confirmed_pool_for_retrain"]
    if len(pool) < min_pool:
        log.warning(
            f"Pool size {len(pool)} below minimum {min_pool} "
            f"— retraining on Hospital A only"
        )
        return train_a, None

    # Combine — align columns
    # Pool may have different columns than features parquet
    # Use only columns present in both
    common_cols = [c for c in train_a.columns if c in pool.columns]
    train_combined = pd.concat(
        [train_a[common_cols], pool[common_cols]],
        ignore_index=True
    )

    log.info(f"Combined training rows: {len(train_combined):,}")
    log.info(
        f"Hospital B contribution: "
        f"{len(pool)/len(train_combined)*100:.1f}%"
    )

    return train_combined, pool


def load_holdout_data(cfg, feature_cols):
    """Load held-out evaluation set — locked on day 1, never changes."""
    holdout_patients = pd.read_parquet(
        DATA / "held_out_eval.parquet",
        columns=[cfg["patient_id_col"]]
    )[cfg["patient_id_col"]].unique()

    features   = pd.read_parquet(DATA / "hospital_A_features.parquet")
    holdout_df = features[
        features[cfg["patient_id_col"]].isin(holdout_patients)
    ].copy()

    X = holdout_df[feature_cols].fillna(0)
    y = holdout_df[cfg["target_col"]]
    return X, y


def get_current_model_auroc():
    """
    Gets AUROC of current Production model from MLflow registry.
    Used as the baseline to beat before promoting the new model.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client  = mlflow.MlflowClient()
        versions = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )
        if not versions:
            log.warning("No Production model found — any model will be promoted")
            return 0.0

        run = client.get_run(versions[0].run_id)
        auroc = run.data.metrics.get("auroc", 0.0)
        log.info(f"Current Production model AUROC: {auroc:.4f}")
        return auroc

    except Exception as e:
        log.warning(f"Could not get current model AUROC: {e}")
        return 0.0


def compute_utility_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return float((tp - 0.05 * fp - 2.0 * fn) / max(tp + fn, 1))


def retrain():
    cfg      = load_config()
    git_hash = get_git_hash()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    log.info("=" * 60)
    log.info("RETRAIN PIPELINE — starting")
    log.info(f"Git commit: {git_hash}")
    log.info("=" * 60)

    # Load data
    train_df, pool = load_combined_training_data(cfg)

    # Feature columns — same as original training
    with open(BASELINE / "feature_cols.json") as f:
        feature_cols = json.load(f)

    meta_cols = cfg["meta_cols"] + ["ICULOS", "HospAdmTime"]
    feat_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feat_cols].fillna(0)
    y_train = train_df[cfg["target_col"]]

    # Load holdout
    X_holdout, y_holdout = load_holdout_data(cfg, feat_cols)

    # Recompute scale_pos_weight on combined data
    # Critical — the imbalance ratio changes when Hospital B data
    # is added because B has a different sepsis prevalence (5.7% vs 8.8%)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = round(float(n_neg / max(n_pos, 1)), 2)
    log.info(f"Recomputed scale_pos_weight: {scale_pos_weight}")

    # Get current model AUROC to beat
    current_auroc = get_current_model_auroc()

    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "aucpr",
        "random_state":     42,
        "n_jobs":           -1,
    }

    with mlflow.start_run(run_name=f"retrain-{git_hash}") as run:

        # Log params
        mlflow.log_params(params)
        mlflow.log_param("git_commit",        git_hash)
        mlflow.log_param("run_type",          "retrain")
        mlflow.log_param("hospital_train",    "A+B")
        mlflow.log_param("n_train_rows",      len(X_train))
        mlflow.log_param("n_pool_rows",       len(pool) if pool is not None else 0)
        mlflow.log_param("current_auroc",     current_auroc)

        # Train
        log.info("Training XGBoost on combined data...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_holdout, y_holdout)],
            verbose=100,
        )

        # Evaluate on held-out set
        y_prob    = model.predict_proba(X_holdout)[:, 1]
        y_pred_30 = (y_prob >= 0.3).astype(int)

        new_auroc   = roc_auc_score(y_holdout, y_prob)
        new_auprc   = average_precision_score(y_holdout, y_prob)
        new_utility = compute_utility_score(y_holdout, y_pred_30)
        report      = classification_report(
            y_holdout, y_pred_30, output_dict=True
        )

        log.info(f"New model AUROC:    {new_auroc:.4f}")
        log.info(f"Current model AUROC:{current_auroc:.4f}")
        log.info(f"New model recall:   {report['1']['recall']:.4f}")

        # Log metrics
        mlflow.log_metric("auroc",          new_auroc)
        mlflow.log_metric("auprc",          new_auprc)
        mlflow.log_metric("utility_score",  new_utility)
        mlflow.log_metric("recall_t30",     report["1"]["recall"])
        mlflow.log_metric("precision_t30",  report["1"]["precision"])
        mlflow.log_metric("current_auroc",  current_auroc)
        mlflow.log_metric("auroc_delta",    new_auroc - current_auroc)

  
        model.save_model(str(BASELINE / "model_local.json"))
        mlflow.log_artifact(str(BASELINE / "model_local.json"), "model")

        # Register in model registry
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            client = mlflow.MlflowClient()
            mv = client.create_model_version(
                name=MODEL_NAME,
                source=f"{run.info.artifact_uri}/model",
                run_id=run.info.run_id,
            )
            new_version = mv.version
            log.info(f"Model registered as version {new_version}")
        except Exception as e:
            log.warning(f"Registry failed: {e} — model saved locally")
            new_version = None

        # Promotion decision
        # New model must beat current by at least 0.001 AUROC
        # Small threshold prevents noise-driven promotion
        if new_auroc >= current_auroc - 0.001:
            # Promote new model to Production
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(
                MODEL_NAME, stages=["None", "Staging"]
            )
            new_version = max(versions, key=lambda v: int(v.version)).version

            # Archive current Production
            prod_versions = client.get_latest_versions(
                MODEL_NAME, stages=["Production"]
            )
            for v in prod_versions:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=v.version,
                    stage="Archived"
                )
                log.info(f"Archived version {v.version}")

            # Promote new version
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=new_version,
                stage="Production"
            )

            # Save new model locally for API
            model.save_model(str(BASELINE / "model_local.json"))

            mlflow.log_param("promotion_outcome", "promoted")
            log.info(f"Version {new_version} promoted to Production")
            log.info("model_local.json updated — restart API to load new model")

            outcome = "promoted"

        else:
            mlflow.log_param("promotion_outcome", "rejected")
            log.warning(
                f"New model AUROC {new_auroc:.4f} worse than "
                f"current {current_auroc:.4f} — keeping current model"
            )
            outcome = "rejected"

        log.info("=" * 60)
        log.info(f"Retrain complete — outcome: {outcome}")
        log.info(f"Run ID: {run.info.run_id}")
        log.info("=" * 60)

        return outcome, run.info.run_id


if __name__ == "__main__":
    outcome, run_id = retrain()
    print(f"\nOutcome: {outcome}  Run ID: {run_id}")