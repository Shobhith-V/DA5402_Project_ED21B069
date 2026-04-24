"""
Training Script — Sepsis Watch
================================
Trains XGBoost on Hospital A features.
Logs everything to MLflow — params, metrics, git hash, artifacts.
Registers model in MLflow registry.

Usage:
    # Start MLflow server first (separate terminal):
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

    # Then run training:
    python src/training/train.py
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
import shap
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

ROOT     = Path(__file__).parent.parent.parent
DATA     = ROOT / "data" / "processed"
BASELINE = ROOT / "data" / "baseline"
CONFIG   = ROOT / "src" / "config.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "sepsis-watch"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"

def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def get_git_hash():
    """
    Captures the current git commit hash.
    This links the MLflow run to the exact code version that
    produced it — satisfying the reproducibility requirement:
    'every experiment reproducible via git commit hash + MLflow run ID'
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


def load_training_data(cfg):
    """
    Loads feature matrix from hospital_A_features.parquet
    filtered to training patients only (from train_hospital_a.parquet).

    Why filter by patient ID rather than loading train parquet directly:
    train_hospital_a.parquet was created before feature engineering —
    it has the original 42 columns not the 82 engineered features.
    We need the 82-feature version but only for training patients.
    """
    log.info("Loading training data...")

    train_patients = pd.read_parquet(
        DATA / "train_hospital_a.parquet",
        columns=[cfg["patient_id_col"]]
    )[cfg["patient_id_col"]].unique()

    log.info(f"Training patients: {len(train_patients):,}")

    features = pd.read_parquet(DATA / "hospital_A_features.parquet")
    train_df = features[
        features[cfg["patient_id_col"]].isin(train_patients)
    ].copy()

    log.info(f"Training rows: {len(train_df):,}")

    meta_cols    = cfg["meta_cols"] + ["ICULOS", "HospAdmTime"]
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    X = train_df[feature_cols].fillna(0)
    y = train_df[cfg["target_col"]]

    log.info(f"Features: {len(feature_cols)}")
    log.info(f"Positive rate: {y.mean():.4f}")

    return X, y, feature_cols


def load_holdout_data(cfg, feature_cols):
    """
    Loads held-out evaluation set.
    Locked on day 1 — never trained on.
    Every model version evaluated here for fair comparison.
    """
    log.info("Loading held-out evaluation set...")

    holdout_patients = pd.read_parquet(
        DATA / "held_out_eval.parquet",
        columns=[cfg["patient_id_col"]]
    )[cfg["patient_id_col"]].unique()

    features   = pd.read_parquet(DATA / "hospital_A_features.parquet")
    holdout_df = features[
        features[cfg["patient_id_col"]].isin(holdout_patients)
    ].copy()

    X_holdout = holdout_df[feature_cols].fillna(0)
    y_holdout = holdout_df[cfg["target_col"]]

    log.info(f"Held-out rows: {len(holdout_df):,}")
    log.info(f"Held-out positive rate: {y_holdout.mean():.4f}")

    return X_holdout, y_holdout


def compute_utility_score(y_true, y_pred):
    """
    Simplified PhysioNet 2019 utility score.
    Rewards early sepsis predictions.
    Penalises FN heavily (2x) — missing sepsis is dangerous.
    Penalises FP lightly (0.05x) — false alarms are manageable.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    utility = (tp - 0.05 * fp - 2.0 * fn) / max(tp + fn, 1)
    return float(utility)


def train():
    cfg      = load_config()
    git_hash = get_git_hash()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    log.info(f"MLflow: {MLFLOW_TRACKING_URI}")
    log.info(f"Git commit: {git_hash}")

    X_train, y_train, feature_cols = load_training_data(cfg)
    X_holdout, y_holdout           = load_holdout_data(cfg, feature_cols)

    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": cfg["scale_pos_weight"],
        "eval_metric":      "aucpr",
        "random_state":     42,
        "n_jobs":           -1,
    }

    with mlflow.start_run(run_name=f"train-{git_hash}") as run:

        # ── Log parameters ────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_param("git_commit",     git_hash)
        mlflow.log_param("hospital_train", "A")
        mlflow.log_param("n_train_rows",   len(X_train))
        mlflow.log_param("n_features",     len(feature_cols))
        mlflow.log_param("imputation",     "carry_forward_then_hospital_A_mean")
        mlflow.log_param("pos_rate_train", round(float(y_train.mean()), 4))

        # ── Train ─────────────────────────────────────────────────
        log.info("Training XGBoost...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_holdout, y_holdout)],
            verbose=100,
        )

        # ── Evaluate ──────────────────────────────────────────────
        # Threshold 0.3 not 0.5 — lower threshold increases recall
        # at cost of precision. Missing sepsis is more dangerous
        # than a false alarm, so we bias toward recall.
        y_prob    = model.predict_proba(X_holdout)[:, 1]
        y_pred_30 = (y_prob >= 0.3).astype(int)

        auroc     = roc_auc_score(y_holdout, y_prob)
        auprc     = average_precision_score(y_holdout, y_prob)
        utility   = compute_utility_score(y_holdout, y_pred_30)
        report_30 = classification_report(
            y_holdout, y_pred_30, output_dict=True
        )

        log.info(f"AUROC:          {auroc:.4f}")
        log.info(f"AUPRC:          {auprc:.4f}")
        log.info(f"Utility score:  {utility:.4f}")
        log.info(f"Recall@0.3:     {report_30['1']['recall']:.4f}")
        log.info(f"Precision@0.3:  {report_30['1']['precision']:.4f}")

        # ── Log metrics ───────────────────────────────────────────
        mlflow.log_metric("auroc",          auroc)
        mlflow.log_metric("auprc",          auprc)
        mlflow.log_metric("utility_score",  utility)
        mlflow.log_metric("recall_t30",     report_30["1"]["recall"])
        mlflow.log_metric("precision_t30",  report_30["1"]["precision"])
        mlflow.log_metric("f1_t30",         report_30["1"]["f1-score"])
        mlflow.log_metric("pos_rate_train", float(y_train.mean()))

        # ── Feature importance ────────────────────────────────────
        # Try SHAP first — game-theoretic, more accurate.
        # Fall back to XGBoost built-in if SHAP version incompatible.
        log.info("Computing feature importance...")
        try:
            sample_idx = np.random.choice(
                len(X_holdout),
                size=min(1000, len(X_holdout)),
                replace=False
            )
            X_sample  = X_holdout.iloc[sample_idx]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            importance = pd.Series(
                np.abs(shap_vals).mean(axis=0),
                index=feature_cols
            ).sort_values(ascending=False)
            log.info("Using SHAP importance")

        except Exception as e:
            log.warning(f"SHAP failed: {e} — using XGBoost built-in importance")
            importance = pd.Series(
                model.feature_importances_,
                index=feature_cols
            ).sort_values(ascending=False)

        # Log top 15 and save full list — both run regardless of path above
        for feat, val in importance.head(15).items():
            mlflow.log_metric(f"importance_{feat}", round(float(val), 6))

        importance_path = BASELINE / "feature_importance.json"
        importance.to_json(importance_path, indent=2)
        mlflow.log_artifact(str(importance_path), "feature_importance")

        log.info("Top 10 features:")
        for feat, val in importance.head(10).items():
            log.info(f"  {feat:<35} {val:.4f}")

        # ── Log model ─────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="sepsis-watch-classifier",
        )

        # ── Save feature column order ─────────────────────────────
        # FastAPI uses this to align incoming requests to the exact
        # feature order the model was trained on. Order matters for
        # XGBoost — wrong order = wrong predictions silently.
        feat_path = BASELINE / "feature_cols.json"
        with open(feat_path, "w") as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(str(feat_path), "feature_cols")

        log.info(f"\nRun ID:  {run.info.run_id}")
        log.info(f"Commit:  {git_hash}")
        log.info(f"\nTo promote to Production:")
        log.info(
            f"  mlflow models transition-model-version "
            f"--model-name sepsis-watch-classifier "
            f"--version 1 --stage Production"
        )

        return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    print(f"\nDone. Run ID: {run_id}")