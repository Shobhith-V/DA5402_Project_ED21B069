"""
Feature Pipeline — Sepsis Watch
================================
Run standalone: python src/pipeline/feature_pipeline.py
Then wrapped in Airflow DAG for orchestration.

Tasks:
  1. validate_raw_data      — schema check before processing
  2. impute_features        — carry-forward + mean fill
  3. extract_features       — rolling stats + missingness indicators
  4. compute_baseline_stats — full baseline for drift detection
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent.parent
DATA_PROC = ROOT / "data" / "processed"
BASELINE  = ROOT / "data" / "baseline"
CONFIG    = ROOT / "src" / "config.json"


def load_config():
    with open(CONFIG) as f:
        return json.load(f)


def validate_raw_data():
    log.info("TASK 1: Validating raw data")
    cfg = load_config()

    required = (
        cfg["vital_cols"] + cfg["lab_cols"] +
        cfg["demo_cols"] + [cfg["target_col"], cfg["patient_id_col"]]
    )

    for hospital in ["A", "B"]:
        path = DATA_PROC / f"hospital_{hospital}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"hospital_{hospital}.parquet not found. Run day1.py first."
            )
        df = pd.read_parquet(path)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Hospital {hospital} missing: {missing}")
        log.info(f"Hospital {hospital}: {len(df):,} rows — schema OK")

    log.info("TASK 1 complete")
    return True


def impute_features():
    """
    Two-pass imputation:
    Pass 1 — carry-forward per patient (ffill grouped by patient_id)
              Clinical rationale: last recorded value is best estimate
              of current value. Must sort by ICULOS first so values
              flow forward in time not backward.
    Pass 2 — fill remaining NaN with Hospital A column means.
              Handles first-hour patients with no prior value.
              Means computed from Hospital A only — applying Hospital A
              means to Hospital B prevents data leakage from B into
              the reference distribution.
    EtCO2 dropped — 100% missing in Hospital A, zero information.
    """
    log.info("TASK 2: Imputing features")
    cfg = load_config()

    pid         = cfg["patient_id_col"]
    drop        = cfg["drop_cols"]
    impute_cols = cfg["vital_cols"] + cfg["lab_cols"] + cfg["demo_cols"]

    # Compute means from Hospital A only
    log.info("Computing imputation means from Hospital A...")
    hosp_a = pd.read_parquet(DATA_PROC / "hospital_A.parquet")
    hosp_a = hosp_a.drop(columns=drop, errors="ignore")

    col_means = {}
    for col in impute_cols:
        if col in hosp_a.columns:
            val = hosp_a[col].mean()
            col_means[col] = float(val) if not np.isnan(val) else 0.0

    means_path = BASELINE / "imputation_means.json"
    with open(means_path, "w") as f:
        json.dump(col_means, f, indent=2)
    log.info(f"Imputation means saved: {len(col_means)} columns")

    for hospital in ["A", "B"]:
        out = DATA_PROC / f"hospital_{hospital}_imputed.parquet"
        if out.exists():
            log.info(f"Hospital {hospital} already imputed — skipping")
            continue

        log.info(f"Imputing Hospital {hospital}...")
        df = pd.read_parquet(DATA_PROC / f"hospital_{hospital}.parquet")
        df = df.drop(columns=drop, errors="ignore")
        df = df.sort_values([pid, "ICULOS"]).reset_index(drop=True)

        # Pass 1: carry-forward within each patient
        for col in impute_cols:
            if col in df.columns:
                df[col] = df.groupby(pid)[col].ffill()

        # Pass 2: mean fill for still-missing values
        for col in impute_cols:
            if col in df.columns and col in col_means:
                df[col] = df[col].fillna(col_means[col])

        # Verify vital signs are fully imputed
        vital_nan = df[cfg["vital_cols"]].isna().sum().sum()
        if vital_nan > 0:
            log.warning(f"Hospital {hospital}: {vital_nan} NaN remain in vitals")
        else:
            log.info(f"Hospital {hospital}: vital signs fully imputed")

        df.to_parquet(out, index=False)
        log.info(f"Hospital {hospital} imputed: {len(df):,} rows → {out.name}")

    log.info("TASK 2 complete")


# ══════════════════════════════════════════════════════════════════
# TASK 3 — Extract features
# ══════════════════════════════════════════════════════════════════

def extract_features():
    """
    Engineers three categories of features beyond raw values:

    1. Rolling mean (6h window) per vital per patient
       Rationale: temporal trend more informative than instantaneous
       value. HR=110 for one hour may be normal. HR averaging 110
       over 6 hours is clinically concerning. 6h matches SOFA score
       assessment frequency used in Sepsis-3 definition.

    2. Rolling std (6h window) per vital per patient
       Rationale: variability is itself a clinical signal. High
       variability in blood pressure indicates haemodynamic instability
       even if the mean appears normal.

    3. Missingness indicators per lab column (binary: was lab drawn?)
       Rationale: Hospital B draws blood gas labs at dramatically lower
       rates than Hospital A (e.g. BaseExcess: A=10.4% drawn vs
       B=0.2% drawn). This cultural difference is a drift signal.
       Also clinically meaningful — a lab being drawn often indicates
       the clinician suspected something. The act of ordering the test
       is itself information.
    """
    log.info("TASK 3: Extracting features")
    cfg = load_config()

    pid      = cfg["patient_id_col"]
    vitals   = cfg["vital_cols"]
    labs     = cfg["lab_cols"]
    target   = cfg["target_col"]

    for hospital in ["A", "B"]:
        out = DATA_PROC / f"hospital_{hospital}_features.parquet"
        if out.exists():
            log.info(f"Hospital {hospital} features exist — skipping")
            continue

        log.info(f"Extracting features for Hospital {hospital}...")
        df = pd.read_parquet(
            DATA_PROC / f"hospital_{hospital}_imputed.parquet"
        )

        # Rolling stats — window=6 hours, min_periods=1 so first rows
        # get a valid value (mean of whatever is available so far)
        # rather than NaN. grouped by patient so stats never bleed
        # across patient boundaries.
        for col in vitals:
            if col in df.columns:
                grouped = df.groupby(pid)[col]
                df[f"{col}_roll_mean"] = grouped.transform(
                    lambda x: x.rolling(window=6, min_periods=1).mean()
                )
                df[f"{col}_roll_std"] = grouped.transform(
                    lambda x: x.rolling(window=6, min_periods=1).std()
                ).fillna(0)
                # fillna(0) on std: first row has no variance yet,
                # 0 is correct (single value has zero spread)

        # Missingness indicators — computed from ORIGINAL data
        # (before imputation) so the indicator reflects whether the
        # lab was actually drawn, not whether we filled it in.
        # We reload the original for this reason.
        original = pd.read_parquet(
            DATA_PROC / f"hospital_{hospital}.parquet"
        ).drop(columns=cfg["drop_cols"], errors="ignore")
        original = original.sort_values(
            [pid, "ICULOS"]
        ).reset_index(drop=True)

        for col in labs:
            if col in original.columns:
                # 1 = lab was drawn (value existed in raw data)
                # 0 = lab was not drawn (value was NaN in raw data)
                df[f"{col}_drawn"] = original[col].notna().astype(int)

        # Report feature count
        meta      = cfg["meta_cols"]
        feat_cols = [c for c in df.columns if c not in meta]
        log.info(
            f"Hospital {hospital}: {len(feat_cols)} features · "
            f"{len(df):,} rows → {out.name}"
        )

        df.to_parquet(out, index=False)

    log.info("TASK 3 complete")


# ══════════════════════════════════════════════════════════════════
# TASK 4 — Compute baseline statistics
# ══════════════════════════════════════════════════════════════════

def compute_baseline_stats():
    """
    Computes baseline feature statistics from Hospital A features.
    This is the reference distribution for all KS drift tests.

    Overwrites the day1.py baseline which only covered 7 raw vitals.
    This baseline covers all ~48 engineered features including
    rolling stats and missingness indicators.

    Write-once after initial pipeline run. Updated only when the
    training window expands to include confirmed Hospital B data
    (after first retrain). That update uses the combined distribution
    as the new reference.
    """
    log.info("TASK 4: Computing baseline statistics")

    path = DATA_PROC / "hospital_A_features.parquet"
    if not path.exists():
        raise FileNotFoundError("Run extract_features() first")

    cfg    = load_config()
    meta   = cfg["meta_cols"]

    df     = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c not in meta]

    baseline = {}
    for col in feat_cols:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        baseline[col] = {
            "mean":         round(float(vals.mean()),     6),
            "std":          round(float(vals.std()),      6),
            "p5":           round(float(vals.quantile(0.05)), 6),
            "p25":          round(float(vals.quantile(0.25)), 6),
            "p50":          round(float(vals.quantile(0.50)), 6),
            "p75":          round(float(vals.quantile(0.75)), 6),
            "p95":          round(float(vals.quantile(0.95)), 6),
            "missing_rate": round(float(df[col].isna().mean()), 6),
            "n_samples":    int(vals.count()),
        }

    out = BASELINE / "hospital_a_baseline.json"
    with open(out, "w") as f:
        json.dump(baseline, f, indent=2)

    log.info(f"Baseline: {len(baseline)} features saved to {out.name}")
    log.info("TASK 4 complete")


# ══════════════════════════════════════════════════════════════════
# MAIN — run all tasks in order
# ══════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("FEATURE PIPELINE — starting")
    log.info("=" * 60)

    validate_raw_data()
    impute_features()
    extract_features()
    compute_baseline_stats()

    log.info("=" * 60)
    log.info("FEATURE PIPELINE — complete")
    log.info("Outputs:")
    for f in sorted(DATA_PROC.glob("*.parquet")):
        size_mb = f.stat().st_size / 1_048_576
        log.info(f"  {f.name:<45} {size_mb:.1f} MB")
    log.info("=" * 60)


if __name__ == "__main__":
    main()