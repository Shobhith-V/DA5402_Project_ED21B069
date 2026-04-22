"""
DAY 1 VERIFICATION SCRIPT
==========================
Run this first. Before writing a single line of production code,
this script confirms that every critical assumption about the data is true.

If this script runs cleanly and prints all GREEN checks, you're good to proceed.
If anything prints RED, stop and read the message — it means a core assumption failed.

Usage:
    pip install -r requirements_day1.txt
    python day1_verify.py

Expected runtime: 5-10 minutes (download ~42MB + processing)
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}"); 
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def info(msg):  print(f"  {BLUE}→{RESET}  {msg}")
def header(msg):print(f"\n{BOLD}{msg}{RESET}\n" + "─"*60)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
BASELINE  = ROOT / "data" / "baseline"
PLOTS     = ROOT / "data" / "baseline" / "plots"

# Kaggle download locations — where PSV files actually are
KAGGLE_A  = ROOT / "kaggle_raw" / "training_setA" / "training"
KAGGLE_B  = ROOT / "kaggle_raw" / "training_setB" / "training_setB"

for p in [DATA_RAW, DATA_PROC, BASELINE, PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
# Core vital sign columns — dense, reliable across both hospitals
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

# Lab columns — sparse, culturally variable between hospitals
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST",
    "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets"
]

ALL_FEATURE_COLS = VITAL_COLS + LAB_COLS
TARGET_COL       = "SepsisLabel"
PATIENT_ID_COL   = "patient_id"

# Download URLs — PhysioNet 2019 Challenge
URLS = {
    "A": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip",
    "B": "https://physionet.org/files/challenge-2019/1.0.0/training/training_setB.zip",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download
# ══════════════════════════════════════════════════════════════════════════════

def download_data():
    header("STEP 1 — Locating Kaggle data and copying to canonical paths")

    mapping = {
        "A": KAGGLE_A,
        "B": KAGGLE_B,
    }

    for hospital, source_path in mapping.items():
        dest_path = DATA_RAW / f"hospital_{hospital}"

        # If already copied, skip
        if dest_path.exists() and any(dest_path.rglob("*.psv")):
            n = len(list(dest_path.rglob("*.psv")))
            ok(f"Hospital {hospital}: {n} PSV files already in {dest_path.name} — skipping")
            continue

        # Verify source exists
        if not source_path.exists():
            fail(f"Kaggle data not found at {source_path}")
            fail(f"Expected structure:")
            fail(f"  kaggle_raw/training_setA/training/")
            fail(f"  kaggle_raw/training_setB/training_setB/")
            fail(f"Run: kaggle datasets download -d salikhussaini49/prediction-of-sepsis")
            sys.exit(1)

        psv_files = list(source_path.rglob("*.psv"))
        if not psv_files:
            fail(f"No PSV files found in {source_path}")
            sys.exit(1)

        # Copy to canonical location
        import shutil
        dest_path.mkdir(parents=True, exist_ok=True)
        info(f"Copying {len(psv_files)} Hospital {hospital} PSV files...")
        for f in psv_files:
            shutil.copy2(f, dest_path / f.name)
        ok(f"Hospital {hospital}: {len(psv_files)} files copied to {dest_path.name}/")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Parse PSV files into DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def parse_hospital(hospital: str) -> pd.DataFrame:
    """
    Parse all .psv files for one hospital into a single DataFrame.
    Each file = one patient. We add patient_id from the filename.
    """
    path = DATA_RAW / f"hospital_{hospital}"

    # Find psv files — they may be nested one level deep
    psv_files = list(path.rglob("*.psv"))
    if not psv_files:
        fail(f"No PSV files found in {path}")
        sys.exit(1)

    dfs = []
    for f in psv_files:
        try:
            df = pd.read_csv(f, sep="|")
            df[PATIENT_ID_COL] = f.stem          # filename = patient ID
            df["hospital"]     = hospital
            dfs.append(df)
        except Exception as e:
            warn(f"Could not parse {f.name}: {e} — skipping")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def load_and_save_data():
    header("STEP 2 — Parsing PSV files")

    results = {}
    for hospital in ["A", "B"]:
        parquet_path = DATA_PROC / f"hospital_{hospital}.parquet"

        if parquet_path.exists():
            ok(f"Hospital {hospital} parquet already exists — loading")
            results[hospital] = pd.read_parquet(parquet_path)
            continue

        info(f"Parsing Hospital {hospital} PSV files...")
        df = parse_hospital(hospital)
        df.to_parquet(parquet_path, index=False)
        ok(f"Hospital {hospital}: {len(df):,} rows · "
           f"{df[PATIENT_ID_COL].nunique():,} patients · "
           f"saved to {parquet_path.name}")
        results[hospital] = df

    return results["A"], results["B"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Basic sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(hosp_a: pd.DataFrame, hosp_b: pd.DataFrame):
    header("STEP 3 — Sanity checks")

    checks_passed = True

    for name, df in [("Hospital A", hosp_a), ("Hospital B", hosp_b)]:
        n_patients = df[PATIENT_ID_COL].nunique()
        n_rows     = len(df)
        n_sepsis   = df[df[TARGET_COL] == 1][PATIENT_ID_COL].nunique()
        pct_sepsis = n_sepsis / n_patients * 100

        # Check expected size ranges
        if n_patients < 5000:
            fail(f"{name}: only {n_patients} patients — expected ~20,000")
            checks_passed = False
        else:
            ok(f"{name}: {n_patients:,} patients · {n_rows:,} rows")

        # Check sepsis rate (expect 3-10%)
        if pct_sepsis < 1 or pct_sepsis > 15:
            warn(f"{name}: sepsis rate {pct_sepsis:.1f}% — unusual, expected ~5-7%")
        else:
            ok(f"{name}: {n_sepsis:,} sepsis patients ({pct_sepsis:.1f}%)")

        # Check that core vital columns exist
        missing_cols = [c for c in VITAL_COLS if c not in df.columns]
        if missing_cols:
            fail(f"{name}: missing vital columns: {missing_cols}")
            checks_passed = False
        else:
            ok(f"{name}: all vital columns present")

        # Check SepsisLabel column
        if TARGET_COL not in df.columns:
            fail(f"{name}: missing {TARGET_COL} column — cannot build feedback loop")
            checks_passed = False
        else:
            ok(f"{name}: {TARGET_COL} column present — feedback loop possible")

    if not checks_passed:
        fail("Sanity checks failed — do not proceed until these are resolved")
        sys.exit(1)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — THE CRITICAL CHECK: Does drift actually exist?
# ══════════════════════════════════════════════════════════════════════════════

def verify_drift(hosp_a: pd.DataFrame, hosp_b: pd.DataFrame) -> dict:
    """
    This is the most important function in this script.
    
    Run KS test on each vital sign column comparing Hospital A vs B.
    If p < 0.05 on multiple features, drift exists and is real.
    If all p-values are high, the drift story doesn't hold — stop and reassess.
    """
    header("STEP 4 — CRITICAL: Verifying natural drift between hospitals")
    info("Running Kolmogorov-Smirnov test on each feature...")
    info("H0: Hospital A and B come from the same distribution")
    info("p < 0.05 = distributions differ = natural drift confirmed\n")

    drift_results = {}
    drift_confirmed_count = 0

    for col in VITAL_COLS:
        a_vals = hosp_a[col].dropna().values
        b_vals = hosp_b[col].dropna().values

        if len(a_vals) < 100 or len(b_vals) < 100:
            warn(f"{col}: insufficient data for KS test")
            continue

        stat, pvalue = stats.ks_2samp(a_vals, b_vals)

        drift_results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(pvalue), 6),
            "a_mean": round(float(a_vals.mean()), 3),
            "b_mean": round(float(b_vals.mean()), 3),
            "a_std": round(float(a_vals.std()), 3),
            "b_std": round(float(b_vals.std()), 3),
            "drift_detected": bool(pvalue < 0.05),
        }

        symbol = f"{GREEN}DRIFT{RESET}" if pvalue < 0.05 else f"{YELLOW}STABLE{RESET}"
        print(f"  {col:<12} p={pvalue:.4f}  KS={stat:.4f}  "
              f"A_mean={a_vals.mean():.2f}  B_mean={b_vals.mean():.2f}  "
              f"→ {symbol}")

        if pvalue < 0.05:
            drift_confirmed_count += 1

    # Also check missingness rate drift — this is the hidden drift signal
    print()
    info("Checking missingness rate drift (lab draw culture)...")
    for col in LAB_COLS[:8]:          # check first 8 lab columns
        a_miss = hosp_a[col].isna().mean()
        b_miss = hosp_b[col].isna().mean()
        diff   = abs(a_miss - b_miss)
        if diff > 0.05:
            print(f"  {col:<20} A_missing={a_miss:.1%}  B_missing={b_miss:.1%}  "
                  f"diff={diff:.1%}  → {GREEN}MISSINGNESS DRIFT{RESET}")
            drift_results[f"{col}_missingness"] = {
                "a_missing_rate": round(a_miss, 4),
                "b_missing_rate": round(b_miss, 4),
                "difference": round(diff, 4),
                "drift_detected": True,
            }
            drift_confirmed_count += 1

    print()
    if drift_confirmed_count >= 2:
        ok(f"DRIFT CONFIRMED on {drift_confirmed_count} features/signals")
        ok("Natural cross-hospital drift is real — your project hypothesis holds")
        ok("You do NOT need to inject drift artificially")
    elif drift_confirmed_count == 1:
        warn(f"Only 1 feature shows drift — marginal evidence")
        warn("Project can proceed but drift story is weaker than expected")
        warn("Consider emphasising missingness drift in your narrative")
    else:
        fail("NO DRIFT DETECTED — all features show similar distributions")
        fail("This means the cross-hospital drift hypothesis does not hold for this data")
        fail("STOP: reassess the project before proceeding")
        sys.exit(1)

    return drift_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Compute and save baseline statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline(hosp_a: pd.DataFrame, drift_results: dict):
    """
    Compute baseline statistics from Hospital A training data.
    This JSON file is what your KS test will compare against in production.
    It gets DVC-versioned — never overwritten, only appended per model version.
    """
    header("STEP 5 — Computing baseline statistics from Hospital A")

    baseline = {}
    for col in VITAL_COLS:
        vals = hosp_a[col].dropna()
        baseline[col] = {
            "mean":  round(float(vals.mean()), 4),
            "std":   round(float(vals.std()), 4),
            "p25":   round(float(vals.quantile(0.25)), 4),
            "p50":   round(float(vals.quantile(0.50)), 4),
            "p75":   round(float(vals.quantile(0.75)), 4),
            "p95":   round(float(vals.quantile(0.95)), 4),
            "missing_rate": round(float(hosp_a[col].isna().mean()), 4),
            "n_samples": int(vals.count()),
        }
        ok(f"{col}: mean={baseline[col]['mean']}  std={baseline[col]['std']}")

    # Save baseline
    baseline_path = BASELINE / "hospital_a_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    ok(f"Baseline saved to {baseline_path}")

    # Save drift verification results
    drift_path = BASELINE / "drift_verification.json"
    with open(drift_path, "w") as f:
        json.dump(drift_results, f, indent=2)
    ok(f"Drift results saved to {drift_path}")

    return baseline


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Class imbalance check (sets scale_pos_weight for XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

def check_imbalance(hosp_a: pd.DataFrame) -> float:
    header("STEP 6 — Class imbalance analysis")

    n_neg = (hosp_a[TARGET_COL] == 0).sum()
    n_pos = (hosp_a[TARGET_COL] == 1).sum()
    ratio = n_neg / n_pos

    ok(f"Negative rows (no sepsis): {n_neg:,}")
    ok(f"Positive rows (sepsis):    {n_pos:,}")
    ok(f"Imbalance ratio:           {ratio:.1f}:1")
    ok(f"XGBoost scale_pos_weight:  {ratio:.1f}  ← use this exact value")

    info(f"In your training script: XGBClassifier(scale_pos_weight={ratio:.1f})")

    return ratio


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Generate EDA plots (evidence for your documentation)
# ══════════════════════════════════════════════════════════════════════════════

def generate_plots(hosp_a: pd.DataFrame, hosp_b: pd.DataFrame,
                   drift_results: dict):
    header("STEP 7 — Generating EDA plots")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Hospital A vs B — Feature Distribution Comparison\n"
                 "(Natural drift verification)", fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    vital_to_plot = VITAL_COLS[:7]
    for idx, col in enumerate(vital_to_plot):
        row, col_idx = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col_idx])

        a_vals = hosp_a[col].dropna()
        b_vals = hosp_b[col].dropna()

        ax.hist(a_vals, bins=40, alpha=0.6, label='Hospital A', color='steelblue',
                density=True)
        ax.hist(b_vals, bins=40, alpha=0.6, label='Hospital B', color='coral',
                density=True)

        # Add KS p-value if available
        if col in drift_results:
            p = drift_results[col]["p_value"]
            colour = 'red' if p < 0.05 else 'green'
            ax.set_title(f"{col}\np={p:.4f}", fontsize=10,
                        color=colour, fontweight='bold')
        else:
            ax.set_title(col, fontsize=10)

        ax.legend(fontsize=7)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    plot_path = PLOTS / "hospital_ab_drift_verification.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    ok(f"Plot saved to {plot_path}")
    info("Use this plot in your HLD documentation as evidence of natural drift")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Held-out split (lock this on day 1, never change it)
# ══════════════════════════════════════════════════════════════════════════════

def create_held_out_split(hosp_a: pd.DataFrame, hosp_b: pd.DataFrame):
    """
    Create a fixed held-out evaluation set stratified on SepsisLabel.
    This set is locked on day 1. Every model version is evaluated against it.
    Stored as DVC-tracked parquet — never regenerated.
    """
    header("STEP 8 — Creating fixed held-out evaluation set")

    held_out_path = DATA_PROC / "held_out_eval.parquet"
    train_a_path  = DATA_PROC / "train_hospital_a.parquet"

    if held_out_path.exists():
        ok("Held-out set already exists — skipping (good, never regenerate this)")
        return

    from sklearn.model_selection import train_test_split

    # Get unique patients from Hospital A (not rows — patients)
    patients = hosp_a[[PATIENT_ID_COL, TARGET_COL]].groupby(
        PATIENT_ID_COL)[TARGET_COL].max().reset_index()

    train_patients, eval_patients = train_test_split(
        patients,
        test_size=0.20,
        random_state=42,                          # fixed seed — never change
        stratify=patients[TARGET_COL]
    )

    train_df = hosp_a[hosp_a[PATIENT_ID_COL].isin(train_patients[PATIENT_ID_COL])]
    eval_df  = hosp_a[hosp_a[PATIENT_ID_COL].isin(eval_patients[PATIENT_ID_COL])]

    train_df.to_parquet(train_a_path,  index=False)
    eval_df.to_parquet(held_out_path,  index=False)

    n_eval_sepsis = eval_patients[TARGET_COL].sum()
    ok(f"Training set:  {train_df[PATIENT_ID_COL].nunique():,} patients")
    ok(f"Held-out set:  {eval_df[PATIENT_ID_COL].nunique():,} patients  "
       f"({n_eval_sepsis} sepsis cases)")
    ok(f"Saved to {train_a_path.name} and {held_out_path.name}")
    warn("IMPORTANT: held_out_eval.parquet is locked. Never retrain on it.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Print DVC commands to run
# ══════════════════════════════════════════════════════════════════════════════

def print_dvc_setup():
    header("STEP 9 — DVC setup commands")

    info("Run these commands in your terminal after this script completes:\n")
    commands = [
        "dvc init",
        "dvc add data/raw/hospital_A",
        "dvc add data/raw/hospital_B",
        "dvc add data/processed/hospital_a.parquet",
        "dvc add data/processed/hospital_b.parquet",
        "dvc add data/processed/held_out_eval.parquet",
        "dvc add data/baseline/hospital_a_baseline.json",
        "git add .dvc .gitignore",
        'git commit -m "day1: data downloaded, drift verified, baseline locked"',
    ]
    for cmd in commands:
        print(f"    {BLUE}${RESET} {cmd}")

    print()
    info("After running the above, your data layer is fully versioned.")
    info("The baseline JSON is your reference for all future drift checks.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  SEPSIS WATCH — DAY 1 VERIFICATION{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print("  This script verifies every critical assumption before you")
    print("  write a single line of production code.")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Run all steps
    download_data()
    hosp_a, hosp_b = load_and_save_data()
    sanity_check(hosp_a, hosp_b)
    drift_results  = verify_drift(hosp_a, hosp_b)
    baseline       = compute_baseline(hosp_a, drift_results)
    ratio          = check_imbalance(hosp_a)
    generate_plots(hosp_a, hosp_b, drift_results)
    create_held_out_split(hosp_a, hosp_b)
    print_dvc_setup()

    # Save key values for use in subsequent scripts
    config = {
        "scale_pos_weight": round(ratio, 2),
        "vital_cols":       VITAL_COLS,
        "lab_cols":         LAB_COLS,
        "target_col":       TARGET_COL,
        "patient_id_col":   PATIENT_ID_COL,
        "drift_threshold":  0.05,
        "min_ks_window":    500,
        "min_confirmed_pool_for_retrain": 1000,
        "recall_alert_threshold":         0.85,
        "flag_rate_alert_threshold":      0.05,
    }
    config_path = ROOT / "src" / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    header("ALL STEPS COMPLETE")
    ok("Data downloaded and parsed")
    ok("Drift verified between hospitals")
    ok("Baseline statistics locked")
    ok(f"scale_pos_weight = {ratio:.1f} computed")
    ok("Held-out evaluation set created and locked")
    ok("EDA plots saved to data/baseline/plots/")
    ok("Project config saved to src/config.json")
    print(f"\n  {GREEN}{BOLD}Day 1 complete. You can commit and proceed to Day 2.{RESET}\n")


if __name__ == "__main__":
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        fail("scikit-learn not installed — run: pip install -r requirements_day1.txt")
        sys.exit(1)
    main()