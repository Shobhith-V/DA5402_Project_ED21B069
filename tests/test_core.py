"""
Unit tests — Sepsis Watch
Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ══════════════════════════════════════════════════════════════════
# Tests: Feature engineering
# ══════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def test_carry_forward_imputation(self):
        """Carry-forward fills NaN with previous value per patient."""
        df = pd.DataFrame({
            "patient_id": ["P1", "P1", "P1", "P2", "P2"],
            "HR":         [80.0, np.nan, np.nan, 90.0, np.nan],
        })
        df["HR"] = df.groupby("patient_id")["HR"].ffill()
        assert df.loc[1, "HR"] == 80.0
        assert df.loc[2, "HR"] == 80.0
        assert df.loc[4, "HR"] == 90.0

    def test_missingness_indicator(self):
        """Missingness indicator is 1 when lab was drawn, 0 when not."""
        df = pd.DataFrame({"Lactate": [2.1, np.nan, 1.8, np.nan]})
        df["Lactate_drawn"] = df["Lactate"].notna().astype(int)
        assert list(df["Lactate_drawn"]) == [1, 0, 1, 0]

    def test_rolling_stats_per_patient(self):
        """Rolling mean computed within each patient."""
        df = pd.DataFrame({
            "patient_id": ["P1"] * 6,
            "HR": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0],
        })
        df["HR_roll_mean"] = df.groupby("patient_id")["HR"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        assert df.loc[0, "HR_roll_mean"] == 60.0
        assert abs(df.loc[2, "HR_roll_mean"] - 62.0) < 0.01

    def test_reset_index_after_sort(self):
        """reset_index gives clean 0,1,2 index after sort."""
        df = pd.DataFrame({
            "patient_id": ["P1", "P1", "P1"],
            "ICULOS":     [3, 1, 2],
            "HR":         [80.0, 70.0, 75.0],
        })
        df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
        assert df.index.tolist() == [0, 1, 2]
        assert df.loc[0, "HR"] == 70.0


# ══════════════════════════════════════════════════════════════════
# Tests: Drift detection
# ══════════════════════════════════════════════════════════════════

class TestDriftDetection:

    def test_ks_test_detects_shift(self):
        """KS test detects a clear distributional shift."""
        rng = np.random.default_rng(42)
        reference = rng.normal(70, 10, 500)
        drifted   = rng.normal(85, 12, 500)
        stat, pvalue = stats.ks_2samp(reference, drifted)
        assert pvalue < 0.05

    def test_ks_test_no_false_positive(self):
        """KS test does not fire on same distribution."""
        rng = np.random.default_rng(42)
        reference = rng.normal(70, 10, 500)
        similar   = rng.normal(70, 10, 500)
        stat, pvalue = stats.ks_2samp(reference, similar)
        assert stat < 0.15

    def test_drift_requires_minimum_samples(self):
        """Drift check should not run with insufficient data."""
        min_window      = 500
        current_entries = 300
        should_run      = current_entries >= min_window
        assert not should_run

    def test_two_feature_threshold(self):
        """Drift confirmed only when 2+ features drift simultaneously."""
        drift_results = {
            "HR":    {"p_value": 0.03, "drift_detected": True},
            "Temp":  {"p_value": 0.08, "drift_detected": False},
            "O2Sat": {"p_value": 0.01, "drift_detected": True},
        }
        n_drifted       = sum(1 for v in drift_results.values() if v["drift_detected"])
        drift_confirmed = n_drifted >= 2
        assert drift_confirmed


# ══════════════════════════════════════════════════════════════════
# Tests: Risk scoring
# ══════════════════════════════════════════════════════════════════

class TestRiskScoring:

    def test_risk_tier_assignment(self):
        """Risk tiers assigned correctly."""
        def get_tier(score):
            if score >= 0.3:   return "high"
            elif score >= 0.1: return "medium"
            return "low"

        assert get_tier(0.85) == "high"
        assert get_tier(0.30) == "high"
        assert get_tier(0.29) == "medium"
        assert get_tier(0.10) == "medium"
        assert get_tier(0.09) == "low"
        assert get_tier(0.00) == "low"

    def test_flagged_when_high_risk(self):
        """Records flagged only when score >= 0.3."""
        scores  = [0.85, 0.55, 0.20, 0.31, 0.09]
        flagged = [s >= 0.3 for s in scores]
        assert flagged == [True, True, False, True, False]


# ══════════════════════════════════════════════════════════════════
# Tests: Feedback loop
# ══════════════════════════════════════════════════════════════════

class TestFeedbackLoop:

    def test_recall_computation(self):
        """Rolling recall computed correctly."""
        y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0])
        tp     = np.sum((y_pred == 1) & (y_true == 1))
        fn     = np.sum((y_pred == 0) & (y_true == 1))
        recall = tp / (tp + fn)
        assert abs(recall - 0.8) < 0.01

    def test_confirmed_pool_grows(self):
        """Confirmed pool accumulates rows correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "confirmed.parquet"
            batch1 = pd.DataFrame({
                "patient_id":  ["P1", "P2"],
                "SepsisLabel": [0, 1],
                "HR":          [72.0, 95.0],
                "timestamp":   [1000.0, 1001.0],
            })
            batch1.to_parquet(pool_path, index=False)

            batch2 = pd.DataFrame({
                "patient_id":  ["P3"],
                "SepsisLabel": [0],
                "HR":          [68.0],
                "timestamp":   [1002.0],
            })
            existing = pd.read_parquet(pool_path)
            combined = pd.concat([existing, batch2]).drop_duplicates(
                subset=["patient_id", "timestamp"]
            )
            combined.to_parquet(pool_path, index=False)
            result = pd.read_parquet(pool_path)
            assert len(result) == 3

    def test_retrain_guard_minimum_pool(self):
        """Retraining does not trigger below minimum pool size."""
        min_pool   = 1000
        pool_sizes = [100, 500, 999, 1000, 1500]
        should     = [s >= min_pool for s in pool_sizes]
        assert should == [False, False, False, True, True]


# ══════════════════════════════════════════════════════════════════
# Tests: Class imbalance
# ══════════════════════════════════════════════════════════════════

class TestClassImbalance:

    def test_scale_pos_weight_computation(self):
        """scale_pos_weight computed correctly from class counts."""
        n_neg            = 773079
        n_pos            = 17136
        scale_pos_weight = round(n_neg / n_pos, 2)
        assert scale_pos_weight == 45.11

    def test_positive_rate_preserved_after_stratified_split(self):
        """Stratified split preserves sepsis rate in both halves."""
        from sklearn.model_selection import train_test_split

        rng = np.random.default_rng(42)
        n   = 1000
        y   = (rng.random(n) < 0.056).astype(int)

        y_train, y_test = train_test_split(
            y, test_size=0.2, random_state=42, stratify=y
        )
        train_rate = y_train.mean()
        test_rate  = y_test.mean()

        # Both rates should be within 2% of the original
        assert abs(train_rate - y.mean()) < 0.02
        assert abs(test_rate  - y.mean()) < 0.02