# SepsisWatch — ICU Sepsis Early Warning System

This repo consists of all the codes for the DA5402 (ML Operations Lab) project. This project is a production-ready MLOps system for sepsis early detection in ICU patients. Sepsis affects 48 million people annually and kills 11 million — every hour of delayed treatment increases mortality by 7%. SepsisWatch predicts sepsis 6 hours before clinical recognition, monitors its own performance in real time, and automatically retrains when performance degrades due to cross-hospital data drift.

---

## Problem Statement

The model is trained on Hospital A (Beth Israel Deaconess, Boston) and deployed against Hospital B (Emory University, Atlanta). This gives natural cross-hospital distributional drift — confirmed on day 1 via KS test (p≈0 on all 7 vital signs) — without any artificial injection. The system detects this drift, monitors recall degradation, and triggers automated retraining.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                            │
│                                                                  │
│  Frontend:3000  ──REST──▶  FastAPI:8000  ──HTTP──▶  MLflow:5000 │
│                                 │                                │
│                            ┌────▼────┐                           │
│                            │Background│                          │
│                            │  Jobs   │                           │
│                            └─────────┘                           │
│                                 │                                │
│               Prometheus:9090 ◀─┘    Grafana:3001               │
│                                                                  │
│  Airflow:8080            Node Exporter:9100                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Training hospital | Hospital A — 20,336 patients, 790,215 rows |
| Production hospital | Hospital B — 20,000 patients, 761,995 rows |
| Features | 77 (vitals + rolling stats + missingness indicators) |
| Class imbalance | 45.1:1 (row-level) |
| AUROC | 0.7546 |
| Recall @ 0.3 threshold | 0.7318 |
| Drift confirmed | p≈0 on all 7 vital signs (KS test) |
| Inference latency | < 10ms p95 |
| Retrain trigger | pool ≥ 1000 rows AND recall < 0.85 |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| Data Pipeline | Apache Airflow |
| Data Versioning | DVC |
| API | FastAPI |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Frontend | HTML / JS / nginx |

---

## Project Structure

```
DA5402_Project/
├── src/
│   ├── api/
│   │   └── main.py                  # FastAPI service + background jobs
│   ├── pipeline/
│   │   ├── feature_pipeline.py      # 4-task data pipeline
│   │   └── replay.py                # Hospital B stream simulator
│   └── training/
│       ├── train.py                 # XGBoost + MLflow training
│       └── retrain.py               # Sliding window retrain
├── airflow/
│   └── dags/
│       └── sepsis_data_pipeline.py  # Airflow DAG definition
├── frontend/
│   ├── index.html                   # 3-screen clinical dashboard
│   └── Dockerfile
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── grafana/
│       └── provisioning/
├── tests/
│   └── test_core.py                 # 15 unit tests
├── data/                            # DVC tracked
│   ├── raw/                         # Hospital A and B PSV files
│   ├── processed/                   # Parquet feature files
│   ├── baseline/                    # Drift reference + model artifacts
│   └── confirmed/                   # Prediction log + confirmed pool
├── docs/
│   └── report.pdf                   # HLD, LLD, test plan, user manual
├── day1.py                          # Data verification script
├── docker-compose.yml
├── Dockerfile.api
├── MLproject
├── requirements.txt
└── requirements_api.txt
```

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Conda (recommended)
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### 1. Clone and set up environment

```bash
git clone https://github.com/Shobhith-V/DA5402_Project_ED21B069.git
cd DA5402_Project_ED21B069

conda create -n sepsis python=3.10 -y
conda activate sepsis
pip install -r requirements.txt
```

### 2. Download data

```bash
kaggle datasets download -d salikhussaini49/prediction-of-sepsis
unzip prediction-of-sepsis.zip -d kaggle_raw
```

### 3. Run Day 1 verification

```bash
python day1.py
```

This parses all 40,336 PSV files, verifies cross-hospital drift via KS test,
locks the held-out evaluation set (random_state=42), and saves baseline statistics.
All checks must print GREEN before proceeding.

### 4. Run the feature pipeline

```bash
python src/pipeline/feature_pipeline.py
```

Produces 77-feature parquet files for both hospitals including rolling statistics
and lab missingness indicators.

### 5. Start MLflow and train

```bash
# Terminal 1 — start MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination ./mlruns/artifacts \
  --serve-artifacts \
  --allowed-hosts "*"

# Terminal 2 — run training
python src/training/train.py
```

Then promote the model to Production:

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.MlflowClient()
versions = client.search_model_versions(\"name='sepsis-watch-classifier'\")
latest = max(versions, key=lambda v: int(v.version))
client.transition_model_version_stage(
    name='sepsis-watch-classifier',
    version=latest.version,
    stage='Production'
)
print(f'Version {latest.version} promoted to Production')
"
```

### 6. Start the full Docker stack

```bash
docker compose up -d
sleep 30
docker compose ps
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / admin |

### 7. Run the Hospital B replay

```bash
# Simulate live Hospital B patient stream
python src/pipeline/replay.py --fast --n-patients 1000

# Resume from a specific patient index
python src/pipeline/replay.py --fast --start-patient 1000 --n-patients 500
```

---

## MLOps Pipeline

### Data Pipeline (Airflow)

Four-task DAG `sepsis_data_pipeline` triggered manually:

```
validate_raw_data → impute_features → extract_features → compute_baseline_stats
```

Trigger from Airflow UI at http://localhost:8080 or via CLI:

```bash
docker exec sepsis-airflow airflow dags trigger sepsis_data_pipeline
```

### Experiment Tracking (MLflow)

Every training run logs:

- **Parameters:** git commit hash, scale_pos_weight, n_estimators, hospital source, n_train_rows
- **Metrics:** AUROC, AUPRC, utility score, recall@0.3, precision@0.3, F1@0.3, feature importances (top 15)
- **Artifacts:** feature_cols.json, feature_importance.json, trained model

Models are registered in the MLflow registry with `Staging → Production → Archived` stages.

### Automated Retraining Loop

The FastAPI service runs three background jobs via APScheduler:

| Job | Interval | Purpose |
|-----|----------|---------|
| `feedback_loop` | 5 min | Compute rolling recall/precision vs ground truth labels, grow confirmed Hospital B pool |
| `drift_check` | 10 min | KS test on 500-row window vs Hospital A baseline, update Prometheus drift metrics |
| `check_retrain_trigger` | 2 min | Fire retrain when pool ≥ 1000 AND recall < 0.85. 10-minute cooldown between retrains. |

When triggered:
1. Trains new XGBoost on Hospital A + confirmed Hospital B pool
2. Evaluates on locked held-out set
3. Promotes to Production in MLflow registry if AUROC ≥ current − 0.001
4. API loads new model from MLflow registry automatically
5. Prediction log cleared so recall recomputes on new model's predictions

### Monitoring (Prometheus + Grafana)

Custom Prometheus metrics exposed at `http://localhost:8000/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `model_rolling_recall` | Gauge | Rolling recall vs ground truth |
| `model_rolling_precision` | Gauge | Rolling precision vs ground truth |
| `drift_ks_pvalue{feature}` | Gauge | KS p-value per vital sign |
| `confirmed_pool_size` | Gauge | Hospital B confirmed pool size |
| `sepsis_predictions_total{risk_tier}` | Counter | Predictions by risk tier |
| `model_inference_latency_seconds` | Histogram | XGBoost inference time |
| `drift_detected_total` | Counter | Confirmed drift events |
| `retraining_triggered_total{outcome}` | Counter | Retrain events by outcome |

Grafana dashboard at http://localhost:3001 refreshes every 5 seconds.

Alert rules fire when:
- Recall < 0.85
- Drift confirmed on 2+ features (p < 0.05)
- API is down
- Inference latency p95 > 200ms

---

## Frontend

Three-screen clinical dashboard at http://localhost:3000:

- **Ward Monitor** — patient grid sorted by risk score, real-time updates every 10s, search and filter
- **Patient Detail** — vital sign gauges, risk trajectory chart, top contributing features, clinical summary
- **Model Health** — MLOps lifecycle diagram, drift p-values, performance history, system event log

The frontend connects to FastAPI exclusively via REST. No direct model or database access — satisfying the loose coupling requirement.

---

## CI/CD

GitHub Actions pipeline runs on every push to `main`:

```
lint (flake8) → test (pytest, 15 tests) → build Docker image
```

All 15 unit tests cover: feature engineering, drift detection, risk scoring, feedback loop, class imbalance handling.

---

## Data Versioning (DVC)

All data artifacts are DVC-tracked. To reproduce any experiment:

```bash
git checkout <commit-hash>
dvc pull
mlflow run . -e train
```

The git commit hash pins the code, DVC pins the data, and the MLflow run ID pins the hyperparameters and metrics.

---

## Design Decisions

**Why XGBoost over neural networks**
Tabular data with 45:1 class imbalance is handled natively via `scale_pos_weight`. Trains in under 5 minutes on CPU satisfying the no-cloud constraint. Interpretable via feature importance scores used in the clinical UI.

**Why cross-site validation instead of train/val/test split**
Hospital B serves as the validation set. A model that generalises to a different hospital with different demographics and clinical protocols has demonstrated real-world capability, not just within-distribution pattern matching.

**Why carry-forward imputation**
The last recorded vital sign is the most clinically relevant estimate of the current value. Imputation means are computed from Hospital A only to prevent data leakage from Hospital B into the reference distribution.

**Why KS test for drift detection**
Non-parametric — makes no assumptions about distribution shape. Per-feature testing gives actionable named alerts. Two features must drift simultaneously to confirm — this addresses the multiple comparisons problem (30% false positive rate with a single-feature threshold at α=0.05).

**Why Airflow over Spark**
The 42MB PhysioNet dataset does not warrant distributed computing. Airflow provides orchestration, retry logic, task dependency management, and a visual DAG without JVM overhead.

**Why separate Docker services for frontend and backend**
Frontend and backend are independent software blocks connected only via REST API. Either can be updated, scaled, or redeployed independently.

---

## Reproducing Results

```bash
# Clone and set up
git clone https://github.com/Shobhith-V/DA5402_Project_ED21B069.git
cd DA5402_Project_ED21B069
conda create -n sepsis python=3.10 -y && conda activate sepsis
pip install -r requirements.txt

# Download data
kaggle datasets download -d salikhussaini49/prediction-of-sepsis
unzip prediction-of-sepsis.zip -d kaggle_raw

# Verify data and drift
python day1.py

# Run feature pipeline
python src/pipeline/feature_pipeline.py

# Train and register model
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination ./mlruns/artifacts \
  --serve-artifacts --allowed-hosts "*" &
python src/training/train.py

# Start full stack
docker compose up -d

# Run Hospital B stream
python src/pipeline/replay.py --fast --n-patients 1000
```

---

## Author

**Shobhith Vadlamudi** — ED21B069  
DA5402 Machine Learning Operations  
Indian Institute of Technology Madras