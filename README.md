# YouTube Comment Sentiment Analysis — MLOps Pipeline

An end-to-end MLOps project that classifies YouTube comments as **Positive**, **Neutral**, or **Negative**. The project takes a raw comment dataset all the way to a versioned, tracked, and deployable model using **DVC** for pipeline/data versioning, **MLflow** for experiment tracking and model registry, **Docker** for containerization, and **GitHub Actions** for CI/CD to **AWS ECR/EC2**.

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [ML Pipeline (DVC Stages)](#ml-pipeline-dvc-stages)
- [Model](#model)
- [Experimentation Notebooks](#experimentation-notebooks)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Running the DVC Pipeline](#running-the-dvc-pipeline)
- [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [License](#license)
- [Author](#author)

---

## Overview

The goal of this project is to build a robust sentiment classifier for YouTube comments and wrap it in production-grade MLOps tooling. Instead of a single training script, the project is organized as a **reproducible DVC pipeline** with four stages — ingestion, preprocessing, model building, and evaluation — followed by automatic **model registration** in an MLflow Model Registry. The final model is a **Stacking Classifier** (LightGBM + Logistic Regression, with a KNN meta-learner) trained on TF-IDF features, reaching roughly **86.6% test accuracy** in the tracked experiments.

## Project Architecture

```
                ┌──────────────────┐
                │ dataset/sentiments.csv │
                └─────────┬────────┘
                          │
                 ┌────────▼─────────┐
                 │  Data Ingestion   │  → train/test split (data/raw)
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Data Preprocessing│  → clean/lemmatize (data/interim)
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │  Model Building   │  → TF-IDF + Stacking Classifier
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Model Evaluation  │  → metrics + confusion matrix → MLflow
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Model Registration│  → MLflow Model Registry (Staging)
                 └───────────────────┘
```

All stages are orchestrated by **DVC** (`dvc.yaml` / `dvc.lock`) so the entire pipeline can be reproduced with a single command, and every run is versioned end-to-end (code, data, parameters, and model artifacts).

## Repository Structure

```
youtube_sentiment_analysis-MLOPS/
├── .dvc/                       # DVC internal config
├── .github/workflows/
│   └── cicd.yaml                # CI/CD: build, test, push to ECR, deploy on EC2
├── notebooks/
│   ├── 1_Preprocessing_&_EDA.ipynb
│   ├── 2_experiment_1_baseline_model.ipynb
│   ├── 3_experiment_2_bow_tfidf.ipynb
│   ├── 4_experiment_3_tfidf_(1,3)_max_features.ipynb
│   ├── 5_experiment_4_handling_imbalanced_data.ipynb
│   ├── 6_experiment_5_xgboost_with_hpt.ipynb
│   ├── 7_experiment_6_lightgbm_detailed_hpt.ipynb
│   ├── 8_stacking.ipynb          # Final Stacking Classifier experiment
│   └── artifacts/                # Saved plots/artifacts from experiments
├── src/
│   ├── data/
│   │   ├── data_ingestion.py     # Load raw CSV, split into train/test
│   │   └── data_preprocessing.py # Clean, remove stopwords, lemmatize
│   └── model/
│       ├── model_building.py     # TF-IDF + Stacking Classifier training
│       ├── model_evaluation.py   # Evaluate on test set, log to MLflow
│       └── register_model.py     # Register best model in MLflow registry
├── Dockerfile
├── dvc.yaml                      # DVC pipeline stage definitions
├── dvc.lock                      # DVC pipeline lock file (hashes/versions)
├── params.yaml                   # Central hyperparameters/config
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

> **Note:** `data/`, `dataset/`, and generated artifacts (`*.pkl`, `experiment_info.json`) are excluded from Git via `.gitignore` and are managed/reproduced through DVC instead.

## Dataset

- Expected at `dataset/sentiments.csv` (not committed to Git — supply your own copy or pull it via DVC if a remote is configured).
- Required columns:
  - `clean_comment` — the raw YouTube comment text.
  - `category` — the sentiment label, encoded as:
    - `1` → Positive
    - `0` → Neutral
    - `-1` → Negative
- EDA (see `notebooks/1_Preprocessing_&_EDA.ipynb`) explores word-count distributions, stop-word frequency, and common bigrams/trigrams across the three sentiment classes.

## ML Pipeline (DVC Stages)

The pipeline is defined in `dvc.yaml` and executed with `dvc repro`.

| Stage | Script | Description | Key Outputs |
|---|---|---|---|
| `data_ingestion` | `src/data/data_ingestion.py` | Loads `dataset/sentiments.csv`, drops nulls/duplicates/empty comments, splits into train/test using `data_ingestion.test_size` | `data/raw/train.csv`, `data/raw/test.csv` |
| `data_preprocessing` | `src/data/data_preprocessing.py` | Lowercases text, strips whitespace/newlines, removes non-alphanumeric characters (keeps basic punctuation), removes stopwords (while preserving negation words like *not*, *but*, *no*, *however*, *yet*), and lemmatizes with NLTK's `WordNetLemmatizer` | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| `model_building` | `src/model/model_building.py` | Fits a `TfidfVectorizer` and trains a `StackingClassifier` | `stacking_model.pkl`, `tfidf_vectorizer.pkl` |
| `model_evaluation` | `src/model/model_evaluation.py` | Loads the trained model + vectorizer, evaluates on the test split, logs params/metrics/confusion matrix to MLflow | `experiment_info.json` |
| `model_registration` | `src/model/register_model.py` | Registers the evaluated model in the MLflow Model Registry and promotes it to the **Staging** stage | Registered model version |

## Model

The final model (see `src/model/model_building.py` and `notebooks/8_stacking.ipynb`) is a **`StackingClassifier`** built on top of TF-IDF features:

- **Feature extraction:** `TfidfVectorizer` with `max_features=10000` and `ngram_range=(1, 3)` (unigrams through trigrams).
- **Base learners:**
  - `LightGBM` (`LGBMClassifier`) — multiclass objective, balanced class weights, tuned `learning_rate`, `max_depth`, `n_estimators` (values sourced from hyperparameter tuning experiments and stored in `params.yaml`).
  - `LogisticRegression` — balanced class weights, `lbfgs` solver.
- **Meta-learner:** `KNeighborsClassifier` (`n_neighbors=5`) with 5-fold cross-validation (`cv=5`).
- **Reported performance:** ~**86.6% test accuracy** in the tracked stacking experiment, with per-class precision/recall/F1 logged to MLflow along with a confusion matrix artifact.

The path to this architecture was iterative — the `notebooks/` folder documents the progression from a Bag-of-Words baseline, through TF-IDF and n-gram tuning, class-imbalance handling (e.g. SMOTE), and hyperparameter-tuned XGBoost/LightGBM models, before converging on the final stacked ensemble.

## Experimentation Notebooks

| Notebook | Purpose |
|---|---|
| `1_Preprocessing_&_EDA.ipynb` | Exploratory data analysis: class distribution, word/character counts, stopwords, n-grams |
| `2_experiment_1_baseline_model.ipynb` | Baseline model for comparison |
| `3_experiment_2_bow_tfidf.ipynb` | Bag-of-Words vs. TF-IDF comparison |
| `4_experiment_3_tfidf_(1,3)_max_features.ipynb` | TF-IDF n-gram range and `max_features` tuning |
| `5_experiment_4_handling_imbalanced_data.ipynb` | Techniques for class imbalance |
| `6_experiment_5_xgboost_with_hpt.ipynb` | XGBoost with hyperparameter tuning |
| `7_experiment_6_lightgbm_detailed_hpt.ipynb` | Detailed LightGBM hyperparameter tuning |
| `8_stacking.ipynb` | Final Stacking Classifier (LightGBM + Logistic Regression + KNN meta-learner) |

Each experiment is logged to MLflow for side-by-side comparison of parameters and metrics.

## Tech Stack

- **Language:** Python 3.11
- **ML/NLP:** scikit-learn, LightGBM, NLTK, VADER Sentiment, imbalanced-learn
- **Experiment Tracking / Registry:** MLflow (remote tracking server on AWS EC2)
- **Data & Pipeline Versioning:** DVC (with S3-compatible remote support via `dvc[s3]`)
- **Serving:** Flask, Flask-CORS
- **Data Source:** Google API Client (`google-api-python-client`) for pulling YouTube data
- **Containerization:** Docker
- **CI/CD:** GitHub Actions → Amazon ECR → self-hosted EC2 runner
- **Testing:** pytest

## Getting Started

### Prerequisites

- Python 3.8+ (project developed on 3.11)
- pip
- Docker (optional, for containerized runs)
- AWS credentials (optional, only needed for the MLflow remote tracking server / S3 DVC remote / ECR deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Jeeleej/youtube_sentiment_analysis-MLOPS.git
cd youtube_sentiment_analysis-MLOPS

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies (also installs this project as an editable package via `-e .`)
pip install -r requirements.txt
```

Place your dataset at `dataset/sentiments.csv` with the `clean_comment` and `category` columns described above.

## Running the DVC Pipeline

```bash
# Reproduce the full pipeline (ingestion → preprocessing → training → evaluation)
dvc repro

# Run a single stage
python src/data/data_ingestion.py
python src/data/data_preprocessing.py
python src/model/model_building.py
python src/model/model_evaluation.py
python src/model/register_model.py

# Inspect pipeline DAG
dvc dag
```

Pipeline parameters (test size, TF-IDF settings, model hyperparameters) are centralized in `params.yaml`:

```yaml
data_ingestion:
  test_size: 0.2
model_building:
  max_features: 10000
  ngram_range: [1, 3]
  learning_rate: 0.08081298097796712
  max_depth: 20
  n_estimators: 367
```

## Experiment Tracking with MLflow

`model_evaluation.py` and `register_model.py` point to a remote MLflow tracking server. To use your own:

1. Update the `mlflow.set_tracking_uri(...)` calls in `src/model/model_evaluation.py` and `src/model/register_model.py` with your own tracking server URL (or a local one, e.g. `mlflow ui`).
2. Run the pipeline — parameters, metrics, the confusion matrix image, and the model itself will be logged under the `dvc-pipeline-runs` experiment.
3. `register_model.py` promotes the evaluated run's model to the **Staging** stage under the registered name `my_model`.

## Docker

Build and run the project in a container:

```bash
docker build -t youtube-sentiment-analysis .
docker run -p 8080:8080 youtube-sentiment-analysis
```

The current `Dockerfile` installs `requirements.txt` and starts the app with `python app.py` — add your Flask serving entry point (`app.py`) at the project root before building the image if it isn't present yet.

## CI/CD

`.github/workflows/cicd.yaml` defines a three-stage GitHub Actions pipeline triggered on pushes to `main` (ignoring README-only changes):

1. **Continuous Integration** — checks out the code, lints, and runs unit tests.
2. **Continuous Delivery** — builds the Docker image and pushes it to **Amazon ECR**.
3. **Continuous Deployment** — runs on a **self-hosted runner**, pulls the latest image from ECR, stops any existing container, and starts a fresh one on port `8080`.

Required GitHub Secrets:

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS authentication |
| `AWS_SECRET_ACCESS_KEY` | AWS authentication |
| `AWS_REGION` | AWS region for ECR |
| `AWS_ACCOUNT_ID` | Used to construct the ECR image URI |
| `ECR_REPOSITORY_NAME` | Target ECR repository |

## Configuration

| File | Purpose |
|---|---|
| `params.yaml` | Central place for all pipeline hyperparameters |
| `dvc.yaml` | Declares pipeline stages, dependencies, and outputs |
| `dvc.lock` | Auto-generated lock file with hashes for reproducibility |
| `.dvc/config` | DVC remote configuration (add your own remote, e.g. S3, here) |
| `setup.py` | Packages the project as `youtube_sentiment_analysis` |
| `requirements.txt` | Python dependencies |

## License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

## Author

**Jeel Vaghasiya**
GitHub: [@Jeeleej](https://github.com/Jeeleej)
