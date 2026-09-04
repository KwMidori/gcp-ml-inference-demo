# GCP ML Inference Demo

A small MLOps demonstration project for training, evaluating, selecting, and deploying a text classification model on Google Cloud.

The application classifies Japanese inquiry text into two categories:

- `high`: high-priority / urgent inquiry
- `normal`: normal inquiry

## Architecture

```text
Training Data
    |
    v
train.py
    |
    v
Saved Model (.joblib)
    |
    v
compare_models.py
    |
    v
Model Evaluation / Selection
    |
    v
selected_model.json
    |
    +--------------------+
    |                    |
    v                    v
prepare_deploy.py    prepare_cloud_run_update.py
    |                    |
    v                    v
Cloud Storage         Cloud Run
```

## Technologies

- Python 3.11
- scikit-learn
- Flask
- Gunicorn
- Docker
- Google Cloud Storage
- Artifact Registry
- Cloud Run

## Model

The classifier uses a scikit-learn pipeline based on:

- character-level TF-IDF
- Logistic Regression

Multiple model versions can be evaluated using the same evaluation dataset.

Example model artifacts:

```text
model.joblib
model_v2.joblib
model_v3.joblib
```

## Training

Example:

```powershell
python .\src\train.py `
  --train-data data/training_data_v3.csv `
  --eval-data data/challenge_data.csv `
  --model-out model_v3.joblib
```

## Model Evaluation and Selection

Saved model artifacts are compared using:

```powershell
python .\src\compare_models.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

The current selection policy gives priority to high recall.

If multiple models achieve the same best performance, the current production model is retained instead of performing an unnecessary update.

The selection result is written to:

```text
selected_model.json
```

Example:

```json
{
  "selected_version": "v3",
  "model_file": "model_v3.joblib",
  "current_version": "v3",
  "eval_data": "data/challenge_data.csv",
  "metrics": {
    "accuracy": 0.85,
    "high_recall": 1.0,
    "normal_recall": 0.7
  },
  "reason": "同点のため現行モデルを維持"
}
```

## Model Upload

`prepare_deploy.py` reads `selected_model.json` and determines the Cloud Storage destination.

Dry run:

```powershell
python .\src\prepare_deploy.py
```

Actual upload:

```powershell
python .\src\prepare_deploy.py --upload
```

Existing model versions are not overwritten.

Models are stored using versioned paths such as:

```text
gs://gcp-ml-inference-demo-eh01-models/models/v3/model.joblib
```

## Cloud Run Model Update

`prepare_cloud_run_update.py` compares the selected model version with the current model version.

Dry run:

```powershell
python .\src\prepare_cloud_run_update.py
```

Actual update:

```powershell
python .\src\prepare_cloud_run_update.py --update
```

If the versions are identical, no Cloud Run update is performed.

If they differ, the script updates:

```text
MODEL_OBJECT=models/<version>/model.joblib
```

and Cloud Run creates a new revision.

## Inference API

Endpoint:

```text
POST /predict
```

Example request body:

```json
{
  "text": "一部の利用者が現在ログインできない状態です"
}
```

Example prediction:

```text
high
```

## MLOps Workflow

```text
Train
  ↓
Evaluate saved models
  ↓
Select candidate
  ↓
Write selected_model.json
  ↓
Prepare / upload model to Cloud Storage
  ↓
Determine whether Cloud Run update is required
  ↓
Dry Run
  ↓
Explicit deployment with --update
  ↓
Verify inference
```

The deployment scripts use explicit `--upload` and `--update` options to avoid accidental changes to cloud resources.

## Cost Management

Cloud Run is deployed only when needed for testing and can be deleted after verification to avoid leaving unnecessary resources running.
