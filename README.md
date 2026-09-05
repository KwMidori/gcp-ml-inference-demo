# GCP ML Inference Demo

A small MLOps demonstration project for training, evaluating, selecting, tracking, and deploying a Japanese text classification model on Google Cloud.

The application classifies inquiry text into two categories:

- `high`: high-priority / urgent inquiry
- `normal`: normal inquiry

## Overview

This project started as a simple Flask inference API and was gradually extended into a small MLOps workflow.

Current capabilities include:

- model training
- model versioning
- fixed evaluation datasets
- model comparison
- model selection based on evaluation criteria
- confidence comparison with `predict_proba()`
- evaluation and selection history
- common `run_id` tracking
- execution logging
- automated tests with `pytest`
- Cloud Storage model upload preparation
- Cloud Run state detection
- automatic deploy / update / no-op decision
- dry-run safety for cloud-changing operations
- one-command pipeline orchestration

## Architecture

```text
Training Data
    |
    v
train.py
    |
    v
Saved Model Artifacts
    |
    v
pytest
    |
    v
compare_models.py
    |
    +----------------------+
    |                      |
    v                      v
evaluation_results.csv   selected_model.json
    |                      |
    v                      v
history/                 history/
    |                      |
    +----------+-----------+
               |
               v
        prepare_deploy.py
               |
               v
         Cloud Storage
               |
               v
prepare_cloud_run_update.py
               |
               v
     Cloud Run state check
               |
       +-------+-------+
       |               |
       v               v
    deploy           update
       \               /
        \             /
         +---- no-op +
```

The main steps can also be executed together through `run_pipeline.py`.

## Technologies

- Python 3.11
- scikit-learn
- pandas
- joblib
- Flask
- Gunicorn
- pytest
- Docker
- Google Cloud Storage
- Artifact Registry
- Cloud Run

## Model

The classifier uses a scikit-learn pipeline based on:

- character-level TF-IDF
- Logistic Regression

Current model artifacts:

```text
model.joblib
model_v2.joblib
model_v3.joblib
```

The models are intentionally kept as separate artifacts so that multiple versions can be compared using the same evaluation dataset.

## Training

Example:

```powershell
python .\src\train.py `
  --train-data data/training_data_v3.csv `
  --eval-data data/challenge_data.csv `
  --model-out model_v3.joblib
```

## Evaluation Data

The project currently uses fixed evaluation datasets so that model versions can be compared reproducibly.

Examples:

```text
data/test_data.csv
data/challenge_data.csv
```

`challenge_data.csv` contains more difficult boundary cases, including examples with misleading urgency-related keywords.

## Model Evaluation and Selection

Saved model artifacts are compared using:

```powershell
python .\src\compare_models.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

The current selection policy prioritizes `high` recall.

A model must satisfy:

```text
high_recall >= 1.0
```

Among eligible models, accuracy is compared.

If the current model is tied for the best accuracy, the current model is retained instead of performing an unnecessary model switch.

Example result:

```text
version  accuracy  high_recall  normal_recall  avg_confidence
v1       0.55      1.00         0.10           0.60
v2       0.85      1.00         0.70           0.71
v3       0.85      1.00         0.70           0.71
```

## Confidence Analysis

`compare_confidence.py` compares v2 and v3 using `predict_proba()`.

Run:

```powershell
python .\src\compare_confidence.py `
  --eval-data data/challenge_data.csv
```

The script compares:

```text
v2_prediction
v3_prediction
v2_high_prob
v3_high_prob
prob_change
abs_prob_diff
```

This confirmed that v2 and v3 are distinct learned models even when their final class predictions are identical on the current challenge dataset.

## Selection Result

The latest model selection is written to:

```text
selected_model.json
```

Example:

```json
{
  "run_id": "20260905_165625",
  "evaluated_at": "2026-09-05T16:56:25+09:00",
  "selected_version": "v3",
  "model_file": "model_v3.joblib",
  "current_version": "v3",
  "eval_data": "data/challenge_data.csv",
  "metrics": {
    "accuracy": 0.85,
    "high_recall": 1.0,
    "normal_recall": 0.7,
    "avg_confidence": 0.71
  },
  "reason": "同点のため現行モデルを維持"
}
```

## History Tracking

Each evaluation run can be associated with a common `run_id`.

Example:

```text
run_id = 20260905_165625
```

The same ID is used for:

```text
logs/pipeline_20260905_165625.log
history/evaluation_results_20260905_165625.csv
history/selected_model_20260905_165625.json
```

This makes it possible to trace what was evaluated, which model was selected, why it was selected, and whether the pipeline completed successfully.

## Viewing Selection History

Use:

```powershell
python .\src\show_history.py
```

To inspect a specific run:

```powershell
python .\src\show_history.py `
  --run-id 20260905_165625
```

## Logging

`run_pipeline.py` creates a log file for each pipeline execution.

Example:

```text
logs/pipeline_20260905_165625.log
```

The log records:

- pipeline start
- run_id
- evaluation data
- current model version
- executed commands
- step start / completion
- stdout
- stderr
- Python tracebacks
- pipeline completion or failure

## Automated Tests

Model artifact tests are implemented with `pytest`.

Run:

```powershell
python -m pytest
```

Current tests verify that:

- all model files exist
- all model artifacts can be loaded
- expected classes are present
- `predict()` works
- `predict_proba()` works and returns valid probabilities

Current result:

```text
5 passed
```

The automated tests are also executed as the first step of the MLOps pipeline.

If any test fails, the pipeline stops before model evaluation or cloud-related steps.

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

## Cloud Run State Check and Deployment

`prepare_cloud_run_update.py` checks the actual Cloud Run state before deciding what action is required.

The script queries Google Cloud instead of relying only on a locally recorded current version.

Decision flow:

```text
Check actual Cloud Run state
        |
        v
Does service exist?
   |               |
  No              Yes
   |               |
   v               v
Deploy needed   Read actual MODEL_OBJECT
                   |
                   v
             Same as selected?
                |       |
               Yes      No
                |       |
                v       v
             No action  Update needed
```

Dry run:

```powershell
python .\src\prepare_cloud_run_update.py
```

Apply the required cloud change:

```powershell
python .\src\prepare_cloud_run_update.py --apply
```

## One-Command MLOps Pipeline

`run_pipeline.py` connects the main workflow into one command.

Safe dry run:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

Pipeline order:

```text
1. Automated tests
2. Model evaluation and selection
3. Save evaluation / selection history
4. Model upload preparation
5. Check actual Cloud Run state
6. Decide deploy / update / no-op
7. Stop before changing cloud resources unless explicitly allowed
```

Allow model upload:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model
```

Allow Cloud Run deployment or update:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --update-cloud-run
```

Allow both:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model `
  --update-cloud-run
```

Without the explicit change flags, cloud-changing actions remain in dry-run mode.

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

## Verified End-to-End Workflow

The following workflow has been tested:

```text
Train
  ↓
Run pytest
  ↓
Evaluate saved models
  ↓
Select candidate
  ↓
Write selected_model.json
  ↓
Write evaluation and selection history
  ↓
Prepare / upload model to Cloud Storage
  ↓
Check actual Cloud Run state
  ↓
Deploy if service does not exist
  or
Update MODEL_OBJECT if model differs
  or
Do nothing if selected model is already active
  ↓
Verify inference
```

A model switch from v2 to v3 was also tested by updating the Cloud Run `MODEL_OBJECT`, creating a new revision, routing 100% of traffic to the new revision, and verifying the prediction API.

## Safety Features

The workflow includes several safeguards:

- automated tests run before model evaluation
- pipeline stops immediately on test or command failure
- cloud-changing actions are disabled by default
- explicit flags are required for upload or Cloud Run changes
- existing model versions in Cloud Storage are not overwritten
- actual Cloud Run state is checked before deployment decisions
- unnecessary model switches are avoided
- dry-run output shows the command that would be executed
- logs preserve stdout, stderr, and traceback details
- `run_id` links logs and evaluation history

## Generated Files and Git

Runtime-generated artifacts are intentionally excluded from Git.

Examples:

```text
logs/
history/
evaluation_results.csv
selected_model.json
__pycache__/
*.pyc
```

These files are generated locally and are ignored through `.gitignore`.

## Cost Management

Cloud Run is deployed only when needed for testing.

After verification, the service can be deleted:

```powershell
gcloud.cmd run services delete gcp-ml-inference-demo `
  --region=asia-northeast1
```

Verify deletion:

```powershell
gcloud.cmd run services list `
  --region=asia-northeast1
```

A successful cleanup shows:

```text
Listed 0 items.
```
