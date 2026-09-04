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

The individual steps can also be executed together through `run_pipeline.py`.

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

## Cloud Run State Check and Deployment

`prepare_cloud_run_update.py` checks the actual state of Cloud Run before deciding what action is required.

The script does not rely only on the local `current_version` value. It queries Google Cloud to determine whether the service exists and, if it does, which `MODEL_OBJECT` is actually configured.

Decision flow:

```text
Check actual Cloud Run state
        |
        v
Does the service exist?
   |               |
  No              Yes
   |               |
   v               v
Deploy needed   Read current MODEL_OBJECT
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

Apply the required change:

```powershell
python .\src\prepare_cloud_run_update.py --apply
```

Possible outcomes:

- Cloud Run service does not exist: a new deployment is prepared.
- Cloud Run service exists and already uses the selected model: no action is taken.
- Cloud Run service exists but uses a different model: the service is updated to the selected `MODEL_OBJECT`.

The script uses the following model path convention:

```text
MODEL_OBJECT=models/<version>/model.joblib
```

## One-Command MLOps Pipeline

`run_pipeline.py` connects the main MLOps steps and executes them in sequence.

Dry run:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

This performs:

```text
1. Compare saved models
2. Select a candidate model
3. Write selected_model.json
4. Prepare the Cloud Storage upload
5. Check the actual Cloud Run state
6. Decide whether deploy, update, or no action is required
7. Stop before modifying cloud resources
```

To allow model upload:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model
```

To allow Cloud Run deployment or update:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --update-cloud-run
```

To allow both model upload and Cloud Run changes:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model `
  --update-cloud-run
```

`run_pipeline.py` exposes `--update-cloud-run` to the user and internally passes `--apply` to `prepare_cloud_run_update.py`.

Without `--upload-model` or `--update-cloud-run`, the pipeline acts as a safe dry run for cloud changes.

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

## Verified MLOps Workflow

The following workflow has been tested end-to-end:

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
Check actual Cloud Run state
  ↓
Deploy if the service does not exist
  or
Update MODEL_OBJECT if the model differs
  or
Do nothing if the selected model is already active
  ↓
Verify inference
```

A model switch from `v2` to `v3` was tested by updating the Cloud Run `MODEL_OBJECT`, creating a new revision, routing 100% of traffic to the new revision, and verifying the prediction API.

## Safety Features

The workflow includes several safeguards:

- cloud-changing actions are disabled by default
- explicit flags are required for upload or Cloud Run changes
- existing model versions in Cloud Storage are not overwritten
- Cloud Run state is checked against the real Google Cloud environment
- unnecessary Cloud Run updates are skipped
- dry-run output shows the command that would be executed before any cloud change

## Cost Management

Cloud Run is deployed only when needed for testing and can be deleted after verification to avoid leaving unnecessary resources running.

Example:

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
