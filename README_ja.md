# GCP ML Inference Demo

Google Cloud 上で、日本語テキスト分類モデルの学習・評価・選定・履歴管理・デプロイを行う、小規模な MLOps デモプロジェクトです。

このアプリケーションは、問い合わせテキストを次の2種類に分類します。

- `high`: 優先度が高い／緊急性の高い問い合わせ
- `normal`: 通常の問い合わせ

## 概要

このプロジェクトは、シンプルな Flask 推論 API から始まり、段階的に小規模な MLOps ワークフローへ拡張してきました。

現在、以下の機能を備えています。

- モデル学習
- モデルのバージョン管理
- 固定評価データセット
- モデル比較
- 評価基準に基づくモデル選定
- `predict_proba()` による信頼度比較
- 評価履歴・選定履歴の保存
- 共通 `run_id` による追跡
- 実行ログ
- `pytest` による自動テスト
- Cloud Storage へのモデルアップロード準備
- Cloud Run の状態確認
- deploy / update / no-op の自動判定
- クラウド変更処理に対する Dry Run の安全機構
- 1コマンドによるパイプライン実行
- GitHub Actions による CI/CD 自動化
- Workload Identity Federation による鍵レスの Google Cloud 認証
- Cloud Storage へのモデル自動アップロード
- Cloud Run の deploy / update / no-op 自動処理
- デプロイ後 API の smoke test

## アーキテクチャ

```text
Git push to main
      |
      v
GitHub Actions
      |
      +----------------------------+
      |                            |
      v                            v
Workload Identity             pytest
Federation                        |
      |                            v
      |                     compare_models.py
      |                            |
      |                 +----------+----------+
      |                 |                     |
      |                 v                     v
      |       evaluation_results.csv   selected_model.json
      |                 |                     |
      |                 +----------+----------+
      |                            |
      |                            v
      |                 Upload selected model
      |                     if not present
      |                            |
      |                            v
      |                    Cloud Storage
      |                            |
      |                            v
      |              prepare_cloud_run_update.py
      |                            |
      |                 +----------+----------+
      |                 |          |          |
      |                 v          v          v
      |               deploy     update      no-op
      |                 \          |          /
      |                  \         |         /
      |                   +--------+--------+
      |                            |
      |                            v
      +-----------------------> Cloud Run
                                   |
                         loads selected model
                                   |
                                   v
                              POST /predict
                                   |
                                   v
                         API smoke test in CI

Artifact Registry
      |
      +------ container image ------> Cloud Run
```

主要な処理は `run_pipeline.py` を使ってまとめて実行することもできます。

## 使用技術

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
- GitHub Actions
- Workload Identity Federation

## モデル

分類器は、scikit-learn の以下の要素で構成したパイプラインを使用しています。

- 文字単位 TF-IDF
- Logistic Regression

現在のモデルファイルは次のとおりです。

```text
model.joblib
model_v2.joblib
model_v3.joblib
```

同一の評価データセットで複数のモデルバージョンを比較できるよう、モデルは意図的に別々の成果物として保持しています。

## 学習

実行例:

```powershell
python .\src\train.py `
  --train-data data/training_data_v3.csv `
  --eval-data data/challenge_data.csv `
  --model-out model_v3.joblib
```

## 評価データ

モデルバージョンを再現可能な形で比較するため、固定の評価データセットを使用しています。

例:

```text
data/test_data.csv
data/challenge_data.csv
```

`challenge_data.csv` には、緊急性を示すように見える単語を含むが実際には通常問い合わせである例など、より難しい境界事例を含めています。

## モデル評価と選定

保存済みモデル成果物は、次のコマンドで比較します。

```powershell
python .\src\compare_models.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

現在の選定ポリシーでは、`high` クラスの recall を優先します。

モデルは次の条件を満たす必要があります。

```text
high_recall >= 1.0
```

条件を満たしたモデルの中で accuracy を比較します。

現行モデルが最高 accuracy と同点であれば、不必要なモデル切り替えを避けるため、現行モデルを維持します。

評価結果の例:

```text
version  accuracy  high_recall  normal_recall  avg_confidence
v1       0.55      1.00         0.10           0.60
v2       0.85      1.00         0.70           0.71
v3       0.85      1.00         0.70           0.71
```

## 信頼度分析

`compare_confidence.py` は `predict_proba()` を使って v2 と v3 を比較します。

実行例:

```powershell
python .\src\compare_confidence.py `
  --eval-data data/challenge_data.csv
```

このスクリプトでは、次の項目を比較します。

```text
v2_prediction
v3_prediction
v2_high_prob
v3_high_prob
prob_change
abs_prob_diff
```

これにより、現在の challenge データセット上では最終的な分類結果が同一であっても、v2 と v3 が異なる学習済みモデルであることを確認しました。

## 選定結果

最新のモデル選定結果は次のファイルに書き出されます。

```text
selected_model.json
```

例:

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

## 履歴管理

各評価実行は共通の `run_id` で関連づけられます。

例:

```text
run_id = 20260905_165625
```

同じ ID が次のファイル群で使用されます。

```text
logs/pipeline_20260905_165625.log
history/evaluation_results_20260905_165625.csv
history/selected_model_20260905_165625.json
```

これにより、何を評価したか、どのモデルを選んだか、なぜそのモデルを選んだか、パイプラインが正常に完了したかを追跡できます。

## 選定履歴の確認

次のコマンドを使用します。

```powershell
python .\src\show_history.py
```

特定の実行を確認する場合:

```powershell
python .\src\show_history.py `
  --run-id 20260905_165625
```

## ログ

`run_pipeline.py` は、パイプライン実行ごとにログファイルを作成します。

例:

```text
logs/pipeline_20260905_165625.log
```

ログには次の内容が記録されます。

- パイプライン開始
- run_id
- 評価データ
- 現行モデルバージョン
- 実行コマンド
- 各ステップの開始・完了
- stdout
- stderr
- Python traceback
- パイプラインの完了または失敗

## 自動テスト

モデル成果物に対するテストは `pytest` で実装しています。

実行:

```powershell
python -m pytest
```

現在のテストでは、以下を確認しています。

- すべてのモデルファイルが存在すること
- すべてのモデル成果物を読み込めること
- 想定したクラスが存在すること
- `predict()` が動作すること
- `predict_proba()` が動作し、妥当な確率を返すこと

現在の結果:

```text
5 passed
```

自動テストは MLOps パイプラインの最初のステップとしても実行されます。

テストが1つでも失敗した場合、モデル評価やクラウド関連処理に進む前にパイプラインを停止します。

## モデルのアップロード

`prepare_deploy.py` は `selected_model.json` を読み込み、Cloud Storage 上の保存先を決定します。

Dry Run:

```powershell
python .\src\prepare_deploy.py
```

実際にアップロードする場合:

```powershell
python .\src\prepare_deploy.py --upload
```

既存のモデルバージョンは上書きしません。

モデルは次のようなバージョン付きパスに保存します。

```text
gs://gcp-ml-inference-demo-eh01-models/models/v3/model.joblib
```

## Cloud Run の状態確認とデプロイ

`prepare_cloud_run_update.py` は、必要な操作を決定する前に実際の Cloud Run の状態を確認します。

ローカルに記録した現行バージョンだけに依存せず、Google Cloud の実状態を問い合わせます。

判定フロー:

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

Dry Run:

```powershell
python .\src\prepare_cloud_run_update.py
```

必要なクラウド変更を適用する場合:

```powershell
python .\src\prepare_cloud_run_update.py --apply
```

## 1コマンド MLOps パイプライン

`run_pipeline.py` は主要な処理を1つのコマンドにまとめます。

安全な Dry Run:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3
```

パイプラインの順序:

```text
1. 自動テスト
2. モデル評価と選定
3. 評価履歴・選定履歴の保存
4. モデルアップロード準備
5. Cloud Run の実状態確認
6. deploy / update / no-op の判定
7. 明示的に許可されていない限り、クラウド資源を変更する前に停止
```

モデルアップロードを許可する場合:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model
```

Cloud Run のデプロイまたは更新を許可する場合:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --update-cloud-run
```

両方を許可する場合:

```powershell
python .\src\run_pipeline.py `
  --eval-data data/challenge_data.csv `
  --current-version v3 `
  --upload-model `
  --update-cloud-run
```

明示的な変更許可フラグがない場合、クラウド変更処理は Dry Run のままです。

## 推論 API

エンドポイント:

```text
POST /predict
```

リクエスト例:

```json
{
  "text": "一部の利用者が現在ログインできない状態です"
}
```

予測結果の例:

```text
high
```

## 確認済みの End-to-End ワークフロー

次の一連の流れを実際に確認済みです。

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

v2 から v3 へのモデル切り替えについても、Cloud Run の `MODEL_OBJECT` を更新して新しい revision を作成し、トラフィックを100%新 revision に向け、推論 API を確認するところまでテスト済みです。

## 安全対策

このワークフローには、以下の安全機構を組み込んでいます。

- モデル評価前に自動テストを実行
- テストやコマンドが失敗した場合は即時停止
- ローカルのクラウド変更スクリプトは、明示的なフラグがない限り Dry Run
- `main` ブランチの CI/CD は、テストとモデル評価が成功した後にのみ認証済みクラウド変更を実行
- Cloud Storage 上の既存モデルバージョンは上書きしない
- デプロイ判断前に Cloud Run の実状態を確認
- 不要なモデル切り替えを回避
- Dry Run では実行予定のコマンドを表示
- stdout、stderr、traceback をログとして保存
- `run_id` によりログと評価履歴を関連づけ

## 生成ファイルと Git

実行時に生成される成果物は、意図的に Git 管理対象から外しています。

例:

```text
logs/
history/
evaluation_results.csv
selected_model.json
__pycache__/
*.pyc
```

これらのファイルはローカルで生成され、`.gitignore` によって除外されます。

## コスト管理

CI/CD ワークフローは、変更を行う前に現在の Cloud Run の状態を確認します。

選定されたモデルがすでに稼働中であれば、新しい Cloud Run revision は作成しません。

デモ用サービスが不要な場合は、Cloud Run を削除できます。

```powershell
gcloud.cmd run services delete gcp-ml-inference-demo `
  --region=asia-northeast1
```

削除確認:

```powershell
gcloud.cmd run services list `
  --region=asia-northeast1
```

削除に成功すると、次のように表示されます。

```text
Listed 0 items.
```

Cloud Run サービスを削除しても、Cloud Storage 上のモデル成果物と Artifact Registry 上のコンテナイメージは別途保持されます。
