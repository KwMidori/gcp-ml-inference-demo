import argparse
import json
from pathlib import Path

from google.cloud import storage


BUCKET_NAME = "gcp-ml-inference-demo-eh01-models"
SELECTION_FILE = "selected_model.json"


def main():
    """選定されたモデルのデプロイ準備とアップロードを行う。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--upload",
        action="store_true",
        help="指定した場合のみCloud Storageへアップロードする",
    )

    args = parser.parse_args()

    with open(
        SELECTION_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        selection = json.load(f)

    selected_version = selection["selected_version"]
    model_file = selection["model_file"]
    reason = selection["reason"]

    gcs_object = f"models/{selected_version}/model.joblib"
    gcs_uri = f"gs://{BUCKET_NAME}/{gcs_object}"

    print("\nデプロイ対象モデル:")
    print(f"version: {selected_version}")
    print(f"local model: {model_file}")
    print(f"GCS object: {gcs_object}")
    print(f"GCS URI: {gcs_uri}")
    print(f"reason: {reason}")

    if not Path(model_file).exists():
        print("\nエラー: ローカルモデルファイルが見つかりません。")
        return

    print("\nローカルモデルファイルを確認しました。")

    if not args.upload:
        print("\n--- Dry Run ---")
        print("以下のアップロードを予定しています。")
        print(f"FROM: {model_file}")
        print(f"TO:   {gcs_uri}")
        print("\nDry Run のため、実際のアップロードは行いません。")
        return

    print("\n--- Upload ---")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_object)

    if blob.exists():
        print(f"アップロード先にはすでにモデルが存在します: {gcs_uri}")
        print("既存のモデルは上書きしません。")
        return

    blob.upload_from_filename(model_file)

    print("アップロードが完了しました。")
    print(f"FROM: {model_file}")
    print(f"TO:   {gcs_uri}")


if __name__ == "__main__":
    main()