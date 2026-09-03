import json


SELECTION_FILE = "selected_model.json"

SERVICE_NAME = "gcp-ml-inference-demo"
REGION = "asia-northeast1"


def main():
    """選定モデルに基づいてCloud Runの更新が必要か判定する。"""

    with open(
        SELECTION_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        selection = json.load(f)

    selected_version = selection["selected_version"]
    current_version = selection["current_version"]

    selected_object = f"models/{selected_version}/model.joblib"

    print("\nCloud Run 更新判定:")
    print(f"current version:  {current_version}")
    print(f"selected version: {selected_version}")

    if selected_version == current_version:
        print("\n現行モデルと選定モデルが同じです。")
        print("Cloud Run の更新は不要です。")
        return

    print("\nモデルの切り替えが必要です。")
    print(f"MODEL_OBJECT={selected_object}")

    print("\n--- Dry Run ---")
    print("以下のコマンドを実行する予定です。")
    print()
    print(
        f"gcloud.cmd run services update {SERVICE_NAME} "
        f"--region={REGION} "
        f"--update-env-vars=MODEL_OBJECT={selected_object}"
    )

    print("\nDry Run のため、Cloud Run は変更しません。")


if __name__ == "__main__":
    main()
