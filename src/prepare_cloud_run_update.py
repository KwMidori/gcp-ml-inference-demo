import argparse
import json
import subprocess


SELECTION_FILE = "selected_model.json"

SERVICE_NAME = "gcp-ml-inference-demo"
REGION = "asia-northeast1"

IMAGE = (
    "asia-northeast1-docker.pkg.dev/"
    "gcp-ml-inference-demo-eh01/"
    "ml-demo/"
    "gcp-ml-inference-demo:latest"
)

MODEL_BUCKET = "gcp-ml-inference-demo-eh01-models"


def cloud_run_service_exists():
    """Cloud Runサービスが存在するか確認する。"""

    command = [
        "gcloud.cmd",
        "run",
        "services",
        "list",
        f"--region={REGION}",
        f"--filter=metadata.name={SERVICE_NAME}",
        "--format=value(metadata.name)",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Cloud Runの状態確認に失敗しました。\n"
            + result.stderr
        )

    return result.stdout.strip() == SERVICE_NAME


def get_current_model_object():
    """Cloud Runで実際に使用中のMODEL_OBJECTを取得する。"""

    command = [
        "gcloud.cmd",
        "run",
        "services",
        "describe",
        SERVICE_NAME,
        f"--region={REGION}",
        "--format=json",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    service = json.loads(result.stdout)

    env_list = (
        service["spec"]
        ["template"]
        ["spec"]
        ["containers"][0]
        .get("env", [])
    )

    for env in env_list:
        if env.get("name") == "MODEL_OBJECT":
            return env.get("value")

    return None


def main():
    """実際のCloud Run状態に基づいてdeploy/updateを判定する。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apply",
        action="store_true",
        help="指定した場合のみCloud Runを実際に変更する",
    )

    args = parser.parse_args()

    with open(
        SELECTION_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        selection = json.load(f)

    selected_version = selection["selected_version"]
    selected_object = f"models/{selected_version}/model.joblib"

    print("\nCloud Run 状態確認:")

    service_exists = cloud_run_service_exists()

    if not service_exists:
        print(
            f"Cloud Runサービス {SERVICE_NAME} は存在しません。"
        )

        command = [
            "gcloud.cmd",
            "run",
            "deploy",
            SERVICE_NAME,
            f"--image={IMAGE}",
            f"--region={REGION}",
            "--platform=managed",
            "--allow-unauthenticated",
            (
                "--set-env-vars="
                f"MODEL_BUCKET={MODEL_BUCKET},"
                f"MODEL_OBJECT={selected_object}"
            ),
        ]

        print("\n判定: 新規デプロイが必要です。")
        print(f"MODEL_OBJECT={selected_object}")

        if not args.apply:
            print("\n--- Dry Run ---")
            print("以下のコマンドを実行する予定です。")
            print()
            print(" ".join(command))
            print(
                "\nDry Run のため、"
                "Cloud Run は変更しません。"
            )
            return

        print("\n--- Deploy ---")
        print("Cloud Run を新規デプロイします。")

        subprocess.run(
            command,
            check=True,
        )

        print("\nCloud Run のデプロイが完了しました。")
        return

    print(
        f"Cloud Runサービス {SERVICE_NAME} は存在します。"
    )

    current_object = get_current_model_object()

    print("\n実際のCloud Run設定:")
    print(f"current MODEL_OBJECT:  {current_object}")
    print(f"selected MODEL_OBJECT: {selected_object}")

    if current_object == selected_object:
        print("\n現行モデルと選定モデルが同じです。")
        print("Cloud Run の更新は不要です。")
        return

    command = [
        "gcloud.cmd",
        "run",
        "services",
        "update",
        SERVICE_NAME,
        f"--region={REGION}",
        f"--update-env-vars=MODEL_OBJECT={selected_object}",
    ]

    print("\n判定: モデルの切り替えが必要です。")

    if not args.apply:
        print("\n--- Dry Run ---")
        print("以下のコマンドを実行する予定です。")
        print()
        print(" ".join(command))
        print(
            "\nDry Run のため、"
            "Cloud Run は変更しません。"
        )
        return

    print("\n--- Update ---")
    print("Cloud Run を更新します。")

    subprocess.run(
        command,
        check=True,
    )

    print("\nCloud Run の更新が完了しました。")


if __name__ == "__main__":
    main()