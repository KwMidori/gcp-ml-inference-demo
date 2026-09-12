import argparse
import json
import os
import subprocess


SELECTION_FILE = "selected_model.json"

SERVICE_NAME = "gcp-ml-inference-demo"
REGION = "asia-northeast1"
GCLOUD = "gcloud.cmd" if os.name == "nt" else "gcloud"

DEFAULT_IMAGE = (
    "asia-northeast1-docker.pkg.dev/"
    "gcp-ml-inference-demo-eh01/"
    "ml-demo/"
    "gcp-ml-inference-demo:latest"
)

IMAGE = os.environ.get(
    "CLOUD_RUN_IMAGE",
    DEFAULT_IMAGE,
)

MODEL_BUCKET = "gcp-ml-inference-demo-eh01-models"


def cloud_run_service_exists():
    """Cloud Runサービスが存在するか確認する。"""

    command = [
        GCLOUD,
        "run",
        "services",
        "list",
        f"--region={REGION}",
        "--format=value(name)",
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

    service_names = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }

    return SERVICE_NAME in service_names


def get_current_model_object():
    """Cloud Runで実際に使用中のMODEL_OBJECTを取得する。"""

    command = [
        GCLOUD,
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


def get_current_image():
    """Cloud Runで実際に使用中のコンテナイメージを取得する。"""

    command = [
        GCLOUD,
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

    return (
        service["spec"]
        ["template"]
        ["spec"]
        ["containers"][0]
        .get("image")
    )


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
            GCLOUD,
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
        print(f"IMAGE={IMAGE}")

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
    current_image = get_current_image()

    print("\n実際のCloud Run設定:")
    print(f"current MODEL_OBJECT:  {current_object}")
    print(f"selected MODEL_OBJECT: {selected_object}")
    print(f"current image:         {current_image}")
    print(f"selected image:        {IMAGE}")

    model_changed = current_object != selected_object
    image_changed = current_image != IMAGE

    if not model_changed and not image_changed:
        print(
            "\nモデルとコンテナイメージの両方が"
            "現行と同じです。"
        )
        print("Cloud Run の更新は不要です。")
        return

    command = [
        GCLOUD,
        "run",
        "services",
        "update",
        SERVICE_NAME,
        f"--region={REGION}",
        f"--image={IMAGE}",
        f"--update-env-vars=MODEL_OBJECT={selected_object}",
    ]

    print("\n判定: Cloud Run の更新が必要です。")

    if model_changed:
        print("- モデルが変更されています。")

    if image_changed:
        print("- コンテナイメージが変更されています。")

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
