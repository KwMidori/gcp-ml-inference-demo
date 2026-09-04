import argparse
import subprocess
import sys


def run_command(command):
    """コマンドを実行し、失敗した場合は処理を停止する。"""

    print("\n========================================")
    print("実行:")
    print(" ".join(command))
    print("========================================\n")

    subprocess.run(
        command,
        check=True,
    )


def main():
    """モデル評価からデプロイ準備までを順番に実行する。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-data",
        required=True,
        help="評価用CSVファイル",
    )

    parser.add_argument(
        "--current-version",
        required=True,
        help="現在採用中のモデルバージョン",
    )

    parser.add_argument(
        "--upload-model",
        action="store_true",
        help="選定モデルをCloud Storageへアップロードする",
    )

    parser.add_argument(
        "--update-cloud-run",
        action="store_true",
        help="Cloud Runを実際に更新する",
    )

    args = parser.parse_args()

    python = sys.executable

    print("\n=== MLOps Pipeline Start ===")

    # 1. モデル比較・選定
    compare_command = [
        python,
        "src/compare_models.py",
        "--eval-data",
        args.eval_data,
        "--current-version",
        args.current_version,
    ]

    run_command(compare_command)

    # 2. Cloud Storage アップロード準備
    deploy_command = [
        python,
        "src/prepare_deploy.py",
    ]

    if args.upload_model:
        deploy_command.append("--upload")

    run_command(deploy_command)

    # 3. Cloud Run 更新判定
    cloud_run_command = [
        python,
        "src/prepare_cloud_run_update.py",
    ]

    if args.update_cloud_run:
        cloud_run_command.append("--apply")

    run_command(cloud_run_command)

    print("\n=== MLOps Pipeline Completed ===")


if __name__ == "__main__":
    main()
