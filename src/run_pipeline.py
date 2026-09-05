import argparse
import locale
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(run_id):
    """パイプライン実行ログを設定する。"""

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"pipeline_{run_id}.log"

    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 重複したハンドラーが残らないようにする
    logger.handlers.clear()

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_file


def run_command(command, step_name, logger):
    """コマンドを1回実行し、標準出力・標準エラーをログに保存する。"""

    print("\n========================================")
    print(f"STEP: {step_name}")
    print("実行:")
    print(" ".join(command))
    print("========================================\n")

    logger.info("STEP START: %s", step_name)
    logger.info("COMMAND: %s", " ".join(command))

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        check=False,
    )

    # 標準出力
    if result.stdout:
        print(
            result.stdout,
            end="",
        )

        logger.info(
            "STDOUT:\n%s",
            result.stdout.rstrip(),
        )

    # 標準エラー
    if result.stderr:
        print(
            result.stderr,
            end="",
            file=sys.stderr,
        )

        if result.returncode == 0:
            logger.warning(
                "STDERR:\n%s",
                result.stderr.rstrip(),
            )
        else:
            logger.error(
                "STDERR:\n%s",
                result.stderr.rstrip(),
            )

    # 子プロセスが失敗した場合
    if result.returncode != 0:
        logger.error(
            "STEP FAILED: %s (return code=%s)",
            step_name,
            result.returncode,
        )

        print("\n========================================")
        print(
            f"エラー: {step_name} で処理に失敗しました。"
        )
        print(f"return code: {result.returncode}")
        print("パイプラインを停止します。")
        print("========================================")

        raise subprocess.CalledProcessError(
            result.returncode,
            command,
        )

    logger.info(
        "STEP COMPLETED: %s",
        step_name,
    )


def main():
    """MLOpsパイプラインを順番に実行する。"""

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
        help=(
            "指定した場合のみ、選定モデルを"
            "Cloud Storageへアップロードする"
        ),
    )

    parser.add_argument(
        "--update-cloud-run",
        action="store_true",
        help=(
            "指定した場合のみ、Cloud Runへの"
            "変更を実際に適用する"
        ),
    )

    args = parser.parse_args()

    # 1回のパイプライン実行で共通のrun_idを作る
    now = datetime.now().astimezone()
    run_id = now.strftime("%Y%m%d_%H%M%S")

    logger, log_file = setup_logger(run_id)

    # 現在実行中のPythonを使用する
    python = sys.executable

    print("\n=== MLOps Pipeline Start ===")
    print(f"run_id: {run_id}")
    print(f"ログファイル: {log_file}")

    logger.info("MLOps Pipeline Start")
    logger.info("run_id=%s", run_id)
    logger.info("eval_data=%s", args.eval_data)
    logger.info(
        "current_version=%s",
        args.current_version,
    )

    try:
        # ========================================
        # STEP 0: 自動テスト
        # ========================================

        test_command = [
            python,
            "-m",
            "pytest",
        ]

        run_command(
            test_command,
            "Automated tests",
            logger,
        )

        # ========================================
        # STEP 1: モデル比較・選定
        # ========================================

        compare_command = [
            python,
            "src/compare_models.py",
            "--eval-data",
            args.eval_data,
            "--current-version",
            args.current_version,
            "--run-id",
            run_id,
        ]

        run_command(
            compare_command,
            "Model evaluation and selection",
            logger,
        )

        # ========================================
        # STEP 2: モデルアップロード準備
        # ========================================

        deploy_command = [
            python,
            "src/prepare_deploy.py",
        ]

        if args.upload_model:
            deploy_command.append(
                "--upload"
            )

        run_command(
            deploy_command,
            "Model upload preparation",
            logger,
        )

        # ========================================
        # STEP 3: Cloud Run状態確認・反映判定
        # ========================================

        cloud_run_command = [
            python,
            "src/prepare_cloud_run_update.py",
        ]

        if args.update_cloud_run:
            cloud_run_command.append(
                "--apply"
            )

        run_command(
            cloud_run_command,
            "Cloud Run deployment decision",
            logger,
        )

    except subprocess.CalledProcessError:
        logger.error(
            "MLOps Pipeline Failed"
        )

        print(
            f"\n詳細はログを確認してください: "
            f"{log_file}"
        )

        sys.exit(1)

    logger.info(
        "MLOps Pipeline Completed"
    )

    print(
        "\n=== MLOps Pipeline Completed ==="
    )
    print(f"run_id: {run_id}")
    print(f"ログファイル: {log_file}")


if __name__ == "__main__":
    main()