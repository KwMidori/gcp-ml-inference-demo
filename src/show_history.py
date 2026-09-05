import argparse
import json
from pathlib import Path

import pandas as pd


HISTORY_DIR = Path("history")


def load_history():
    """保存されたモデル選定履歴を読み込む。"""

    history_files = sorted(
        HISTORY_DIR.glob("selected_model_*.json"),
        reverse=True,
    )

    rows = []

    for history_file in history_files:

        with open(
            history_file,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})

        # 古い履歴にrun_idがない場合はファイル名から取得
        run_id = data.get("run_id")

        if not run_id:
            run_id = history_file.stem.replace(
                "selected_model_",
                "",
            )

        rows.append(
            {
                "run_id": run_id,
                "evaluated_at": data.get("evaluated_at"),
                "selected": data.get("selected_version"),
                "current": data.get("current_version"),
                "accuracy": metrics.get("accuracy"),
                "high_recall": metrics.get("high_recall"),
                "normal_recall": metrics.get("normal_recall"),
                "reason": data.get("reason"),
                "eval_data": data.get("eval_data"),
            }
        )

    return rows


def main():
    """保存されたモデル選定履歴を一覧表示する。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="表示する履歴件数",
    )

    parser.add_argument(
        "--run-id",
        required=False,
        help="指定したrun_idだけを表示する",
    )

    args = parser.parse_args()

    rows = load_history()

    if not rows:
        print("\n選定履歴がありません。")
        return

    history_df = pd.DataFrame(rows)

    # run_idが指定された場合は、その実行だけに絞る
    if args.run_id:

        history_df = history_df[
            history_df["run_id"] == args.run_id
        ]

        if history_df.empty:
            print(
                f"\nrun_id={args.run_id} "
                "の履歴は見つかりませんでした。"
            )
            return

    else:
        history_df = history_df.head(args.limit)

    print("\nモデル選定履歴:")

    display_columns = [
        "run_id",
        "selected",
        "current",
        "accuracy",
        "high_recall",
        "normal_recall",
        "reason",
    ]

    print(
        history_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # run_id指定時は詳細も表示
    if args.run_id:

        row = history_df.iloc[0]

        print("\n詳細:")
        print(f"run_id:       {row['run_id']}")
        print(f"evaluated_at: {row['evaluated_at']}")
        print(f"eval_data:    {row['eval_data']}")
        print(f"selected:     {row['selected']}")
        print(f"current:      {row['current']}")
        print(f"accuracy:     {row['accuracy']:.2f}")
        print(f"high_recall:  {row['high_recall']:.2f}")
        print(f"normal_recall:{row['normal_recall']:.2f}")
        print(f"reason:       {row['reason']}")


if __name__ == "__main__":
    main()