import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


MODELS = {
    "v1": "model.joblib",
    "v2": "model_v2.joblib",
    "v3": "model_v3.joblib",
}


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
    "--run-id",
    required=False,
    help="パイプライン実行ID",
)

args = parser.parse_args()


# 実行日時
now = datetime.now().astimezone()

# run_pipeline.py から run_id が渡された場合はそれを使用する。
# compare_models.py を単独実行した場合は現在時刻から生成する。
if args.run_id:
    run_id = args.run_id
else:
    run_id = now.strftime("%Y%m%d_%H%M%S")


# 履歴保存先
history_dir = Path("history")
history_dir.mkdir(exist_ok=True)


# 評価データ読み込み
df_test = pd.read_csv(args.eval_data)

X_test = df_test["text"]
y_test = df_test["label"]


results = []


# 保存済みモデルを順番に評価
for version, model_file in MODELS.items():

    model = joblib.load(model_file)

    # クラス予測
    predictions = model.predict(X_test)

    # 各クラスの予測確率
    probabilities = model.predict_proba(X_test)

    # model.classes_ の並びを取得
    class_names = list(model.classes_)

    # 各データについて、
    # 実際に予測したクラスの確率を取得する
    predicted_confidences = []

    for prediction, probability_row in zip(
        predictions,
        probabilities,
    ):
        class_index = class_names.index(prediction)

        predicted_confidences.append(
            probability_row[class_index]
        )

    # 予測クラスに対する平均確信度
    average_confidence = (
        sum(predicted_confidences)
        / len(predicted_confidences)
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # Precision / Recall / F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=["high", "normal"],
        zero_division=0,
    )

    results.append(
        {
            "version": version,
            "accuracy": accuracy,
            "high_precision": precision[0],
            "high_recall": recall[0],
            "high_f1": f1[0],
            "normal_precision": precision[1],
            "normal_recall": recall[1],
            "normal_f1": f1[1],
            "avg_confidence": average_confidence,
        }
    )


# 評価結果をDataFrame化
result_df = pd.DataFrame(results)


print("\nモデル比較結果:")

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# 最新の評価結果を保存
result_df.to_csv(
    "evaluation_results.csv",
    index=False,
)


# 評価結果を履歴として保存
history_eval_file = (
    history_dir
    / f"evaluation_results_{run_id}.csv"
)

result_df.to_csv(
    history_eval_file,
    index=False,
)

print(
    f"\n評価履歴を {history_eval_file} に保存しました。"
)


# high recall が 1.0 のモデルだけを採用候補にする
candidates = result_df[
    result_df["high_recall"] >= 1.0
]


if candidates.empty:

    print("\n採用基準を満たすモデルはありません。")

else:

    # 現行モデルが候補に含まれているか確認
    current = candidates[
        candidates["version"] == args.current_version
    ]

    # 候補の中で最高accuracyを取得
    best_accuracy = candidates["accuracy"].max()

    # 最高accuracyのモデルを抽出
    best_candidates = candidates[
        candidates["accuracy"] == best_accuracy
    ]

    # 現行モデルが最高accuracyと同点なら現行モデルを維持
    if (
        not current.empty
        and current.iloc[0]["accuracy"] == best_accuracy
    ):

        selected_model = current.iloc[0]
        reason = "同点のため現行モデルを維持"

    else:

        selected_model = best_candidates.iloc[0]
        reason = "現行モデルより高い性能"


    print("\n採用モデル:")

    print(
        f"version={selected_model['version']}, "
        f"accuracy={selected_model['accuracy']:.2f}, "
        f"high_recall={selected_model['high_recall']:.2f}, "
        f"normal_recall={selected_model['normal_recall']:.2f}, "
        f"avg_confidence="
        f"{selected_model['avg_confidence']:.2f}"
    )

    print(f"判定理由: {reason}")


    # 選定結果をJSON用の辞書にまとめる
    selection_result = {
        "run_id": run_id,
        "evaluated_at": now.isoformat(timespec="seconds"),
        "selected_version": str(
            selected_model["version"]
        ),
        "model_file": MODELS[
            str(selected_model["version"])
        ],
        "current_version": args.current_version,
        "eval_data": args.eval_data,
        "metrics": {
            "accuracy": float(
                selected_model["accuracy"]
            ),
            "high_recall": float(
                selected_model["high_recall"]
            ),
            "normal_recall": float(
                selected_model["normal_recall"]
            ),
            "avg_confidence": float(
                selected_model["avg_confidence"]
            ),
        },
        "reason": reason,
    }


    # 最新の選定結果を保存
    with open(
        "selected_model.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection_result,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        "\n選定結果を selected_model.json に保存しました。"
    )


    # 選定結果を履歴として保存
    history_selection_file = (
        history_dir
        / f"selected_model_{run_id}.json"
    )

    with open(
        history_selection_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selection_result,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print(
        f"選定履歴を {history_selection_file} に保存しました。"
    )