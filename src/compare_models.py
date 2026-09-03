import joblib
import pandas as pd
import argparse
import argparse
import json

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

args = parser.parse_args()


df_test = pd.read_csv(args.eval_data)

X_test = df_test["text"]
y_test = df_test["label"]

all_predictions = {}

results = []


for version, model_file in MODELS.items():

    model = joblib.load(model_file)

    predictions = model.predict(X_test)

    all_predictions[version] = predictions

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
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
        }
    )


result_df = pd.DataFrame(results)

print("\nモデル比較結果:")

print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

candidates = result_df[
    result_df["high_recall"] >= 1.0
]

if candidates.empty:
    print("\n採用基準を満たすモデルはありません。")

else:
    current = candidates[
        candidates["version"] == args.current_version
    ]

    best_accuracy = candidates["accuracy"].max()

    best_candidates = candidates[
        candidates["accuracy"] == best_accuracy
    ]

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
        f"normal_recall={selected_model['normal_recall']:.2f}"
    )

    print(f"判定理由: {reason}")

    selection_result = {
    "selected_version": selected_model["version"],
    "model_file": MODELS[selected_model["version"]],
    "current_version": args.current_version,
    "eval_data": args.eval_data,
    "metrics": {
        "accuracy": float(selected_model["accuracy"]),
        "high_recall": float(selected_model["high_recall"]),
        "normal_recall": float(selected_model["normal_recall"]),
    },
    "reason": reason,
}

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

print("\n選定結果を selected_model.json に保存しました。")