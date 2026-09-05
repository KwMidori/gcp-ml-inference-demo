import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


MODEL_V2 = "model_v2.joblib"
MODEL_V3 = "model_v3.joblib"

HISTORY_DIR = Path("history")


def get_high_probabilities(model, x_data):
    """各データについて high クラスの予測確率を取得する。"""

    probabilities = model.predict_proba(x_data)

    class_names = list(model.classes_)

    if "high" not in class_names:
        raise ValueError(
            "モデルのクラスに 'high' が存在しません。"
        )

    high_index = class_names.index("high")

    return probabilities[:, high_index]


def main():
    """v2とv3の予測確率を比較する。"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-data",
        required=True,
        help="評価用CSVファイル",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="確信度差が大きい順に表示する件数",
    )

    args = parser.parse_args()

    # 評価データ
    df_test = pd.read_csv(args.eval_data)

    x_test = df_test["text"]
    y_test = df_test["label"]

    # モデル読み込み
    model_v2 = joblib.load(MODEL_V2)
    model_v3 = joblib.load(MODEL_V3)

    # クラス予測
    pred_v2 = model_v2.predict(x_test)
    pred_v3 = model_v3.predict(x_test)

    # high クラスの確率
    high_prob_v2 = get_high_probabilities(
        model_v2,
        x_test,
    )

    high_prob_v3 = get_high_probabilities(
        model_v3,
        x_test,
    )

    # 比較表
    comparison_df = pd.DataFrame(
        {
            "text": x_test,
            "actual": y_test,
            "v2_prediction": pred_v2,
            "v3_prediction": pred_v3,
            "v2_high_prob": high_prob_v2,
            "v3_high_prob": high_prob_v3,
        }
    )

    # v3 - v2
    comparison_df["prob_change"] = (
        comparison_df["v3_high_prob"]
        - comparison_df["v2_high_prob"]
    )

    # 差の絶対値
    comparison_df["abs_prob_diff"] = (
        comparison_df["prob_change"].abs()
    )

    # 差が大きい順
    comparison_df = comparison_df.sort_values(
        by="abs_prob_diff",
        ascending=False,
    )

    print("\nv2 と v3 の確信度比較:")
    print("=" * 70)

    top_results = comparison_df.head(args.limit)

    for rank, (_, row) in enumerate(
        top_results.iterrows(),
        start=1,
    ):
        print(f"\n[{rank}] {row['text']}")
        print(f"  正解ラベル : {row['actual']}")
        print(
            f"  v2         : {row['v2_prediction']} "
            f"(high確率={row['v2_high_prob']:.3f})"
        )
        print(
            f"  v3         : {row['v3_prediction']} "
            f"(high確率={row['v3_high_prob']:.3f})"
        )
        print(
            f"  変化量     : {row['prob_change']:+.3f}"
        )
        print(
            f"  絶対差     : {row['abs_prob_diff']:.3f}"
        )

    print("\n" + "=" * 70)

    # 予測クラス自体が変わったもの
    changed_predictions = comparison_df[
        comparison_df["v2_prediction"]
        != comparison_df["v3_prediction"]
    ]

    print("\n予測クラスが変化したデータ:")

    if changed_predictions.empty:
        print("ありません。")
    else:
        print(
            changed_predictions.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    # 履歴として保存
    HISTORY_DIR.mkdir(exist_ok=True)

    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    output_file = (
        HISTORY_DIR
        / f"confidence_comparison_{timestamp}.csv"
    )

    comparison_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\n確信度比較結果を {output_file} に保存しました。"
    )


if __name__ == "__main__":
    main()