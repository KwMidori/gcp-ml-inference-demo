from pathlib import Path

import joblib


MODELS = {
    "v1": Path("model.joblib"),
    "v2": Path("model_v2.joblib"),
    "v3": Path("model_v3.joblib"),
}


def test_model_files_exist():
    """すべてのモデルファイルが存在することを確認する。"""

    for version, model_path in MODELS.items():
        assert model_path.exists(), (
            f"{version} のモデルファイルがありません: "
            f"{model_path}"
        )


def test_models_can_be_loaded():
    """すべてのモデルをjoblibで読み込めることを確認する。"""

    for version, model_path in MODELS.items():
        model = joblib.load(model_path)

        assert model is not None, (
            f"{version} のモデルを読み込めませんでした。"
        )


def test_models_have_expected_classes():
    """モデルがhigh/normalの2クラスを持つことを確認する。"""

    expected_classes = {"high", "normal"}

    for version, model_path in MODELS.items():
        model = joblib.load(model_path)

        actual_classes = set(model.classes_)

        assert actual_classes == expected_classes, (
            f"{version} のクラスが想定と異なります: "
            f"{actual_classes}"
        )


def test_models_can_predict():
    """各モデルがテキストを予測できることを確認する。"""

    sample_texts = [
        "ログイン方法を教えてください",
        "一部の利用者がログインできません",
    ]

    for version, model_path in MODELS.items():
        model = joblib.load(model_path)

        predictions = model.predict(sample_texts)

        assert len(predictions) == len(sample_texts), (
            f"{version} の予測件数が想定と異なります。"
        )

        for prediction in predictions:
            assert prediction in {"high", "normal"}


def test_models_support_predict_proba():
    """各モデルが予測確率を返せることを確認する。"""

    sample_texts = [
        "ログイン方法を教えてください",
        "一部の利用者がログインできません",
    ]

    for version, model_path in MODELS.items():
        model = joblib.load(model_path)

        probabilities = model.predict_proba(sample_texts)

        assert probabilities.shape == (2, 2), (
            f"{version} の予測確率の形が想定と異なります: "
            f"{probabilities.shape}"
        )

        for row in probabilities:
            assert abs(row.sum() - 1.0) < 1e-6