import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


df = pd.read_csv("data/training_data_100.csv")

texts = df["text"]
labels = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=42,
    stratify=labels,
)


model = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight={
                    "high": 1.5,
                    "normal": 1.0,
                }
            ),
        ),
    ]
)


model.fit(X_train, y_train)


predictions = model.predict(X_test)


accuracy = accuracy_score(y_test, predictions)

print(f"accuracy: {accuracy:.2f}")


print("\nclassification report:")
print(classification_report(y_test, predictions))


print("\nconfusion matrix:")
print(confusion_matrix(y_test, predictions))


results = pd.DataFrame(
    {
        "text": X_test,
        "actual": y_test,
        "predicted": predictions,
    }
)

errors = results[results["actual"] != results["predicted"]]

print("\n誤判定したデータ:")
print(errors)


vectorizer = model.named_steps["vectorizer"]
classifier = model.named_steps["classifier"]

print("\nclasses:")
print(classifier.classes_)


feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

feature_weights = pd.DataFrame(
    {
        "feature": feature_names,
        "weight": coefficients,
    }
)


print("\n係数が小さい特徴:")
print(
    feature_weights
    .sort_values("weight")
    .head(20)
)


print("\n係数が大きい特徴:")
print(
    feature_weights
    .sort_values("weight", ascending=False)
    .head(20)
)


def explain_prediction(text):
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    x = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    tfidf_values = x.toarray()[0]
    contributions = tfidf_values * coefficients

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "tfidf": tfidf_values,
            "coefficient": coefficients,
            "contribution": contributions,
        }
    )

    explanation = explanation[
        explanation["tfidf"] > 0
    ].sort_values(
        "contribution",
        key=abs,
        ascending=False,
    )

    prediction = model.predict([text])[0]
    decision_score = classifier.decision_function(x)[0]
    intercept = classifier.intercept_[0]

    contribution_sum = contributions.sum()

    print(f"\n判定を分析する文章: {text}")
    print(f"予測: {prediction}")

    print("\nintercept:")
    print(intercept)

    print("\n特徴量の寄与合計:")
    print(contribution_sum)

    print("\ndecision score:")
    print(decision_score)

    print("\nintercept + 特徴量の寄与合計:")
    print(intercept + contribution_sum)

    print("\n各特徴量の寄与:")
    print(
        explanation[
            [
                "feature",
                "tfidf",
                "coefficient",
                "contribution",
            ]
        ].head(30)
    )


explain_prediction("領収書を発行できますか")


joblib.dump(model, "model.joblib")

print("\nモデルを model.joblib に保存しました")