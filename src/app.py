import os
import joblib

from flask import Flask, request, jsonify


app = Flask(__name__)

model = joblib.load("model.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data["text"]

    prediction = model.predict([text])[0]

    return jsonify(
        {
            "text": text,
            "prediction": prediction,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )