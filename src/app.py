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
    app.run(debug=True)