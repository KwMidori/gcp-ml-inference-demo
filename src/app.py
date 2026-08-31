import os
import tempfile
import joblib

from flask import Flask, request, jsonify
from google.cloud import storage


app = Flask(__name__)

BUCKET_NAME = "gcp-ml-inference-demo-eh01-models"
MODEL_NAME = "model.joblib"

LOCAL_MODEL_PATH = os.path.join(
    tempfile.gettempdir(),
    "model.joblib",
)


def download_model():
    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_NAME)

    blob.download_to_filename(LOCAL_MODEL_PATH)

    print("Model downloaded from Cloud Storage.")


download_model()

model = joblib.load(LOCAL_MODEL_PATH)


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