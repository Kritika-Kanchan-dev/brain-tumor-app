from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    prediction = model.predict(np.array([data]))
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
