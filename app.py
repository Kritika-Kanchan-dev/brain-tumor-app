from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("brain_tumor_model.h5")
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = image.load_img(file, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    label = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    return jsonify({"prediction": label, "confidence": round(confidence * 100, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
