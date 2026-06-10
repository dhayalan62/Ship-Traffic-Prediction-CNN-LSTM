import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model("ship_model.h5", compile=False)

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/search")
def search():
    return render_template("search.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/predict_ai")
def predict_ai():

    sample = np.array([
        [
            [13.08, 80.27, 15, 120],
            [13.09, 80.28, 16, 121],
            [13.10, 80.29, 17, 122],
            [13.11, 80.30, 18, 123],
            [13.12, 80.31, 19, 124],
            [13.13, 80.32, 20, 125],
            [13.14, 80.33, 21, 126],
            [13.15, 80.34, 22, 127],
            [13.16, 80.35, 23, 128],
            [13.17, 80.36, 24, 129]
        ]
    ])

    prediction = model.predict(sample)

    return str(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)