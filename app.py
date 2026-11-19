"""
Example Flask backend to connect your trained ML model to the MindGuard UI.

Run:
    pip install flask flask-cors joblib pandas
    python app.py
"""
import flask
import joblib
import pandas as pd

app = flask.Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Load your trained model (change the file name to yours)
model = joblib.load("model.pkl")

# Column order MUST match what your model was trained on
FEATURES = [
    "age",
    "gender",
    "screen",
    "social",
    "gaming",
    "work",
    "notifications",
    "sleep",
    "stress",
    "opens",
    "impact",
    "weekend",
]

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
LEVEL_MAP = {"Low": 0, "Moderate": 1, "High": 2}


@app.route("/predict", methods=["POST"])
def predict():
    data = flask.request.get_json()

    data["gender"] = GENDER_MAP.get(data["gender"], 0)
    data["stress"] = LEVEL_MAP.get(data["stress"], 1)
    data["impact"] = LEVEL_MAP.get(data["impact"], 1)

    row = pd.DataFrame([data])[FEATURES]

    label = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    if probability >= 0.65:
        risk = "High"
    elif probability >= 0.35:
        risk = "Moderate"
    else:
        risk = "Low"

    return flask.jsonify({"label": label, "risk": risk, "probability": probability})


if __name__ == "__main__":
    app.run(debug=True)
