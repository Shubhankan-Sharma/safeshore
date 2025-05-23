from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("coastal_rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Define thresholds
thresholds = {
    'temperature': (12, 35),
    'currentspeed': (0, 2),
    'ph': (6.5, 8.5),
    'tideLength': (0, 2)
}

features = ['temperature', 'currentspeed', 'ph', 'tideLength']


def predict_new_beach(data):
    new_df = pd.DataFrame([data], columns=features)

    # First, threshold check
    if (
            (new_df['temperature'][0] < thresholds['temperature'][0]) or (
            new_df['temperature'][0] > thresholds['temperature'][1]) or
            (new_df['currentspeed'][0] < thresholds['currentspeed'][0]) or (
            new_df['currentspeed'][0] > thresholds['currentspeed'][1]) or
            (new_df['ph'][0] < thresholds['ph'][0]) or (new_df['ph'][0] > thresholds['ph'][1]) or
            (new_df['tideLength'][0] < thresholds['tideLength'][0]) or (
            new_df['tideLength'][0] > thresholds['tideLength'][1])
    ):
        return "Unsafe (Threshold)"

    # Otherwise, predict
    return "Safe" if rf_model.predict(new_df)[0] == 0 else "Unsafe (Clustering)"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not all(key in data for key in features):
        return jsonify({'error': 'Invalid input. Please provide temperature, currentspeed, ph, and tideLength.'}), 400

    prediction = predict_new_beach(data)
    return jsonify({'prediction': prediction})


if __name__ == "__main__":
    app.run(debug=true)
