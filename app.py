from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

# Load the model
with open("coastal_hybrid_model.pkl", "rb") as f:
    hybrid_model = pickle.load(f)

# Define thresholds and features
thresholds = {
    'temperature': (12, 35),
    'currentspeed': (0, 2),
    'ph': (6.5, 8.5),
    'tideLength': (0, 2)
}

features = ['temperature', 'currentspeed', 'ph', 'tideLength']

app = Flask(__name__)

# Predict safety for a new beach
def predict_new_beach(data):
    new_df = pd.DataFrame([data], columns=features)
    if (
        (new_df['temperature'][0] < thresholds['temperature'][0]) or (new_df['temperature'][0] > thresholds['temperature'][1]) or
        (new_df['currentspeed'][0] < thresholds['currentspeed'][0]) or (new_df['currentspeed'][0] > thresholds['currentspeed'][1]) or
        (new_df['ph'][0] < thresholds['ph'][0]) or (new_df['ph'][0] > thresholds['ph'][1]) or
        (new_df['tideLength'][0] < thresholds['tideLength'][0]) or (new_df['tideLength'][0] > thresholds['tideLength'][1])
    ):
        return "Unsafe (Threshold)"
    return "Safe" if hybrid_model.predict(new_df)[0] == 0 else "Unsafe (Clustering)"

# Evaluate suitability of activities
def evaluate_activities(data):
    row = data
    activity_conditions = {
        'Swimming': (20 <= row['temperature'] <= 30) and (row['currentspeed'] < 2.0) and
                    (6.5 <= row['ph'] <= 8.5) and (row['turbidity'] < 5) and
                    (row['scattering'] < 1.0) and (0.5 <= row['tideLength'] <= 2.0),

        'Scuba Diving': (18 <= row['temperature'] <= 30) and (row['currentspeed'] < 1.0) and
                        (6.5 <= row['ph'] <= 8.5) and (row['turbidity'] < 5) and
                        (row['scattering'] < 1.0) and (0.5 <= row['tideLength'] <= 2.0),

        'Surfing': (15 <= row['temperature'] <= 25) and (row['currentspeed'] < 1.5) and
                   (6.5 <= row['ph'] <= 8.5) and (row['turbidity'] < 10) and
                   (row['scattering'] < 1.0) and (1.0 <= row['tideLength'] <= 3.0),

        'Sunbathing': (20 <= row['temperature'] <= 35),

        'Beach Volleyball': (20 <= row['temperature'] <= 35),

        'Jet Skiing': (20 <= row['temperature'] <= 30) and (row['currentspeed'] < 1.5) and
                      (6.5 <= row['ph'] <= 8.5) and (row['turbidity'] < 10) and
                      (row['scattering'] < 1.0) and (0.5 <= row['tideLength'] <= 2.0)
    }
    return {activity: int(condition) for activity, condition in activity_conditions.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract input data
        user_input = {
            'temperature': data['temperature'],
            'currentspeed': data['currentspeed'],
            'ph': data['ph'],
            'scattering': data['scattering'],
            'tideLength': data['tideLength'],
            'turbidity': data['turbidity']
        }

        # Get safety prediction
        safety_prediction = predict_new_beach(user_input)

        # Get activity suitability
        activity_results = evaluate_activities(user_input)

        # Prepare response
        response = {
            'temperature': user_input['temperature'],
            'currentspeed': user_input['currentspeed'],
            'ph': user_input['ph'],
            'scattering': user_input['scattering'],
            'tideLength': user_input['tideLength'],
            'turbidity': user_input['turbidity'],
            'safety_prediction': 1 if safety_prediction == "Safe" else 0,
            **activity_results
        }

        return jsonify(response)

    except KeyError as e:
        return jsonify({"error": f"Missing parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
