from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the hybrid model
with open("coastal_hybrid_model.pkl", "rb") as f:
    hybrid_model = pickle.load(f)

# Hardcoded threshold limits
thresholds = {
    'temperature': (12, 35),
    'currentspeed': (0, 2),
    'ph': (6.5, 8.5),
    'tideLength': (0, 2)
}

# Features used for the ML model (only these go into the model)
model_features = ['temperature', 'currentspeed', 'ph', 'tideLength']

# Create Flask app
app = Flask(_name_)


# Function to apply threshold checks
def is_within_thresholds(data):
    for feature in model_features:
        value = float(data[feature])
        min_val, max_val = thresholds[feature]
        if not (min_val <= value <= max_val):
            return False
    return True


# Function to predict safety
def predict_new_beach(data):
    # Threshold-based rule-out
    if not is_within_thresholds(data):
        return "Unsafe (Threshold)"

    # Prepare input only with model features
    input_df = pd.DataFrame([{feature: float(data[feature]) for feature in model_features}])
    prediction = hybrid_model.predict(input_df)[0]
    return "Safe" if prediction == 0 else "Unsafe (Clustering)"


# Function to check activity suitability
def evaluate_activities(data):
    temp = float(data['temperature'])
    speed = float(data['currentspeed'])
    ph = float(data['ph'])
    tide = float(data['tideLength'])
    turbidity = float(data['turbidity'])
    scattering = float(data['scattering'])

    activity_conditions = {
        'Swimming': (20 <= temp <= 30) and (speed < 2.0) and
                    (6.5 <= ph <= 8.5) and (turbidity < 5) and
                    (scattering < 1.0) and (0.5 <= tide <= 2.0),

        'Scuba Diving': (18 <= temp <= 30) and (speed < 1.0) and
                        (6.5 <= ph <= 8.5) and (turbidity < 5) and
                        (scattering < 1.0) and (0.5 <= tide <= 2.0),

        'Surfing': (15 <= temp <= 25) and (speed < 1.5) and
                   (6.5 <= ph <= 8.5) and (turbidity < 10) and
                   (scattering < 1.0) and (1.0 <= tide <= 3.0),

        'Sunbathing': (20 <= temp <= 35),

        'Beach Volleyball': (20 <= temp <= 35),

        'Jet Skiing': (20 <= temp <= 30) and (speed < 1.5) and
                      (6.5 <= ph <= 8.5) and (turbidity < 10) and
                      (scattering < 1.0) and (0.5 <= tide <= 2.0)
    }

    return {activity: int(condition) for activity, condition in activity_conditions.items()}


# Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Ensure all required fields are present
        required_fields = model_features + ['turbidity', 'scattering']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing parameter: {field}"}), 400

        # Predict safety
        safety_status = predict_new_beach(data)
        safety_prediction = 1 if safety_status == "Safe" else 0

        # Evaluate recreational activity suitability
        activity_suitability = evaluate_activities(data)

        # Final response
        response = {
            **{k: data[k] for k in required_fields},
            "safety_prediction": safety_prediction,
            "safety_reason": safety_status,
            **activity_suitability
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
