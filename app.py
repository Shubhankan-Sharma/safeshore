from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained models
with open('logistic_regression_model_safety.pkl', 'rb') as model_file:
    model_safety = pickle.load(model_file)

with open('logistic_regression_model_swimming.pkl', 'rb') as model_file:
    model_swimming = pickle.load(model_file)

with open('logistic_regression_model_scuba_diving.pkl', 'rb') as model_file:
    model_scuba_diving = pickle.load(model_file)

with open('logistic_regression_model_surfing.pkl', 'rb') as model_file:
    model_surfing = pickle.load(model_file)

with open('logistic_regression_model_sunbathing.pkl', 'rb') as model_file:
    model_sunbathing = pickle.load(model_file)

with open('logistic_regression_model_beach_volleyball.pkl', 'rb') as model_file:
    model_beach_volleyball = pickle.load(model_file)

with open('logistic_regression_model_jet_skiing.pkl', 'rb') as model_file:
    model_jet_skiing = pickle.load(model_file)

# Load your trained scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define mapping from binary output to safety status and activity suitability
safety_status_mapping = {
    0: "Unsafe",
    1: "Safe"
}

activity_mapping = {
    0: "Not Suitable",
    1: "Suitable"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data.get('input', [])

    # Validate input data
    try:
        input_data = [float(i) for i in input_data]  # Convert to float
        if len(input_data) != 6:  # Ensure correct number of features
            return jsonify({'error': 'Input must contain exactly 6 numeric values.'}), 400
    except ValueError:
        return jsonify({'error': 'Input values must be numeric.'}), 400

    # Scale the input data
    scaled_input_data = scaler.transform([input_data])

    # Make predictions for safety and activities
    safety_prediction = model_safety.predict(scaled_input_data)
    swimming_prediction = model_swimming.predict(scaled_input_data)
    scuba_diving_prediction = model_scuba_diving.predict(scaled_input_data)
    surfing_prediction = model_surfing.predict(scaled_input_data)
    sunbathing_prediction = model_sunbathing.predict(scaled_input_data)
    beach_volleyball_prediction = model_beach_volleyball.predict(scaled_input_data)
    jet_skiing_prediction = model_jet_skiing.predict(scaled_input_data)

    # Map predictions to safety status and activities
    safety_status = safety_status_mapping.get(safety_prediction[0], "Unknown")
    activities = {
        "Swimming": activity_mapping.get(swimming_prediction[0], "Unknown"),
        "Scuba Diving": activity_mapping.get(scuba_diving_prediction[0], "Unknown"),
        "Surfing": activity_mapping.get(surfing_prediction[0], "Unknown"),
        "Sunbathing": activity_mapping.get(sunbathing_prediction[0], "Unknown"),
                "Beach Volleyball": activity_mapping.get(beach_volleyball_prediction[0], "Unknown"),
        "Jet Skiing": activity_mapping.get(jet_skiing_prediction[0], "Unknown")
    }

    return jsonify({
        'safety_status': safety_status,
        'activities': activities
    })

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
