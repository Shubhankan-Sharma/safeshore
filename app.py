from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained KMeans model
with open('kmeans_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your trained scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define mapping from cluster labels to safety status
#     safety_status_mapping = {
# 0: "Safe",
# 1: "Caution",
# 2: "Danger"}

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

    # Make prediction
    prediction = model.predict(scaled_input_data)
    
    # Map prediction to safety status
    # safety_status = safety_status_mapping.get(prediction[0], "Unknown")

    return jsonify({'safety_status': prediction})

# Run the application with Gunicorn in production
if __name__ == '__main__':
    app.run()  # This line can be omitted when using Gunicorn in production
