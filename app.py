from flask import Flask, request, jsonify
from werkzeug.utils import quote
import pickle

app = Flask(__name__)

# Load your trained KMeans model
with open('kmeans_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define mapping from cluster labels to safety status
safety_status_mapping = {
    0: "Safe",
    1: "Caution",
    2: "Danger"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data.get('input', [])
    
    # Validate input data
    try:
        input_data = [float(i) for i in input_data]  # Convert to float
    except ValueError:
        return jsonify({'error': 'Input values must be numeric.'}), 400

    # Make prediction
    prediction = model.predict([input_data])
    
    # Map prediction to safety status
    safety_status = safety_status_mapping.get(prediction[0], "Unknown")

    return jsonify({'safety_status': safety_status})

# Run the application with Gunicorn in production
if __name__ == '__main__':
    app.run()  # This line can be omitted when using Gunicorn in production
