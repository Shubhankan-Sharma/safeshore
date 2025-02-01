from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your trained KMeans model
with open('kmeans_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

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
    
    # Assuming the prediction returns a string directly
    safety_status = str(prediction[0])  # Convert to string if necessary

    # Return a detailed response
    response = {
        'safety_status': safety_status,
        'input_data': input_data,
        'message': f'The beach safety status for the provided input is: {safety_status}.'
    }

    return jsonify(response)

# Run the application with Gunicorn in production
if __name__ == '__main__':
    app.run()  # This line can be omitted when using Gunicorn in production
