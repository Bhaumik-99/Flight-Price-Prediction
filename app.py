from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the trained Random Forest model
try:
    with open('flight_rf.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: flight_rf.pkl not found. Please ensure the model file is in the same directory.")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Define mappings for categorical variables
AIRLINE_MAPPING = {
    'Jet Airways': 0, 'IndiGo': 1, 'Air India': 2, 'Multiple carriers': 3,
    'SpiceJet': 4, 'Vistara': 5, 'Air Asia': 6, 'GoAir': 7,
    'Multiple carriers Premium economy': 8, 'Jet Airways Business': 9,
    'Vistara Premium economy': 10, 'Trujet': 11
}

SOURCE_MAPPING = {'Delhi': 0, 'Kolkata': 1, 'Mumbai': 2, 'Chennai': 3}

DESTINATION_MAPPING = {'Cochin': 0, 'Delhi': 1, 'New Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4}

def extract_datetime_features(datetime_str):
    """Extract features from datetime string"""
    try:
        dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')
        return {
            'hour': dt.hour, 'day': dt.day, 'month': dt.month, 'year': dt.year,
            'weekday': dt.weekday(), 'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
    except:
        return {'hour': 12, 'day': 15, 'month': 6, 'year': 2024, 'weekday': 0, 'is_weekend': 0}

def calculate_duration_hours(dep_time, arr_time):
    """Calculate flight duration in hours"""
    try:
        dep_dt = datetime.strptime(dep_time, '%Y-%m-%dT%H:%M')
        arr_dt = datetime.strptime(arr_time, '%Y-%m-%dT%H:%M')
        duration = arr_dt - dep_dt
        return duration.total_seconds() / 3600
    except:
        return 3.0

def prepare_features(form_data):
    """Prepare features for model prediction"""
    dep_features = extract_datetime_features(form_data['Dep_Time'])
    arr_features = extract_datetime_features(form_data['Arrival_Time'])
    duration_hours = calculate_duration_hours(form_data['Dep_Time'], form_data['Arrival_Time'])
    
    features = [
        AIRLINE_MAPPING.get(form_data['airline'], 0),
        SOURCE_MAPPING.get(form_data['Source'], 0),
        DESTINATION_MAPPING.get(form_data['Destination'], 0),
        int(form_data['stops']),
        dep_features['hour'], dep_features['day'], dep_features['month'],
        dep_features['weekday'], dep_features['is_weekend'],
        arr_features['hour'], arr_features['day'], arr_features['month'],
        arr_features['weekday'], arr_features['is_weekend'],
        duration_hours
    ]
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    """Render the main form page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None:
        return render_template('predict.html', error="Model not loaded. Please check if flight_rf.pkl exists."), 500
    
    try:
        form_data = request.form.to_dict()
        required_fields = ['Dep_Time', 'Arrival_Time', 'Source', 'Destination', 'stops', 'airline']
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return render_template('templates/predict.html', error=f"Missing required field: {field}"), 400

        features = prepare_features(form_data)
        prediction = model.predict(features)[0]
        predicted_price = round(prediction, 2)
        
        response_data = {
            'prediction': f"₹{predicted_price:,.2f}",
            'source': form_data['Source'],
            'destination': form_data['Destination'],
            'departure_time': form_data['Dep_Time'],
            'arrival_time': form_data['Arrival_Time'],
            'stops': form_data['stops'],
            'airline': form_data['airline'],
            'duration': f"{calculate_duration_hours(form_data['Dep_Time'], form_data['Arrival_Time']):.1f} hours"
        }
        
        return render_template('predict.html', **response_data)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return render_template('error.html', error=f"Prediction failed: {str(e)}"), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    try:
        data = request.json
        features = prepare_features(data)
        prediction = model.predict(features)[0]
        return jsonify({
            'prediction': round(prediction, 2),
            'formatted_prediction': f"₹{prediction:,.2f}",
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    print("Starting Flight Price Prediction App...")
    print("Make sure 'flight_rf.pkl' is in the same directory as this script.")
    app.run(debug=True, host='0.0.0.0', port=5000)
