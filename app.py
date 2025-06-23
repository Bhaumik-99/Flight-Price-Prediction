from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

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

# Define mappings for categorical variables (adjust based on your model training)
AIRLINE_MAPPING = {
    'Jet Airways': 0,
    'IndiGo': 1,
    'Air India': 2,
    'Multiple carriers': 3,
    'SpiceJet': 4,
    'Vistara': 5,
    'Air Asia': 6,
    'GoAir': 7,
    'Multiple carriers Premium economy': 8,
    'Jet Airways Business': 9,
    'Vistara Premium economy': 10,
    'Trujet': 11
}

SOURCE_MAPPING = {
    'Delhi': 0,
    'Kolkata': 1,
    'Mumbai': 2,
    'Chennai': 3
}

DESTINATION_MAPPING = {
    'Cochin': 0,
    'Delhi': 1,
    'New Delhi': 2,
    'Hyderabad': 3,
    'Kolkata': 4
}

def extract_datetime_features(datetime_str):
    """Extract features from datetime string"""
    try:
        dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')
        return {
            'hour': dt.hour,
            'day': dt.day,
            'month': dt.month,
            'year': dt.year,
            'weekday': dt.weekday(),
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
    except:
        # Default values if parsing fails
        return {
            'hour': 12,
            'day': 15,
            'month': 6,
            'year': 2024,
            'weekday': 0,
            'is_weekend': 0
        }

def calculate_duration_hours(dep_time, arr_time):
    """Calculate flight duration in hours"""
    try:
        dep_dt = datetime.strptime(dep_time, '%Y-%m-%dT%H:%M')
        arr_dt = datetime.strptime(arr_time, '%Y-%m-%dT%H:%M')
        duration = arr_dt - dep_dt
        return duration.total_seconds() / 3600  # Convert to hours
    except:
        return 3.0  # Default 3 hours if calculation fails

def prepare_features(form_data):
    """Prepare features for model prediction"""
    
    # Extract datetime features
    dep_features = extract_datetime_features(form_data['Dep_Time'])
    arr_features = extract_datetime_features(form_data['Arrival_Time'])
    
    # Calculate duration
    duration_hours = calculate_duration_hours(form_data['Dep_Time'], form_data['Arrival_Time'])
    
    # Prepare feature array (adjust based on your model's expected features)
    features = [
        # Airline (encoded)
        AIRLINE_MAPPING.get(form_data['airline'], 0),
        
        # Source (encoded)
        SOURCE_MAPPING.get(form_data['Source'], 0),
        
        # Destination (encoded)
        DESTINATION_MAPPING.get(form_data['Destination'], 0),
        
        # Stops
        int(form_data['stops']),
        
        # Departure features
        dep_features['hour'],
        dep_features['day'],
        dep_features['month'],
        dep_features['weekday'],
        dep_features['is_weekend'],
        
        # Arrival features
        arr_features['hour'],
        arr_features['day'],
        arr_features['month'],
        arr_features['weekday'],
        arr_features['is_weekend'],
        
        # Duration
        duration_hours,
        
        # Additional features that might be in your model
        # Add more features here based on your training data
    ]
    
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    """Render the main form page"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flight Price Prediction</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 800px; margin-top: 50px; }
            .form-container { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }
            .form-control, .form-select { border-radius: 10px; padding: 12px; border: 2px solid #e9ecef; }
            .form-control:focus, .form-select:focus { border-color: #3498db; box-shadow: 0 0 0 0.2rem rgba(52,152,219,0.25); }
            .btn-predict { background: linear-gradient(45deg, #e74c3c, #c0392b); border: none; padding: 15px 40px; border-radius: 50px; color: white; font-weight: 600; }
            .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(231,76,60,0.4); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="form-container">
                <h1><i class="fas fa-plane"></i> Flight Price Predictor</h1>
                <form action="/predict" method="post">
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">Departure Date & Time</label>
                            <input type="datetime-local" name="Dep_Time" class="form-control" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Arrival Date & Time</label>
                            <input type="datetime-local" name="Arrival_Time" class="form-control" required>
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">Source</label>
                            <select name="Source" class="form-select" required>
                                <option value="">Select Source</option>
                                <option value="Delhi">Delhi</option>
                                <option value="Kolkata">Kolkata</option>
                                <option value="Mumbai">Mumbai</option>
                                <option value="Chennai">Chennai</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Destination</label>
                            <select name="Destination" class="form-select" required>
                                <option value="">Select Destination</option>
                                <option value="Cochin">Cochin</option>
                                <option value="Delhi">Delhi</option>
                                <option value="New Delhi">New Delhi</option>
                                <option value="Hyderabad">Hyderabad</option>
                                <option value="Kolkata">Kolkata</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">Number of Stops</label>
                            <select name="stops" class="form-select" required>
                                <option value="">Select Stops</option>
                                <option value="0">Non-Stop</option>
                                <option value="1">1 Stop</option>
                                <option value="2">2 Stops</option>
                                <option value="3">3 Stops</option>
                                <option value="4">4 Stops</option>
                            </select>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Airline</label>
                            <select name="airline" class="form-select" required>
                                <option value="">Select Airline</option>
                                <option value="Jet Airways">Jet Airways</option>
                                <option value="IndiGo">IndiGo</option>
                                <option value="Air India">Air India</option>
                                <option value="Multiple carriers">Multiple carriers</option>
                                <option value="SpiceJet">SpiceJet</option>
                                <option value="Vistara">Vistara</option>
                                <option value="Air Asia">Air Asia</option>
                                <option value="GoAir">GoAir</option>
                                <option value="Multiple carriers Premium economy">Multiple carriers Premium economy</option>
                                <option value="Jet Airways Business">Jet Airways Business</option>
                                <option value="Vistara Premium economy">Vistara Premium economy</option>
                                <option value="Trujet">Trujet</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="text-center">
                        <button type="submit" class="btn btn-predict">
                            <i class="fas fa-search"></i> Predict Price
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check if flight_rf.pkl exists.',
            'prediction': 'Error: Model not available'
        }), 500
    
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate required fields
        required_fields = ['Dep_Time', 'Arrival_Time', 'Source', 'Destination', 'stops', 'airline']
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'prediction': 'Error: Missing data'
                }), 400
        
        # Prepare features for prediction
        features = prepare_features(form_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Format prediction (assuming the model outputs price in rupees)
        predicted_price = round(prediction, 2)
        
        # Create response with flight details
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
        
        # Return prediction page
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Flight Price Prediction Result</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
            <style>
                body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
                .result-container {{ max-width: 800px; margin: 50px auto; background: white; border-radius: 20px; padding: 40px; box-shadow: 0 15px 35px rgba(0,0,0,0.1); }}
                .price-display {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 30px; border-radius: 15px; text-align: center; margin: 20px 0; }}
                .price-amount {{ font-size: 3rem; font-weight: bold; }}
                .detail-item {{ display: flex; justify-content: space-between; padding: 15px 0; border-bottom: 1px solid #eee; }}
                .detail-label {{ font-weight: 600; color: #2c3e50; }}
                .detail-value {{ color: #3498db; font-weight: bold; }}
                .btn-new-search {{ background: linear-gradient(45deg, #3498db, #2980b9); border: none; padding: 15px 30px; border-radius: 50px; color: white; }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="result-container">
                    <h1><i class="fas fa-chart-line"></i> Prediction Result</h1>
                    
                    <div class="price-display">
                        <div>Predicted Flight Price</div>
                        <div class="price-amount">{response_data['prediction']}</div>
                    </div>
                    
                    <div class="flight-details">
                        <h3>Flight Details</h3>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-plane-departure"></i> From</span>
                            <span class="detail-value">{response_data['source']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-plane-arrival"></i> To</span>
                            <span class="detail-value">{response_data['destination']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-calendar"></i> Departure</span>
                            <span class="detail-value">{response_data['departure_time']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-calendar"></i> Arrival</span>
                            <span class="detail-value">{response_data['arrival_time']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-clock"></i> Duration</span>
                            <span class="detail-value">{response_data['duration']}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-route"></i> Stops</span>
                            <span class="detail-value">{"Non-Stop" if response_data['stops'] == '0' else response_data['stops'] + " Stop(s)"}</span>
                        </div>
                        <div class="detail-item">
                            <span class="detail-label"><i class="fas fa-plane"></i> Airline</span>
                            <span class="detail-value">{response_data['airline']}</span>
                        </div>
                    </div>
                    
                    <div class="text-center mt-4">
                        <a href="/" class="btn btn-new-search">
                            <i class="fas fa-search"></i> New Search
                        </a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'prediction': 'Error: Unable to predict'
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON response)"""
    
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
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
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print("Starting Flight Price Prediction App...")
    print("Make sure 'flight_rf.pkl' is in the same directory as this script.")
    app.run(debug=True, host='0.0.0.0', port=5000)