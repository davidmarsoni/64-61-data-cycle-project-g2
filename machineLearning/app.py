from flask import Flask, request, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import pickle
import pandas as pd
import numpy as np
from io import StringIO
import os
import json

app = Flask(__name__)

# Swagger configuration 
SWAGGER_URL = '/api/docs'  
API_URL = '/static/swagger.json'  

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Solar Production Prediction API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)



@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Load model
model_path = 'models/pred_only_model_pac_2025-03-25_13-21-18.pkl'  

try:
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    print(f"Loaded model from {model_path}")
    print(f"Model features: {model_info['features']}")
except Exception as e:
    print(f"Error loading model: {e}")
    model_info = None

@app.route('/predict', methods=['POST'])
def predict():
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if the request includes a file
    if 'file' in request.files:
        file = request.files['file']
        
        # Read the CSV file
        try:
            data = pd.read_csv(file)
            data = process_prediction_data(data)
            print(f"CSV data loaded, shape: {data.shape}")
        except Exception as e:
            return jsonify({'error': f'Error reading CSV: {str(e)}'}), 400
    
    # Check if JSON data is provided
    elif request.json:
        try:
            # If JSON data is an array of records
            if isinstance(request.json, list):
                data = pd.DataFrame(request.json)
            # If JSON data is a single record
            else:
                data = pd.DataFrame([request.json])
            print(f"JSON data loaded, shape: {data.shape}")
        except Exception as e:
            return jsonify({'error': f'Error parsing JSON: {str(e)}'}), 400
    
    else:
        return jsonify({'error': 'No data provided. Send either a CSV file or JSON data.'}), 400
    
    # Process the data for prediction
    try:
        # Standardize column names (lowercase)
        data.columns = [col.lower() for col in data.columns]
        
        # Convert date and time columns to datetime if they exist
        if 'date' in data.columns and 'time' in data.columns:
            data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
            data = data.set_index('datetime')
        elif 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data.set_index('datetime')
        
        # Add time-based features
        data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
        data['day_of_year'] = data.index.dayofyear / 365.0
        
        # Ensure all required features are available
        for feature in model_info['features']:
            if feature not in data.columns and feature not in ['hour_sin', 'hour_cos', 'day_of_year']:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400
        
        # Select only the features used by the model
        X_pred = data[model_info['features']].copy()
        
        # Scale features
        X_pred_scaled = model_info['scaler'].transform(X_pred)
        
        # Make predictions
        predictions = model_info['model'].predict(X_pred_scaled)
        
        # Format results
        data['predicted_pv_output'] = predictions
        data['predicted_pv_output'] = data['predicted_pv_output'].clip(lower=0)  # Ensure non-negative
        
        # Convert index to string for JSON serialization
        result_df = data.reset_index()
        result_df['datetime'] = result_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Return only datetime and predictions
        result_df = result_df[['datetime', 'predicted_pv_output']]
        
        return jsonify({
            'predictions': result_df.to_dict(orient='records'),
            'model_info': {
                'training_date': model_info.get('training_date', 'unknown'),
                'model_type': model_info.get('model_type', 'unknown'),
                'metrics': model_info.get('metrics', {})
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    
def process_prediction_data(data):
    """
    Process prediction data with multiple entries for the same datetime
    by selecting the most recent prediction (highest horizon/counter) for each timestamp.
    
    Parameters:
        data (DataFrame): Raw prediction data with potentially multiple entries per datetime
        
    Returns:
        DataFrame: Processed data with one entry per datetime
    """
    # Standardize column names to lowercase for case-insensitive comparison
    data.columns = [col.lower() for col in data.columns]
    
    # Check if 'prediction' column exists
    if 'prediction' in data.columns:
        prediction_col = 'prediction'
    else:
        # If there's no Prediction column, we'll return the data as is
        print("Warning: No prediction/horizon column found. Using data as is.")
        return data
    
    # Group by date and time
    if 'date' in data.columns and 'time' in data.columns:
        # Create a temporary ID for grouping
        data['temp_id'] = data['date'] + ' ' + data['time']
        
        # Group by this ID
        grouped = data.groupby('temp_id')
        
        # For each group, get the row with the HIGHEST prediction value (most recent)
        selected_predictions = []
        for name, group in grouped:
            # Use ascending=False to get the highest value first
            best_pred = group.sort_values(prediction_col, ascending=False).iloc[0]
            selected_predictions.append(best_pred)
        
        # Create a new dataframe with the selected predictions
        processed_data = pd.DataFrame(selected_predictions)
        
        # Remove the temporary ID column
        if 'temp_id' in processed_data.columns:
            processed_data = processed_data.drop('temp_id', axis=1)
            
        print(f"Processed prediction data: Reduced from {len(data)} to {len(processed_data)} records")
        
        return processed_data
    else:
        # If there are no Date and Time columns, we can't group properly
        error_msg = f"No date and time columns found. Available columns: {data.columns.tolist()}"
        print(f"Warning: {error_msg}")
        raise ValueError(error_msg)

@app.route('/model-info', methods=['GET'])
def get_model_info():
    if model_info is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'features': model_info['features'],
        'training_date': model_info.get('training_date', 'unknown'),
        'model_type': model_info.get('model_type', 'unknown'),
        'metrics': model_info.get('metrics', {})
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)