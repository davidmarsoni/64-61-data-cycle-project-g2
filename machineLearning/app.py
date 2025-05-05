from flask import Flask, request, jsonify, send_from_directory
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from joblib import load
import pickle

# Import the model components
from EnergyConsumption import load_room_allocation_data_hourly

app = Flask(__name__)

# Swagger configuration 
SWAGGER_URL = '/api/docs'  
API_URL = '/static/swagger.json'  

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Energy Consumption Prediction API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

# Load consumption model
try:
    model_info_cons = load("models/consumption_model.joblib")
    
    # Recréer la fonction de prétraitement
    object_columns = model_info_cons.get('object_columns', [])
    
    def preprocess_data(X):
        X_processed = X.copy()
        for col in object_columns:
            if col in X_processed.columns:
                if X_processed[col].nunique() <= 1:
                    X_processed[col] = 0
                else:
                    X_processed[col] = X_processed[col].astype('category').cat.codes
        return X_processed
    
    # Ajouter la fonction au dictionnaire model_info
    model_info_cons['preprocess_func'] = preprocess_data
    
    print("Consumption model loaded successfully!")
except Exception as e:
    print("Failed to load consumption model:", e)
    model_info_cons = None

try:
    model_info_prod = pickle.load(open("models/prediction_model.pkl", "rb"))
    print("Production model loaded successfully!")
except Exception as e:
    print("Failed to load production model:", e)
    model_info_prod = None


def find_latest_room_allocation_file(base_dir=None):
    """
    Recherche le fichier d'allocation de salles le plus récent dans les dossiers cleaned_data_*
    
    Parameters:
        base_dir (str): Répertoire de base (par défaut: C:/DataCollection)
        
    Returns:
        tuple: (chemin du répertoire contenant le fichier, liste des fichiers trouvés)
    """
    import os
    import glob
    import re
    from datetime import datetime

    # Répertoire de base
    if base_dir is None:
        base_dir = os.getenv('BASE_DIR', 'C:/DataCollection')
    
    # Trouver tous les dossiers cleaned_data_*
    data_dirs = glob.glob(os.path.join(base_dir, 'cleaned_data_*'))
    
    # Trier par date, du plus récent au plus ancien
    def extract_date(path):
        match = re.search(r'cleaned_data_(\d{4}-\d{2}-\d{2})', path)
        if match:
            return datetime.strptime(match.group(1), '%Y-%m-%d')
        return datetime.min
    
    data_dirs.sort(key=extract_date, reverse=True)
    
    # Parcourir les dossiers et chercher des fichiers d'allocation de salles
    for data_dir in data_dirs:
        booking_dir = os.path.join(data_dir, 'BellevueBooking')
        if os.path.isdir(booking_dir):
            room_files = [f for f in os.listdir(booking_dir) 
                         if f.endswith('.csv') and 'RoomAllocations' in f]
            
            if room_files:
                print(f"Found room allocation files in {booking_dir}: {room_files}")
                return booking_dir, room_files
    
    print("No room allocation files found in any cleaned_data_* directory")
    return None, []

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

@app.route('/predictConsumption', methods=['GET'])
def predict_consumption():
    """
    Generates daily consumption predictions for the next 3 days,
    and returns a CSV with:
      - date (DD.MM.YYYY)
      - consumptionPrediction (float)
    """
    if model_info_cons is None:
        return jsonify({'error': 'Consumption model not loaded'}), 500

    try:
        # 1. Determine the prediction window (tomorrow + 2 more days)
        tomorrow = datetime.now().date() + timedelta(days=1)
        end_date = tomorrow + timedelta(days=2)  # inclusive, 3 days total

        # 2. Locate the latest weather forecast file
        meteo_dir = os.getenv(
            'METEO_DIR',
            f"C:/DataCollection/cleaned_data_{datetime.now():%Y-%m-%d}/Meteo"
        )
        if not os.path.isdir(meteo_dir):
            return jsonify({'error': f'Meteo directory not found: {meteo_dir}'}), 404

        # Find all Pred_*.csv and pick the most recent
        pred_files = [
            f for f in os.listdir(meteo_dir)
            if f.startswith('Pred_') and f.endswith('.csv')
        ]
        
        if not pred_files:
            return jsonify({'error': 'No prediction files found in meteo directory'}), 404
            
        pred_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(meteo_dir, f)),
            reverse=True
        )
        latest_pred = pred_files[0]

        # Load and process the weather predictions
        raw_pred = pd.read_csv(os.path.join(meteo_dir, latest_pred))
        weather_df = process_prediction_data(raw_pred)
        
        # Convert date and time to datetime
        weather_df['datetime'] = pd.to_datetime(weather_df['date'] + ' ' + weather_df['time'])
        weather_df.set_index('datetime', inplace=True)

        # 3. Load room occupancy data
        room_dir_default = os.getenv(
            'ROOM_DIR',
            f"C:/DataCollection/cleaned_data_{datetime.now():%Y-%m-%d}/BellevueBooking"
        )

        # Utiliser la nouvelle fonction pour trouver le fichier d'allocation le plus récent
        latest_room_dir, room_files = find_latest_room_allocation_file()

        if latest_room_dir:
            print(f"Using room data from: {latest_room_dir}")
            room_df = load_room_allocation_data_hourly(latest_room_dir)
        else:
            print("No room allocation data found, using default directory")
            # Essayer avec le répertoire par défaut, mais préparer un fallback
            try:
                room_df = load_room_allocation_data_hourly(room_dir_default)
            except Exception as e:
                print(f"Error loading room data: {e}")
                print("Using empty DataFrame for room occupancy")
                room_df = pd.DataFrame()
        
        # 4. Create the prediction DataFrame with proper date range
        date_range = pd.date_range(
            start=tomorrow.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d 23:00:00'),
            freq='h'
        )
        prediction_df = pd.DataFrame(index=date_range)
        prediction_df.index.name = 'datetime'
        
        # 5. Merge weather data
        if not weather_df.empty:
            prediction_df = pd.merge(
                prediction_df,
                weather_df,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_weather')
            )
            
            # Interpolate missing weather values
            for col in weather_df.columns:
                if col in prediction_df.columns and prediction_df[col].isna().any():
                    prediction_df[col] = prediction_df[col].interpolate(method='time').bfill().ffill()
        
        # 6. Merge room occupancy data
        if not room_df.empty:
            prediction_df = pd.merge(
                prediction_df,
                room_df,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_room')
            )
            
            # Fill missing room occupancy values
            for col in room_df.columns:
                if col in prediction_df.columns and prediction_df[col].isna().any():
                    if 'occupied' in col or 'occupation' in col or 'rooms' in col:
                        prediction_df[col] = prediction_df[col].fillna(0)
                    else:
                        prediction_df[col] = prediction_df[col].interpolate(method='time').bfill().ffill()
        
        # 7. Add time-based features
        prediction_df['hour'] = prediction_df.index.hour
        prediction_df['day'] = prediction_df.index.day
        prediction_df['month'] = prediction_df.index.month
        prediction_df['day_of_week'] = prediction_df.index.dayofweek
        prediction_df['week_of_year'] = prediction_df.index.isocalendar().week
        prediction_df['day_of_year'] = prediction_df.index.dayofyear
        
        prediction_df['is_day_hours'] = ((prediction_df['hour'] >= 8) & (prediction_df['hour'] < 17)).astype(int)
        prediction_df['is_evening_hours'] = ((prediction_df['hour'] >= 17) & (prediction_df['hour'] < 22)).astype(int)
        prediction_df['is_night_hours'] = ((prediction_df['hour'] < 8) | (prediction_df['hour'] >= 22)).astype(int)
        
        prediction_df['is_weekend'] = (prediction_df['day_of_week'] >= 5).astype(int)
        prediction_df['is_working_day'] = (~prediction_df.index.dayofweek.isin([5, 6])).astype(int)
        
        prediction_df['hour_sin'] = np.sin(2 * np.pi * prediction_df['hour'] / 24)
        prediction_df['hour_cos'] = np.cos(2 * np.pi * prediction_df['hour'] / 24)
        
        prediction_df['day_of_week_sin'] = np.sin(2 * np.pi * prediction_df['day_of_week'] / 7)
        prediction_df['day_of_week_cos'] = np.cos(2 * np.pi * prediction_df['day_of_week'] / 7)
        
        prediction_df['month_sin'] = np.sin(2 * np.pi * prediction_df['month'] / 12)
        prediction_df['month_cos'] = np.cos(2 * np.pi * prediction_df['month'] / 12)
        
        prediction_df['is_building_open'] = ((prediction_df['hour'] >= 7) & 
                                            (prediction_df['hour'] < 22) & 
                                            (prediction_df['is_working_day'] == 1)).astype(int)
        prediction_df['is_building_closed'] = 1 - prediction_df['is_building_open']
        
        # 8. Add other derived features
        if 'temperature' in prediction_df.columns:
            prediction_df['is_cold_and_occupied'] = ((prediction_df['temperature'] < 10) & 
                                                    (prediction_df['is_building_open'] == 1)).astype(int)
        
        if 'rooms_occupied' in prediction_df.columns and 'temperature' in prediction_df.columns:
            prediction_df['temp_occupation_interaction'] = prediction_df['temperature'] * prediction_df['rooms_occupied']
        
        # 9. Convert object columns to numeric codes as needed
        object_columns = model_info_cons.get('object_columns', [])
        for col in object_columns:
            if col in prediction_df.columns:
                if prediction_df[col].nunique() <= 1:
                    prediction_df[col] = 0
                else:
                    prediction_df[col] = prediction_df[col].astype('category').cat.codes
        
        # 10. Make predictions
        pipeline = model_info_cons['model']
        model_features = pipeline.feature_names_in_
        
        # Ensure all required features exist
        missing_features = [feat for feat in model_features if feat not in prediction_df.columns]
        if missing_features:
            print(f"Adding missing features: {missing_features}")
            for feat in missing_features:
                prediction_df[feat] = 0
        
        # Select only the features used by the model
        X_pred = prediction_df[model_features].copy()
        
        # Make hourly predictions
        hourly_predictions = pipeline.predict(X_pred)
        
        # 11. Create predictions dataframe
        hourly_results = pd.DataFrame({
            'predicted_consumption': hourly_predictions
        }, index=prediction_df.index)
        
        # 12. Aggregate to daily predictions
        daily_results = hourly_results.resample('D').sum()
        daily_results = daily_results.reset_index()
        daily_results = daily_results.rename(columns={'datetime': 'date'})
        
        # 13. Format output
        df_out = daily_results.copy()
        df_out['date'] = pd.to_datetime(df_out['date']).dt.strftime('%d.%m.%Y')
        df_out.rename(
            columns={'predicted_consumption': 'consumptionPrediction'},
            inplace=True
        )
        df_final = df_out[['date', 'consumptionPrediction']]


        return jsonify({
            'status': 'success',
            'predictions': df_final.to_dict(orient='records')
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/predictProduction', methods=['GET'])
def predict_Production():
    """
    Generates predictions for the next 3 days using the latest weather prediction file available.
    Returns hourly predictions with simplified CSV format:
    - date: DD.MM.YYYY format
    - hour: HH:MM format
    - productionPrediction: decimal value
    """
    if model_info_prod is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # 1. Find the most recent weather prediction file
        meteo_dir = os.getenv('METEO_DIR', 'C:/DataCollection/cleaned_data_' + datetime.now().strftime('%Y-%m-%d') + '/Meteo')
        
        if not os.path.exists(meteo_dir):
            return jsonify({'error': f'Meteo directory not found: {meteo_dir}'}), 404
        
        # Look for all Pred_*.csv files and take the most recent one
        pred_files = [f for f in os.listdir(meteo_dir) if f.startswith('Pred_') and f.endswith('.csv')]
        
        if not pred_files:
            return jsonify({'error': 'No prediction files found in meteo directory'}), 404
        
        # Sort by modification time (most recent first)
        pred_files.sort(key=lambda f: os.path.getmtime(os.path.join(meteo_dir, f)), reverse=True)
        latest_pred_file = pred_files[0]
        
        print(f"Using most recent prediction file: {latest_pred_file} (last modified: {datetime.fromtimestamp(os.path.getmtime(os.path.join(meteo_dir, latest_pred_file)))})")
        
        # 2. Load the weather prediction file
        pred_path = os.path.join(meteo_dir, latest_pred_file)
        pred_df = pd.read_csv(pred_path)
        print(f"Loaded {len(pred_df)} rows from {latest_pred_file}")
        
        # Standardize column names
        pred_df.columns = [col.lower() for col in pred_df.columns]
        
        # 3. Prepare dates for prediction
        tomorrow = datetime.now().date() + timedelta(days=1)
        num_days = 3  # Predict for the next 3 days
        
        print(f"Target prediction window: {tomorrow} to {tomorrow + timedelta(days=num_days-1)}")
        
        # Create a dataframe with predictions for each hour of the next 3 days
        # Start at midnight tomorrow and include 24 hours * 3 days = 72 hours
        hourly_times = pd.date_range(
            start=datetime.combine(tomorrow, datetime.min.time()),
            periods=24 * num_days,
            freq='H'
        )
        
        # Create empty dataframe with hourly timestamps
        hourly_df = pd.DataFrame(index=hourly_times)
        
        # If we have a prediction file, use it to extract the prediction data
        if 'date' in pred_df.columns and 'time' in pred_df.columns:
            # Create datetime field in the prediction data
            pred_df['datetime'] = pd.to_datetime(pred_df['date'] + ' ' + pred_df['time'], errors='coerce')
            pred_df = pred_df.dropna(subset=['datetime'])
            
            # Check for prediction column - this will have the future hours offset
            has_prediction_col = 'prediction' in pred_df.columns
            
            # Get the prediction columns we need for the model
            pred_columns = [col for col in pred_df.columns if col.startswith('pred_')]
            
            if not pred_columns:
                return jsonify({'error': 'No prediction columns found in weather data'}), 400
            
            # For each hour in our target range
            for hour_datetime in hourly_times:
                hour_of_day = hour_datetime.hour
                
                # Find weather predictions for this hour of day
                hour_preds = pred_df[pred_df['datetime'].dt.hour == hour_of_day]
                
                # If we have predictions for this hour
                if not hour_preds.empty:
                    # Use prediction column if available to get most relevant forecast
                    if has_prediction_col:
                        # Sort by prediction value (horizon) to get most relevant
                        hour_preds = hour_preds.sort_values('prediction', ascending=False)
                    
                    # Get the best prediction for this hour
                    best_pred = hour_preds.iloc[0]
                    
                    # Copy prediction values to our hourly dataframe
                    for col in pred_columns:
                        hourly_df.loc[hour_datetime, col] = best_pred[col]
                else:
                    # No predictions for this hour, use zeros or average values
                    for col in pred_columns:
                        # Use overall average if available, otherwise 0
                        if len(pred_df) > 0:
                            hourly_df.loc[hour_datetime, col] = pred_df[col].mean()
                        else:
                            hourly_df.loc[hour_datetime, col] = 0
        else:
            # No proper prediction data, use default values
            return jsonify({'error': 'Weather prediction file does not have proper date and time columns'}), 400
        
        # Add time-based features required by the model
        hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df.index.hour / 24)
        hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df.index.hour / 24)
        hourly_df['day_of_year'] = hourly_df.index.dayofyear / 365.0
        
        # Check if all model features are available
        missing_features = [f for f in model_info_prod['features'] if f not in hourly_df.columns]
        if missing_features:
            for feature in missing_features:
                # Add default values for missing features
                hourly_df[feature] = 0
            print(f"Added default values for missing features: {missing_features}")
        
        # Select only features used by the model
        X_pred = hourly_df[model_info_prod['features']]
        
        # Make predictions for each hour
        X_pred_scaled = model_info_prod['scaler'].transform(X_pred)
        predictions = model_info_prod['model'].predict(X_pred_scaled)
        
        # Add predictions to DataFrame
        hourly_df['predicted_pv_output'] = predictions
        # Ensure non-negative predictions
        hourly_df['predicted_pv_output'] = hourly_df['predicted_pv_output'].clip(lower=0)
        
        print(f"Successfully generated {len(hourly_df)} hourly PV output predictions.")
        
        # Format output dataframe with only the requested columns
        output_df = pd.DataFrame()
        output_df['datetime'] = hourly_df.index
        
        # Format date as DD.MM.YYYY
        output_df['date'] = output_df['datetime'].dt.strftime('%d.%m.%Y')
        
        # Format hour as HH:MM
        output_df['hour'] = output_df['datetime'].dt.strftime('%H:%M')
        
        # Round prediction to 2 decimal places
        output_df['productionPrediction'] = hourly_df['predicted_pv_output'].values.round(2)
        
        # Final dataframe with only the requested columns
        final_df = output_df[['date', 'hour', 'productionPrediction']]
        
        
        # Return result with sample data and summary
        return jsonify({
            'status': 'success',
            'message': f'Generated {len(final_df)} hourly predictions for the next {num_days} days',
            'weather_source': latest_pred_file,
            'predictions': final_df.to_dict(orient='records'),
            'summary': {
                'dates_covered': [str(tomorrow + timedelta(days=i)) for i in range(num_days)],
                'total_hours': len(final_df),
                'hours_per_day': 24,
                'days': num_days
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error generating predictions: {str(e)}'}), 500
    
@app.route('/reload-model', methods=['GET'])
def reload_model():
    """Reloads both consumption and production models without restarting the API"""
    global model_info_cons, model_info_prod
    
    result = {"consumption": {}, "production": {}}
    success = True
    
    # Reload consumption model
    try:
        # Try to load the consumption model
        model_info_cons = load("models/consumption_model.joblib")
        
        # Recreate the preprocessing function
        object_columns = model_info_cons.get('object_columns', [])
        
        def preprocess_data(X):
            X_processed = X.copy()
            for col in object_columns:
                if col in X_processed.columns:
                    if X_processed[col].nunique() <= 1:
                        X_processed[col] = 0
                    else:
                        X_processed[col] = X_processed[col].astype('category').cat.codes
            return X_processed
        
        # Add the function to the model_info dictionary
        model_info_cons['preprocess_func'] = preprocess_data
        
        result["consumption"] = {
            'status': 'success',
            'message': 'Consumption model reloaded successfully',
            'model_info': {
                'model_type': model_info_cons.get('model_type', 'unknown'),
                'metrics': model_info_cons.get('metrics', {})
            }
        }
    except Exception as e:
        result["consumption"] = {
            'status': 'error',
            'message': f'Error reloading consumption model: {str(e)}'
        }
        success = False
    
    # Reload production model
    try:
        model_info_prod = pickle.load(open("models/prediction_model.pkl", "rb"))
        result["production"] = {
            'status': 'success',
            'message': 'Production model reloaded successfully'
        }
        if hasattr(model_info_prod, 'get'):
            result["production"]["model_info"] = {
                'features': model_info_prod.get('features', [])[:5] + ['...'],  # Show first 5 features
                'metrics': model_info_prod.get('metrics', {})
            }
    except Exception as e:
        result["production"] = {
            'status': 'error',
            'message': f'Error reloading production model: {str(e)}'
        }
        success = False
    
    return jsonify(result), 200 if success else 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Returns information about the loaded models"""
    result = {"consumption": {}, "production": {}}
    success = True
    
    if model_info_cons is None:
        result["consumption"] = {'status': 'error', 'message': 'Model not loaded'}
        success = False
    else:
        result["consumption"] = {
            'status': 'loaded',
            'model_type': model_info_cons.get('model_type', 'unknown'),
            'metrics': {
                'hourly': model_info_cons.get('metrics', {}),
                'daily': model_info_cons.get('daily_metrics', {})
            },
            'feature_importance': model_info_cons.get('feature_importance', {}).to_dict() 
                if hasattr(model_info_cons.get('feature_importance', {}), 'to_dict') else None
        }
    
    if model_info_prod is None:
        result["production"] = {'status': 'error', 'message': 'Model not loaded'}
        success = False
    else:
        result["production"] = {
            'status': 'loaded',
            'features': model_info_prod.get('features', [])[:5] + ['...'] if hasattr(model_info_prod, 'get') else None,
            'metrics': model_info_prod.get('metrics', {}) if hasattr(model_info_prod, 'get') else None
        }
    
    return jsonify(result), 200 if success else 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple route to check that the API is operational"""
    return jsonify({
        'status': 'ok',
        'models': {
            'consumption': model_info_cons is not None,
            'production': model_info_prod is not None
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/', methods=['GET'])
def home():
    """Home page with API information"""
    return jsonify({
        'name': 'Energy Prediction API',
        'description': 'API for predicting building energy consumption and solar production based on weather forecasts',
        'endpoints': {
            '/predictConsumption': 'GET - Generate consumption predictions for the next 3 days',
            '/predictProduction': 'GET - Generate solar production predictions for the next 3 days',
            '/model-info': 'GET - Get information about the loaded models',
            '/reload-model': 'GET - Reload the models',
            '/health': 'GET - Check the API status',
            '/api/docs': 'GET - Swagger documentation'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)